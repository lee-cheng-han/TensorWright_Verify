from __future__ import annotations

import math
import unittest

from tensorwright.compiler import Graph, Operation, Tensor, optimize_graph
from tensorwright.compiler.passes import (
    AssignBackends,
    CanonicalizeShapeOperations,
    EliminateDeadCode,
    FoldBatchNormalization,
    FoldConstants,
    FuseConvBiasRelu,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    constant: object = None,
) -> Tensor:
    return Tensor(
        name=name,
        shape=shape,
        original_dtype="float",
        compiled_dtype="float",
        layout="NCHW" if len(shape) == 4 else "UNSPECIFIED",
        is_constant=constant is not None,
        constant_data=constant,  # type: ignore[arg-type]
    )


def _operation(
    name: str,
    operation_type: str,
    inputs: list[str],
    outputs: list[str],
    attributes: dict[str, object] | None = None,
) -> Operation:
    return Operation(
        name=name,
        operation_type=operation_type,
        inputs=inputs,
        outputs=outputs,
        attributes=attributes or {},  # type: ignore[arg-type]
        hardware_supported=operation_type in {"Conv", "Relu"},
        assigned_backend="unassigned",
    )


def _graph(
    tensors: list[Tensor], operations: list[Operation], outputs: list[str]
) -> Graph:
    return Graph(
        name="pass_test",
        opset_imports={"ai.onnx": 13},
        inputs=["input"] if any(tensor.name == "input" for tensor in tensors) else [],
        outputs=outputs,
        tensors={tensor.name: tensor for tensor in tensors},
        operations=operations,
    )


class ConstantFoldingTest(unittest.TestCase):
    def test_constant_add_is_folded_without_mutating_input(self) -> None:
        graph = _graph(
            [
                _tensor("left", [2], constant=[1, 2]),
                _tensor("right", [2], constant=[3, 4]),
                _tensor("sum", [2]),
            ],
            [_operation("add", "Add", ["left", "right"], ["sum"])],
            ["sum"],
        )

        folded = FoldConstants().run(graph)

        self.assertEqual(folded.operations, [])
        self.assertTrue(folded.tensors["sum"].is_constant)
        self.assertEqual(folded.tensors["sum"].constant_data, [4, 6])
        self.assertEqual(len(graph.operations), 1)
        self.assertFalse(graph.tensors["sum"].is_constant)

    def test_constant_node_is_removed_after_frontend_extraction(self) -> None:
        graph = _graph(
            [_tensor("value", [1], constant=[7])],
            [_operation("constant", "Constant", [], ["value"], {"value": [7]})],
            ["value"],
        )

        folded = FoldConstants().run(graph)

        self.assertEqual(folded.operations, [])
        self.assertEqual(folded.tensors["value"].constant_data, [7])


class BatchNormalizationFoldingTest(unittest.TestCase):
    def test_batch_normalization_folds_and_preserves_numeric_result(self) -> None:
        graph = _graph(
            [
                _tensor("input", [1, 1, 1, 1]),
                _tensor("weights", [1, 1, 1, 1], constant=[[[[2.0]]]]),
                _tensor("conv_bias", [1], constant=[3.0]),
                _tensor("conv_output", [1, 1, 1, 1]),
                _tensor("scale", [1], constant=[4.0]),
                _tensor("bn_bias", [1], constant=[5.0]),
                _tensor("mean", [1], constant=[6.0]),
                _tensor("variance", [1], constant=[9.0]),
                _tensor("output", [1, 1, 1, 1]),
            ],
            [
                _operation(
                    "conv",
                    "Conv",
                    ["input", "weights", "conv_bias"],
                    ["conv_output"],
                ),
                _operation(
                    "batch_norm",
                    "BatchNormalization",
                    ["conv_output", "scale", "bn_bias", "mean", "variance"],
                    ["output"],
                    {"epsilon": 0.0},
                ),
            ],
            ["output"],
        )

        folded = FoldBatchNormalization().run(graph)

        self.assertEqual([operation.name for operation in folded.operations], ["conv"])
        convolution = folded.operations[0]
        self.assertEqual(convolution.outputs, ["output"])
        self.assertEqual(convolution.fused_operations, ["batch_norm"])
        folded_weights = folded.tensors[convolution.inputs[1]].constant_data
        folded_bias = folded.tensors[convolution.inputs[2]].constant_data
        self.assertIsInstance(folded_weights, list)
        self.assertIsInstance(folded_bias, list)
        assert isinstance(folded_weights, list)
        assert isinstance(folded_bias, list)

        input_value = 2.0
        original_conv = input_value * 2.0 + 3.0
        original_result = (original_conv - 6.0) * 4.0 / math.sqrt(9.0) + 5.0
        folded_result = input_value * folded_weights[0][0][0][0] + folded_bias[0]  # type: ignore[index,operator]
        self.assertAlmostEqual(folded_result, original_result)

    def test_invalid_variance_is_not_folded(self) -> None:
        graph = _graph(
            [
                _tensor("input", [1, 1, 1, 1]),
                _tensor("weights", [1, 1, 1, 1], constant=[[[[1.0]]]]),
                _tensor("conv_output", [1, 1, 1, 1]),
                _tensor("scale", [1], constant=[1.0]),
                _tensor("offset", [1], constant=[0.0]),
                _tensor("mean", [1], constant=[0.0]),
                _tensor("variance", [1], constant=[-1.0]),
                _tensor("output", [1, 1, 1, 1]),
            ],
            [
                _operation("conv", "Conv", ["input", "weights"], ["conv_output"]),
                _operation(
                    "batch_norm",
                    "BatchNormalization",
                    ["conv_output", "scale", "offset", "mean", "variance"],
                    ["output"],
                    {"epsilon": 0.0},
                ),
            ],
            ["output"],
        )

        result = FoldBatchNormalization().run(graph)

        self.assertEqual(len(result.operations), 2)


class FusionTest(unittest.TestCase):
    def test_conv_bias_relu_fusion_preserves_channel_arithmetic(self) -> None:
        graph = _graph(
            [
                _tensor("input", [1, 2, 1, 1]),
                _tensor("weights", [2, 2, 1, 1], constant=[[[[1]]], [[[1]]]]),
                _tensor("old_bias", [2], constant=[1.0, -1.0]),
                _tensor("conv_output", [1, 2, 1, 1]),
                _tensor("added_bias", [1, 2, 1, 1], constant=[[[[2.0]], [[3.0]]]]),
                _tensor("add_output", [1, 2, 1, 1]),
                _tensor("output", [1, 2, 1, 1]),
            ],
            [
                _operation(
                    "conv",
                    "Conv",
                    ["input", "weights", "old_bias"],
                    ["conv_output"],
                ),
                _operation(
                    "bias_add",
                    "Add",
                    ["conv_output", "added_bias"],
                    ["add_output"],
                ),
                _operation("relu", "Relu", ["add_output"], ["output"]),
            ],
            ["output"],
        )

        fused = FuseConvBiasRelu().run(graph)

        self.assertEqual(len(fused.operations), 1)
        convolution = fused.operations[0]
        self.assertEqual(convolution.outputs, ["output"])
        self.assertEqual(convolution.fused_operations, ["bias_add", "relu"])
        self.assertEqual(convolution.attributes["relu"], True)
        fused_bias = fused.tensors[convolution.inputs[2]].constant_data
        self.assertEqual(fused_bias, [3.0, 2.0])
        assert isinstance(fused_bias, list)
        numeric_bias = [
            float(value) for value in fused_bias if isinstance(value, (int, float))
        ]
        self.assertEqual(len(numeric_bias), 2)
        inputs = [-5.0, 4.0]
        expected = [
            max(0.0, value + bias)
            for value, bias in zip(inputs, [3.0, 2.0], strict=True)
        ]
        actual = [
            max(0.0, value + bias)
            for value, bias in zip(inputs, numeric_bias, strict=True)
        ]
        self.assertEqual(actual, expected)

    def test_fusion_stops_at_shared_intermediate(self) -> None:
        graph = _graph(
            [
                _tensor("input", [1, 1, 1, 1]),
                _tensor("weights", [1, 1, 1, 1], constant=[[[[1.0]]]]),
                _tensor("conv_output", [1, 1, 1, 1]),
                _tensor("relu_output", [1, 1, 1, 1]),
                _tensor("other_output", [1, 1, 1, 1]),
            ],
            [
                _operation("conv", "Conv", ["input", "weights"], ["conv_output"]),
                _operation("relu", "Relu", ["conv_output"], ["relu_output"]),
                _operation("other", "Relu", ["conv_output"], ["other_output"]),
            ],
            ["relu_output", "other_output"],
        )

        fused = FuseConvBiasRelu().run(graph)

        self.assertEqual(len(fused.operations), 3)


class CanonicalizationAndCleanupTest(unittest.TestCase):
    def test_static_reshape_and_flatten_become_metadata_views(self) -> None:
        graph = _graph(
            [
                _tensor("input", [1, 2, 2, 2]),
                _tensor("shape", [2], constant=[1, 8]),
                _tensor("reshaped", [1, 8]),
                _tensor("output", [1, 8]),
            ],
            [
                _operation("reshape", "Reshape", ["input", "shape"], ["reshaped"]),
                _operation("flatten", "Flatten", ["reshaped"], ["output"]),
            ],
            ["output"],
        )

        canonical = CanonicalizeShapeOperations().run(graph)

        self.assertEqual(
            [operation.operation_type for operation in canonical.operations],
            ["View", "View"],
        )
        self.assertEqual(canonical.operations[0].inputs, ["input"])
        self.assertEqual(canonical.operations[0].attributes, {"target_shape": [1, 8]})
        self.assertEqual(canonical.operations[1].assigned_backend, "metadata")

    def test_dead_code_is_removed_by_backward_liveness(self) -> None:
        graph = _graph(
            [
                _tensor("input", [1]),
                _tensor("live", [1]),
                _tensor("output", [1]),
                _tensor("dead_constant", [1], constant=[1]),
                _tensor("dead", [1]),
            ],
            [
                _operation("live_relu", "Relu", ["input"], ["live"]),
                _operation("output_relu", "Relu", ["live"], ["output"]),
                _operation("dead_add", "Add", ["input", "dead_constant"], ["dead"]),
            ],
            ["output"],
        )

        cleaned = EliminateDeadCode().run(graph)

        self.assertEqual(
            [operation.name for operation in cleaned.operations],
            ["live_relu", "output_relu"],
        )
        self.assertNotIn("dead", cleaned.tensors)
        self.assertNotIn("dead_constant", cleaned.tensors)

    def test_live_multi_output_operation_retains_all_output_tensors(self) -> None:
        graph = _graph(
            [
                _tensor("input", [1]),
                _tensor("output", [1]),
                _tensor("auxiliary", [1]),
            ],
            [
                _operation(
                    "multi_output",
                    "CustomMultiOutput",
                    ["input"],
                    ["output", "auxiliary"],
                )
            ],
            ["output"],
        )

        cleaned = EliminateDeadCode().run(graph)

        self.assertIn("auxiliary", cleaned.tensors)
        self.assertEqual(cleaned.tensors["auxiliary"].producer, "multi_output")

    def test_partition_annotations_cover_execution_classes(self) -> None:
        graph = _graph(
            [
                _tensor("input", [1]),
                _tensor("conv", [1]),
                _tensor("view", [1]),
                _tensor("output", [1]),
            ],
            [
                _operation("conv_op", "Conv", ["input"], ["conv"]),
                _operation("view_op", "View", ["conv"], ["view"]),
                _operation("softmax", "Softmax", ["view"], ["output"]),
            ],
            ["output"],
        )

        partitioned = AssignBackends().run(graph)

        self.assertEqual(
            [operation.assigned_backend for operation in partitioned.operations],
            ["fpga", "metadata", "arm"],
        )
        self.assertEqual(
            [operation.hardware_supported for operation in partitioned.operations],
            [True, False, False],
        )

    def test_default_pipeline_fuses_canonicalizes_and_partitions(self) -> None:
        graph = _graph(
            [
                _tensor("input", [1, 1, 1, 1]),
                _tensor("weights", [1, 1, 1, 1], constant=[[[[1.0]]]]),
                _tensor("conv_output", [1, 1, 1, 1]),
                _tensor("relu_output", [1, 1, 1, 1]),
                _tensor("output", [1, 1]),
            ],
            [
                _operation("conv", "Conv", ["input", "weights"], ["conv_output"]),
                _operation("relu", "Relu", ["conv_output"], ["relu_output"]),
                _operation("flatten", "Flatten", ["relu_output"], ["output"]),
            ],
            ["output"],
        )

        optimized = optimize_graph(graph)

        self.assertEqual(
            [operation.operation_type for operation in optimized.operations],
            ["Conv", "View"],
        )
        self.assertEqual(optimized.operations[0].fused_operations, ["relu"])
        self.assertEqual(optimized.operations[0].assigned_backend, "fpga")
        self.assertEqual(optimized.operations[1].assigned_backend, "metadata")
