from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from tensorwright.compiler import (
    ModelValidationError,
    StaticShapeError,
    UnsupportedOperatorError,
    import_onnx_model,
    load_onnx,
    optimize_graph,
)


def _model(
    nodes: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    initializers: list[onnx.TensorProto] | None = None,
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        "test_graph",
        inputs,
        outputs,
        initializer=initializers or [],
    )
    return helper.make_model(
        graph,
        producer_name="tensorwright-tests",
        opset_imports=[helper.make_opsetid("", 13)],
    )


def _conv_relu_model() -> onnx.ModelProto:
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 2, 2]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1, 2, 2]
    )
    weights = helper.make_tensor("weights", TensorProto.FLOAT, [1, 1, 1, 1], [2.0])
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [1], [1.0])
    nodes = [
        helper.make_node(
            "Conv",
            ["input", "weights", "bias"],
            ["convolution"],
            strides=[1, 1],
        ),
        helper.make_node("Relu", ["convolution"], ["output"]),
    ]
    return _model(nodes, [model_input], [model_output], [weights, bias])


def _mvp_cnn_model() -> onnx.ModelProto:
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 28, 28]
    )
    model_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])
    initializers = [
        helper.make_tensor("conv1_w", TensorProto.FLOAT, [4, 1, 3, 3], [0.0] * 36),
        helper.make_tensor("conv1_b", TensorProto.FLOAT, [4], [0.0] * 4),
        helper.make_tensor("conv2_w", TensorProto.FLOAT, [8, 4, 3, 3], [0.0] * 288),
        helper.make_tensor("conv2_b", TensorProto.FLOAT, [8], [0.0] * 8),
        helper.make_tensor("fc_w", TensorProto.FLOAT, [392, 10], [0.0] * 3920),
        helper.make_tensor("fc_b", TensorProto.FLOAT, [10], [0.0] * 10),
    ]
    nodes = [
        helper.make_node(
            "Conv",
            ["input", "conv1_w", "conv1_b"],
            ["conv1"],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("Relu", ["conv1"], ["relu1"]),
        helper.make_node(
            "MaxPool", ["relu1"], ["pool1"], kernel_shape=[2, 2], strides=[2, 2]
        ),
        helper.make_node(
            "Conv",
            ["pool1", "conv2_w", "conv2_b"],
            ["conv2"],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("Relu", ["conv2"], ["relu2"]),
        helper.make_node(
            "MaxPool", ["relu2"], ["pool2"], kernel_shape=[2, 2], strides=[2, 2]
        ),
        helper.make_node("Flatten", ["pool2"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "fc_w", "fc_b"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["output"], axis=1),
    ]
    return _model(nodes, [model_input], [model_output], initializers)


class OnnxFrontendTest(unittest.TestCase):
    def test_recommended_mvp_cnn_imports_without_node_name_assumptions(self) -> None:
        graph = import_onnx_model(_mvp_cnn_model())

        self.assertEqual(len(graph.operations), 9)
        self.assertEqual(graph.tensors["pool2"].shape, [1, 8, 7, 7])
        self.assertEqual(graph.tensors["flat"].shape, [1, 392])
        self.assertEqual(graph.tensors["output"].shape, [1, 10])
        self.assertEqual(graph.operations[-1].name, "softmax_8")

    def test_recommended_mvp_cnn_optimizes_into_execution_groups(self) -> None:
        optimized = optimize_graph(import_onnx_model(_mvp_cnn_model()))

        self.assertEqual(
            [operation.operation_type for operation in optimized.operations],
            ["Conv", "MaxPool", "Conv", "MaxPool", "View", "Gemm", "Softmax"],
        )
        self.assertEqual(optimized.operations[0].fused_operations, ["relu_1"])
        self.assertEqual(optimized.operations[0].source_operation_id, "onnx:Conv:0")
        self.assertEqual(
            optimized.operations[0].fused_source_operation_ids, ["onnx:Relu:1"]
        )
        self.assertEqual(optimized.operations[2].fused_operations, ["relu_4"])
        self.assertEqual(
            [operation.assigned_backend for operation in optimized.operations],
            ["fpga", "arm", "fpga", "arm", "metadata", "arm", "arm"],
        )

    def test_import_builds_typed_graph_and_relationships(self) -> None:
        graph = import_onnx_model(_conv_relu_model())

        self.assertEqual(graph.name, "test_graph")
        self.assertEqual(graph.opset_imports, {"ai.onnx": 13})
        self.assertEqual(graph.inputs, ["input"])
        self.assertEqual(graph.outputs, ["output"])
        operation_names = [operation.name for operation in graph.operations]
        self.assertEqual(operation_names, ["conv_0", "relu_1"])
        self.assertEqual(graph.operations[0].attributes, {"strides": [1, 1]})
        self.assertTrue(graph.operations[0].hardware_supported)
        self.assertEqual(graph.operations[0].assigned_backend, "fpga")
        self.assertEqual(graph.operations[0].source_operation_id, "onnx:Conv:0")
        self.assertEqual(graph.operations[1].source_operation_id, "onnx:Relu:1")

        self.assertEqual(graph.tensors["input"].shape, [1, 1, 2, 2])
        self.assertEqual(graph.tensors["input"].layout, "NCHW")
        self.assertEqual(graph.tensors["input"].consumers, ["conv_0"])
        self.assertEqual(graph.tensors["convolution"].producer, "conv_0")
        self.assertEqual(graph.tensors["convolution"].consumers, ["relu_1"])
        self.assertEqual(graph.tensors["output"].producer, "relu_1")

        self.assertTrue(graph.tensors["weights"].is_constant)
        self.assertEqual(graph.tensors["weights"].constant_data, [[[[2.0]]]])
        self.assertEqual(graph.tensors["bias"].constant_data, [1.0])

    def test_graph_serialization_is_deterministic(self) -> None:
        graph = import_onnx_model(_conv_relu_model())
        first = graph.to_json()
        second = graph.to_json()

        self.assertEqual(first, second)
        document = json.loads(first)
        self.assertEqual(document["operations"][0]["operation_type"], "Conv")
        self.assertEqual(document["tensors"]["weights"]["shape"], [1, 1, 1, 1])

    def test_duplicate_source_node_names_receive_unique_ir_names(self) -> None:
        model_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
        model_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
        model = _model(
            [
                helper.make_node("Relu", ["input"], ["middle"], name="duplicate"),
                helper.make_node("Relu", ["middle"], ["output"], name="duplicate"),
            ],
            [model_input],
            [model_output],
        )

        graph = import_onnx_model(model)

        self.assertEqual(
            [operation.name for operation in graph.operations],
            ["duplicate", "duplicate_1"],
        )

    def test_loads_model_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.onnx"
            onnx.save(_conv_relu_model(), model_path)
            graph = load_onnx(model_path)

        self.assertEqual(len(graph.operations), 2)

    def test_missing_model_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ModelValidationError, "does not exist"):
            load_onnx("missing-model.onnx")

    def test_constant_node_data_is_extracted(self) -> None:
        model_output = helper.make_tensor_value_info(
            "constant_output", TensorProto.INT64, [2]
        )
        value = helper.make_tensor("value", TensorProto.INT64, [2], [4, 5])
        model = _model(
            [helper.make_node("Constant", [], ["constant_output"], value=value)],
            [],
            [model_output],
        )

        graph = import_onnx_model(model)

        self.assertTrue(graph.tensors["constant_output"].is_constant)
        self.assertEqual(graph.tensors["constant_output"].constant_data, [4, 5])
        self.assertEqual(graph.tensors["constant_output"].producer, "constant_0")

    def test_unsupported_operator_has_node_diagnostic(self) -> None:
        model_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
        model_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
        model = _model(
            [helper.make_node("Sin", ["input"], ["output"], name="sin_node")],
            [model_input],
            [model_output],
        )

        with self.assertRaises(UnsupportedOperatorError) as context:
            import_onnx_model(model)

        self.assertEqual(context.exception.node_name, "sin_node")
        self.assertEqual(context.exception.operation_type, "Sin")
        message = str(context.exception)
        self.assertIn('Compilation failed at node "sin_node"', message)

    def test_dynamic_shape_is_rejected(self) -> None:
        model_input = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, 1, "height", 2]
        )
        model_output = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 1, "height", 2]
        )
        model = _model(
            [helper.make_node("Relu", ["input"], ["output"])],
            [model_input],
            [model_output],
        )

        with self.assertRaisesRegex(StaticShapeError, "positive static shape"):
            import_onnx_model(model)

    def test_invalid_graph_is_wrapped_as_compiler_error(self) -> None:
        model_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
        model = _model(
            [helper.make_node("Relu", ["missing"], ["output"])],
            [],
            [model_output],
        )

        with self.assertRaisesRegex(ModelValidationError, "Invalid ONNX model"):
            import_onnx_model(model)
