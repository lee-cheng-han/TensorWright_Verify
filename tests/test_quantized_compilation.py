from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper

from tensorwright.compiler import (
    Graph,
    QuantizationError,
    compile_quantized,
    execute_quantized,
    import_onnx_model,
    optimize_graph,
)


def _small_classifier_graph() -> Graph:
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 4, 4]
    )
    model_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
    initializers = [
        helper.make_tensor("conv_w", TensorProto.FLOAT, [1, 1, 1, 1], [0.75]),
        helper.make_tensor("conv_b", TensorProto.FLOAT, [1], [0.1]),
        helper.make_tensor(
            "fc_w",
            TensorProto.FLOAT,
            [4, 2],
            [0.4, -0.3, 0.2, 0.1, -0.5, 0.6, 0.7, -0.2],
        ),
        helper.make_tensor("fc_b", TensorProto.FLOAT, [2], [0.05, -0.1]),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "conv_w", "conv_b"], ["conv"]),
        helper.make_node("Relu", ["conv"], ["relu"]),
        helper.make_node(
            "MaxPool", ["relu"], ["pool"], kernel_shape=[2, 2], strides=[2, 2]
        ),
        helper.make_node("Flatten", ["pool"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "fc_w", "fc_b"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["output"], axis=1),
    ]
    graph = helper.make_graph(
        nodes, "small_classifier", [model_input], [model_output], initializers
    )
    model = helper.make_model(
        graph,
        producer_name="tensorwright-tests",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    return optimize_graph(import_onnx_model(model))


class QuantizedCompilationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = _small_classifier_graph()
        self.samples = [
            {"input": np.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4)},
            {"input": np.linspace(1.0, -0.5, 16).reshape(1, 1, 4, 4)},
        ]

    def test_compiles_and_executes_complete_mixed_backend_path(self) -> None:
        result = compile_quantized(self.graph, self.samples)
        output = execute_quantized(result.graph, self.samples[0])["output"]

        self.assertEqual(output.shape, (1, 2))
        self.assertAlmostEqual(float(np.sum(output)), 1.0)
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertEqual(result.graph.tensors["input"].compiled_dtype, "int8")
        self.assertEqual(result.graph.tensors["logits"].compiled_dtype, "int8")
        self.assertEqual(result.graph.tensors["output"].compiled_dtype, "float32")

        linear_operations = [
            operation
            for operation in result.graph.operations
            if operation.operation_type in {"Conv", "Gemm"}
        ]
        for operation in linear_operations:
            self.assertIn("requantization_multipliers", operation.attributes)
            self.assertIn("requantization_shifts", operation.attributes)
            self.assertEqual(
                result.graph.tensors[operation.inputs[1]].compiled_dtype, "int8"
            )
            self.assertEqual(
                result.graph.tensors[operation.inputs[2]].compiled_dtype, "int32"
            )
            multipliers = operation.attributes["requantization_multipliers"]
            assert isinstance(multipliers, list)
            integer_multipliers = [
                value
                for value in multipliers
                if isinstance(value, int) and not isinstance(value, bool)
            ]
            self.assertEqual(len(integer_multipliers), len(multipliers))
            self.assertTrue(
                all(0 <= value < (1 << 31) for value in integer_multipliers)
            )

    def test_report_contains_measured_deterministic_comparison(self) -> None:
        first = compile_quantized(self.graph, self.samples, labels=[0, 1])
        second = compile_quantized(self.graph, self.samples, labels=[0, 1])

        self.assertEqual(first.report, second.report)
        self.assertEqual(json.loads(first.report_json()), first.report)
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "compilation_report.json"
            first.write_report(report_path)
            self.assertEqual(json.loads(report_path.read_text()), first.report)
        self.assertEqual(first.report["calibration_sample_count"], 2)
        comparison = first.report["comparison"]
        self.assertGreaterEqual(comparison["max_absolute_error"], 0.0)
        self.assertGreaterEqual(comparison["mean_absolute_error"], 0.0)
        self.assertGreaterEqual(comparison["top1_agreement"], 0.0)
        self.assertLessEqual(comparison["top1_agreement"], 1.0)
        self.assertGreaterEqual(comparison["float_top1_accuracy"], 0.0)
        self.assertLessEqual(comparison["quantized_top1_accuracy"], 1.0)

    def test_calibration_range_and_scale_are_observed(self) -> None:
        result = compile_quantized(self.graph, self.samples)
        input_range = result.report["calibration_ranges"]["input"]

        self.assertEqual(input_range["minimum"], -1.0)
        self.assertEqual(input_range["maximum"], 1.0)
        self.assertAlmostEqual(input_range["scale"], 1.0 / 127.0)

    def test_empty_nonfinite_and_wrong_shape_samples_fail(self) -> None:
        with self.assertRaisesRegex(QuantizationError, "calibration sample"):
            compile_quantized(self.graph, [])
        with self.assertRaisesRegex(QuantizationError, "Labels"):
            compile_quantized(self.graph, self.samples, labels=[0])
        with self.assertRaisesRegex(QuantizationError, "not finite"):
            compile_quantized(self.graph, [{"input": np.full((1, 1, 4, 4), np.nan)}])
        with self.assertRaisesRegex(QuantizationError, "expected"):
            compile_quantized(self.graph, [{"input": np.zeros((1, 1, 2, 2))}])
