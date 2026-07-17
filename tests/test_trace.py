from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from tensorwright.cli import main
from tensorwright.trace import (
    TRACE_VERSION,
    QuantizationMetadata,
    TraceError,
    TraceEvent,
    read_trace,
    write_reference_trace,
    write_trace,
)
from tests.test_deployment_bundle import _compilation_result


def _event(**changes: object) -> TraceEvent:
    values = {
        "trace_version": TRACE_VERSION,
        "event_type": "scalar",
        "run_id": "run_test",
        "source_backend": "tensorwright.python_reference",
        "model_id": "model",
        "source_operation_id": "onnx:Conv_3",
        "compiled_operation_id": "compiled:op_0000",
        "fused_source_operation_ids": ["onnx:Relu_4"],
        "graph_stage": "post_quantization",
        "operation_name": "conv",
        "operation_type": "Conv",
        "hardware_stage": "software_operation_output",
        "trace_point": "operation_output",
        "tensor_name": "output",
        "coordinate": [0, 0, 0, 0],
        "shape": [1, 1, 1, 1],
        "layout": "NCHW",
        "dtype": "int8",
        "value": -5,
        "cycle": None,
        "quantization": QuantizationMetadata(1.0),
        "metadata": {},
    }
    values.update(changes)
    return TraceEvent(**values)  # type: ignore[arg-type]


class TraceSchemaTest(unittest.TestCase):
    def test_round_trip_preserves_typed_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_trace(Path(directory) / "trace.jsonl", [_event()])
            loaded = read_trace(path)
        self.assertEqual(loaded.events, [_event()])

    def test_rejects_version_and_coordinate_errors(self) -> None:
        with self.assertRaisesRegex(TraceError, "Unsupported trace version"):
            _event(trace_version=1).validate()
        with self.assertRaisesRegex(TraceError, "rank"):
            _event(coordinate=[0]).validate()
        with self.assertRaisesRegex(TraceError, "outside"):
            _event(coordinate=[0, 0, 0, 1]).validate()

    def test_reference_writer_records_stable_operation_output(self) -> None:
        result = _compilation_result()
        inputs = {"input": np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)}
        with tempfile.TemporaryDirectory() as directory:
            path = write_reference_trace(
                result.graph,
                inputs,
                Path(directory) / "reference.jsonl",
                run_id="run_42",
            )
            event = read_trace(path).events[0]
        self.assertEqual(event.run_id, "run_42")
        self.assertEqual(event.compiled_operation_id, "compiled:op_0000")
        self.assertEqual(event.source_operation_id, "synthetic:conv")
        self.assertEqual(event.coordinate, [0, 0, 0, 0])
        self.assertEqual(event.value, -5)
        self.assertEqual(event.quantization, QuantizationMetadata(1.0))

    def test_reference_writer_uses_npy_payload_for_large_tensor(self) -> None:
        result = _compilation_result()
        inputs = {"input": np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)}
        with tempfile.TemporaryDirectory() as directory:
            path = write_reference_trace(
                result.graph,
                inputs,
                Path(directory) / "reference.jsonl",
                scalar_event_limit=0,
            )
            event = read_trace(path).events[0]
            payload = np.load(
                Path(directory) / str(event.data_file), allow_pickle=False
            )
        self.assertEqual(event.event_type, "tensor_chunk")
        self.assertEqual(event.trace_point, "operation_output")
        self.assertEqual(event.start_coordinate, [0, 0, 0, 0])
        self.assertEqual(payload.tolist(), [[[[-5]]]])

    def test_backend_names_are_extensible_but_well_formed(self) -> None:
        _event(source_backend="custom.my_adapter").validate()
        with self.assertRaisesRegex(TraceError, "Malformed"):
            _event(source_backend="unknown").validate()

    def test_cli_trace_inspection_reports_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_trace(Path(directory) / "trace.jsonl", [_event()])
            output = StringIO()
            with redirect_stdout(output):
                status = main(["trace", "inspect", str(path)])
        self.assertEqual(status, 0)
        self.assertIn("Trace version: 2", output.getvalue())
        self.assertIn("Events: 1", output.getvalue())
        self.assertIn("Cycle information: no", output.getvalue())
        self.assertIn("Quantization metadata: yes", output.getvalue())

    def test_cli_trace_inspection_rejects_invalid_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            path.write_text("{}\n")
            error = StringIO()
            with redirect_stderr(error):
                status = main(["trace", "inspect", str(path)])
        self.assertEqual(status, 1)
        self.assertIn("trace inspection failed", error.getvalue())


if __name__ == "__main__":
    unittest.main()
