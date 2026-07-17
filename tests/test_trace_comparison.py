from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from tensorwright.cli import main
from tensorwright.trace import (
    TRACE_VERSION,
    AlignmentError,
    TraceEvent,
    compare_trace_files,
    write_trace,
)


def _event(
    value: int,
    coordinate: list[int],
    *,
    backend: str = "tensorwright.python_reference",
    trace_point: str = "operation_output",
    cycle: int | None = None,
    shape: list[int] | None = None,
) -> TraceEvent:
    return TraceEvent(
        trace_version=TRACE_VERSION,
        event_type="scalar",
        run_id="run_reference" if "python" in backend else "run_rtl",
        source_backend=backend,
        model_id="tiny_conv",
        source_operation_id="onnx:Conv_0",
        compiled_operation_id="compiled:op_0000",
        fused_source_operation_ids=["onnx:Relu_1"],
        graph_stage="post_quantization",
        operation_name="conv_0",
        operation_type="Conv",
        hardware_stage="output",
        trace_point=trace_point,
        tensor_name="output",
        shape=shape or [1, 1, 1, 2],
        layout="NCHW",
        dtype="int8",
        value=value,
        coordinate=coordinate,
        cycle=cycle,
    )


class TraceComparisonTest(unittest.TestCase):
    def _paths(
        self,
        directory: str,
        reference: list[TraceEvent],
        candidate: list[TraceEvent],
    ) -> tuple[Path, Path]:
        root = Path(directory)
        return (
            write_trace(root / "reference.jsonl", reference),
            write_trace(root / "candidate.jsonl", candidate),
        )

    def test_aligns_operation_output_with_stream_transfer(self) -> None:
        reference = [_event(-3, [0, 0, 0, 0]), _event(4, [0, 0, 0, 1])]
        candidate = [
            _event(
                -3,
                [0, 0, 0, 0],
                backend="tensorwright.verilator_rtl",
                trace_point="stream_transfer",
                cycle=41,
            ),
            _event(
                4,
                [0, 0, 0, 1],
                backend="tensorwright.verilator_rtl",
                trace_point="stream_transfer",
                cycle=45,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(directory, reference, candidate)
            report = compare_trace_files(*paths)
        self.assertTrue(report.matched)
        self.assertEqual(report.matched_values, 2)

    def test_reports_first_value_divergence_with_cycle(self) -> None:
        reference = [_event(1, [0, 0, 0, 0]), _event(2, [0, 0, 0, 1])]
        candidate = [
            _event(
                1,
                [0, 0, 0, 0],
                backend="tensorwright.cocotb_rtl",
                trace_point="stream_transfer",
                cycle=10,
            ),
            _event(
                9,
                [0, 0, 0, 1],
                backend="tensorwright.cocotb_rtl",
                trace_point="stream_transfer",
                cycle=14,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            report = compare_trace_files(*self._paths(directory, reference, candidate))
        self.assertFalse(report.matched)
        self.assertEqual(report.matched_values, 1)
        assert report.first_divergence is not None
        self.assertEqual(report.first_divergence.kind, "value_mismatch")
        self.assertEqual(report.first_divergence.coordinate, [0, 0, 0, 1])
        self.assertEqual(report.first_divergence.reference_value, 2)
        self.assertEqual(report.first_divergence.candidate_value, 9)
        self.assertEqual(report.first_divergence.candidate_cycle, 14)

    def test_reports_missing_candidate_value(self) -> None:
        reference = [_event(1, [0, 0, 0, 0]), _event(2, [0, 0, 0, 1])]
        candidate = [
            _event(
                1,
                [0, 0, 0, 0],
                backend="tensorwright.verilator_rtl",
                trace_point="stream_transfer",
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            report = compare_trace_files(*self._paths(directory, reference, candidate))
        assert report.first_divergence is not None
        self.assertEqual(report.first_divergence.kind, "missing_candidate_value")

    def test_reports_unexpected_candidate_and_metadata_mismatch(self) -> None:
        reference = [_event(1, [0, 0, 0, 0])]
        candidate = [
            _event(1, [0, 0, 0, 0], backend="custom.rtl"),
            _event(2, [0, 0, 0, 1], backend="custom.rtl"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            report = compare_trace_files(*self._paths(directory, reference, candidate))
        assert report.first_divergence is not None
        self.assertEqual(report.first_divergence.kind, "unexpected_candidate_value")

        with tempfile.TemporaryDirectory() as directory:
            report = compare_trace_files(
                *self._paths(
                    directory,
                    reference,
                    [
                        _event(
                            1,
                            [0, 0, 0, 0],
                            backend="custom.rtl",
                            shape=[1, 1, 1, 3],
                        )
                    ],
                )
            )
        assert report.first_divergence is not None
        self.assertEqual(report.first_divergence.kind, "metadata_mismatch")

    def test_expands_reference_tensor_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tensors").mkdir()
            np.save(root / "tensors" / "output.npy", np.array([[[[3, 4]]]], np.int8))
            scalar = _event(0, [0, 0, 0, 0])
            chunk_data = scalar.to_dict()
            chunk_data.update(
                event_type="tensor_chunk",
                value=None,
                coordinate=None,
                start_coordinate=[0, 0, 0, 0],
                chunk_shape=[1, 1, 1, 2],
                data_file="tensors/output.npy",
            )
            chunk = TraceEvent.from_dict(chunk_data)
            reference_path = write_trace(root / "reference.jsonl", [chunk])
            candidate_path = write_trace(
                root / "candidate.jsonl",
                [
                    _event(3, [0, 0, 0, 0], backend="custom.rtl"),
                    _event(4, [0, 0, 0, 1], backend="custom.rtl"),
                ],
            )
            report = compare_trace_files(reference_path, candidate_path)
        self.assertTrue(report.matched)

    def test_rejects_ambiguous_and_different_model_traces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference_path, candidate_path = self._paths(
                directory,
                [_event(1, [0, 0, 0, 0])],
                [_event(1, [0, 0, 0, 0]), _event(1, [0, 0, 0, 0])],
            )
            with self.assertRaisesRegex(AlignmentError, "Ambiguous"):
                compare_trace_files(reference_path, candidate_path)

            different = _event(1, [0, 0, 0, 0], backend="custom.rtl")
            different_data = different.to_dict()
            different_data["model_id"] = "other_model"
            write_trace(candidate_path, [TraceEvent.from_dict(different_data)])
            with self.assertRaisesRegex(AlignmentError, "model IDs"):
                compare_trace_files(reference_path, candidate_path)

    def test_cli_human_and_json_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(
                directory,
                [_event(1, [0, 0, 0, 0])],
                [_event(2, [0, 0, 0, 0], backend="custom.rtl", cycle=7)],
            )
            output = StringIO()
            report_path = Path(directory) / "report.json"
            with redirect_stdout(output):
                status = main(
                    [
                        "trace",
                        "compare",
                        str(paths[0]),
                        str(paths[1]),
                        "--report",
                        str(report_path),
                    ]
                )
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(status, 2)
        self.assertIn("Result: DIVERGENCE", output.getvalue())
        self.assertIn("Candidate cycle: 7", output.getvalue())
        self.assertFalse(report_data["matched"])

    def test_cli_rejects_unalignable_traces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(
                directory,
                [_event(1, [0, 0, 0, 0])],
                [_event(1, [0, 0, 0, 0], backend="custom.rtl")],
            )
            candidate = json.loads(paths[1].read_text(encoding="utf-8"))
            candidate["model_id"] = "other"
            paths[1].write_text(json.dumps(candidate) + "\n", encoding="utf-8")
            error = StringIO()
            with redirect_stderr(error):
                status = main(["trace", "compare", str(paths[0]), str(paths[1])])
        self.assertEqual(status, 1)
        self.assertIn("trace comparison failed", error.getvalue())


if __name__ == "__main__":
    unittest.main()
