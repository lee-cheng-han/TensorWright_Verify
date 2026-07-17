from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from tensorwright.cli import main
from tensorwright.dashboard import DashboardError, generate_dashboard
from tensorwright.trace import TraceEvent, write_trace
from tests.test_trace_comparison import _event


def _candidate(value: int, *, model_id: str = "tiny_conv") -> TraceEvent:
    event = _event(
        value,
        [0, 0, 0, 0],
        backend="tensorwright.verilator_rtl",
        trace_point="stream_transfer",
        cycle=12,
    )
    data = event.to_dict()
    data["model_id"] = model_id
    data["metadata"] = {
        "valid": True,
        "ready": True,
        "tlast": True,
        "sequence": 0,
    }
    return TraceEvent.from_dict(data)


def _reference(value: int, *, model_id: str = "tiny_conv") -> TraceEvent:
    event = _event(value, [0, 0, 0, 0])
    data = event.to_dict()
    data["model_id"] = model_id
    return TraceEvent.from_dict(data)


class DashboardTest(unittest.TestCase):
    def _traces(
        self, directory: str, reference: TraceEvent, candidate: TraceEvent
    ) -> tuple[Path, Path]:
        root = Path(directory)
        return (
            write_trace(root / "reference.jsonl", [reference]),
            write_trace(root / "candidate.jsonl", [candidate]),
        )

    def test_renders_divergence_diagnosis_and_protocol_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._traces(directory, _reference(10), _candidate(11))
            output = Path(directory) / "dashboard.html"
            result = generate_dashboard(*paths, output)
            document = output.read_text(encoding="utf-8")
        self.assertFalse(result.comparison.matched)
        self.assertIn("DIVERGENCE", document)
        self.assertIn("requantization_rounding_mismatch", document)
        self.assertIn("Protocol", document)
        self.assertIn("No protocol findings", document)
        self.assertIn("Complete machine-readable report", document)

    def test_renders_match_and_optional_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._traces(directory, _reference(10), _candidate(10))
            minimization = root / "minimal.json"
            regression = root / "manifest.json"
            minimization.write_text(
                json.dumps({"minimized_nonzero_values": 2}), encoding="utf-8"
            )
            regression.write_text(json.dumps({"name": "conv_case"}), encoding="utf-8")
            output = root / "dashboard.html"
            result = generate_dashboard(
                *paths,
                output,
                minimization_report=minimization,
                regression_manifest=regression,
            )
            document = output.read_text(encoding="utf-8")
        self.assertTrue(result.comparison.matched)
        self.assertIn("MATCH", document)
        self.assertIn("Minimization", document)
        self.assertIn("Generated regression", document)

    def test_renders_video_presentation_context_and_tensor_slice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, candidate = self._traces(
                directory, _reference(10), _candidate(11)
            )
            baseline = write_trace(root / "baseline.jsonl", [_candidate(10)])
            regression = root / "test_case.py"
            regression.write_text(
                "def test_case():\n    assert True\n", encoding="utf-8"
            )
            output = root / "dashboard.html"
            generate_dashboard(
                reference,
                candidate,
                output,
                baseline_candidate_trace=baseline,
                scenario_note="One-unit controlled fault",
                arithmetic_evidence={
                    "accumulator": 24,
                    "bias": -491,
                    "biased": -467,
                    "multiplier": 1,
                    "shift": 2,
                    "product": -467,
                    "rounding_offset": 2,
                    "software_result": -117,
                    "rtl_result": -116,
                },
                generated_regression=regression,
            )
            document = output.read_text(encoding="utf-8")
        self.assertIn("Controlled demo fault", document)
        self.assertIn("1/1 values matched", document)
        self.assertIn("SOFTWARE", document)
        self.assertIn("RTL", document)
        self.assertIn("Bounded tensor window", document)
        self.assertIn("Values before first divergence", document)
        self.assertIn("Why this value differs", document)
        self.assertIn("Rounding + shift", document)
        self.assertIn("Recommended fix", document)
        self.assertIn("rounded_magnitude", document)
        self.assertIn("Bug locked into regression", document)

    def test_escapes_untrusted_trace_content(self) -> None:
        malicious = "model<script>alert(1)</script>"
        with tempfile.TemporaryDirectory() as directory:
            paths = self._traces(
                directory,
                _reference(10, model_id=malicious),
                _candidate(10, model_id=malicious),
            )
            output = Path(directory) / "dashboard.html"
            generate_dashboard(*paths, output)
            document = output.read_text(encoding="utf-8")
        self.assertNotIn(malicious, document)
        self.assertIn("&lt;script&gt;", document)

    def test_large_chunk_renders_only_bounded_failure_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = [1, 1, 1024, 1024]
            reference_value = np.zeros(shape, dtype=np.int16)
            candidate_value = reference_value.copy()
            candidate_value[0, 0, 512, 512] = 1
            paths: list[Path] = []
            for name, backend, value in (
                ("reference", "tensorwright.python_reference", reference_value),
                ("candidate", "tensorwright.verilator_rtl", candidate_value),
            ):
                trace_root = root / name
                trace_root.mkdir()
                np.save(trace_root / "payload.npy", value, allow_pickle=False)
                event = _event(
                    0,
                    [0, 0, 0, 0],
                    backend=backend,
                    trace_point=(
                        "operation_output"
                        if name == "reference"
                        else "stream_transfer"
                    ),
                )
                data = event.to_dict()
                data.update(
                    {
                        "event_type": "tensor_chunk",
                        "shape": shape,
                        "dtype": "int16",
                        "value": None,
                        "coordinate": None,
                        "start_coordinate": [0, 0, 0, 0],
                        "chunk_shape": shape,
                        "data_file": "payload.npy",
                    }
                )
                paths.append(
                    write_trace(
                        trace_root / "trace.jsonl", [TraceEvent.from_dict(data)]
                    )
                )
            output = root / "dashboard.html"
            generate_dashboard(paths[0], paths[1], output)
            document = output.read_text(encoding="utf-8")
        self.assertIn("Bounded tensor window", document)
        self.assertIn("[0, 0, 512, 512]", document)
        self.assertIn("Δ +1", document)
        tensor_table = document.split('<table class="tensor-grid">', 1)[1].split(
            "</table>", 1
        )[0]
        self.assertLessEqual(tensor_table.count("<td"), 25)

    def test_generated_source_preview_is_capped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._traces(directory, _reference(10), _candidate(11))
            source = root / "large_test.py"
            source.write_text("x" * 20_000 + "TAIL", encoding="utf-8")
            output = root / "dashboard.html"
            generate_dashboard(*paths, output, generated_regression=source)
            document = output.read_text(encoding="utf-8")
        self.assertIn("preview truncated by TensorWright", document)
        self.assertNotIn("TAIL", document)

    def test_is_deterministic_and_validates_optional_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._traces(directory, _reference(10), _candidate(10))
            first = root / "first.html"
            second = root / "second.html"
            generate_dashboard(*paths, first)
            generate_dashboard(*paths, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            invalid = root / "invalid.json"
            invalid.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(DashboardError, "root"):
                generate_dashboard(
                    *paths, root / "bad.html", minimization_report=invalid
                )
            with self.assertRaisesRegex(DashboardError, "extension"):
                generate_dashboard(*paths, root / "bad.txt")

    def test_cli_exit_codes_follow_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._traces(directory, _reference(10), _candidate(11))
            output = Path(directory) / "dashboard.html"
            stdout = StringIO()
            with redirect_stdout(stdout):
                status = main(["dashboard", str(paths[0]), str(paths[1]), str(output)])
        self.assertEqual(status, 2)
        self.assertIn("Generated TensorWright dashboard", stdout.getvalue())

    def test_cli_rejects_invalid_input(self) -> None:
        error = StringIO()
        with redirect_stderr(error):
            status = main(
                ["dashboard", "missing.jsonl", "missing.jsonl", "report.html"]
            )
        self.assertEqual(status, 1)
        self.assertIn("dashboard generation failed", error.getvalue())


if __name__ == "__main__":
    unittest.main()
