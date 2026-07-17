from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

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
