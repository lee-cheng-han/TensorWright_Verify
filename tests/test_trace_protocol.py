from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from tensorwright.cli import main
from tensorwright.trace import TraceEvent, analyze_protocol_files, write_trace
from tests.test_trace_comparison import _event


def _stream_event(
    value: int,
    coordinate: list[int],
    *,
    sequence: object,
    cycle: int | None,
    valid: object = True,
    ready: object = True,
    last: object = False,
) -> TraceEvent:
    event = _event(
        value,
        coordinate,
        backend="tensorwright.verilator_rtl",
        trace_point="stream_transfer",
        cycle=cycle,
    )
    data = event.to_dict()
    data["metadata"] = {
        "sequence": sequence,
        "valid": valid,
        "ready": ready,
        "tlast": last,
    }
    return TraceEvent.from_dict(data)


class TraceProtocolTest(unittest.TestCase):
    def _write_pair(
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

    def test_valid_stream_passes_protocol_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_pair(
                directory,
                [_event(1, [0, 0, 0, 0]), _event(2, [0, 0, 0, 1])],
                [
                    _stream_event(1, [0, 0, 0, 0], sequence=0, cycle=10),
                    _stream_event(2, [0, 0, 0, 1], sequence=1, cycle=14, last=True),
                ],
            )
            report = analyze_protocol_files(*paths)
        self.assertTrue(report.protocol_ok)
        self.assertEqual(report.findings, [])

    def test_numerical_mismatch_does_not_become_protocol_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_pair(
                directory,
                [_event(1, [0, 0, 0, 0])],
                [_stream_event(9, [0, 0, 0, 0], sequence=0, cycle=10, last=True)],
            )
            report = analyze_protocol_files(*paths)
        self.assertTrue(report.protocol_ok)
        assert report.comparison is not None
        self.assertFalse(report.comparison.matched)

    def test_detects_handshake_sequence_cycle_and_tlast_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_pair(
                directory,
                [_event(1, [0, 0, 0, 0]), _event(2, [0, 0, 0, 1])],
                [
                    _stream_event(
                        1,
                        [0, 0, 0, 0],
                        sequence=3,
                        cycle=10,
                        ready=False,
                        last=True,
                    ),
                    _stream_event(2, [0, 0, 0, 1], sequence=4, cycle=9),
                ],
            )
            report = analyze_protocol_files(*paths)
        rules = {finding.rule_id for finding in report.findings}
        self.assertFalse(report.protocol_ok)
        self.assertTrue(
            {
                "unaccepted_transfer_recorded",
                "transfer_sequence_discontinuity",
                "non_monotonic_transfer_cycle",
                "early_tlast",
                "missing_final_tlast",
            }.issubset(rules)
        )

    def test_correlates_missing_and_unexpected_transfers(self) -> None:
        reference = [_event(1, [0, 0, 0, 0]), _event(2, [0, 0, 0, 1])]
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_pair(
                directory,
                reference,
                [_stream_event(1, [0, 0, 0, 0], sequence=0, cycle=10, last=True)],
            )
            missing = analyze_protocol_files(*paths)
        self.assertIn(
            "missing_output_transfer", {item.rule_id for item in missing.findings}
        )

        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_pair(
                directory,
                [_event(1, [0, 0, 0, 0])],
                [
                    _stream_event(1, [0, 0, 0, 0], sequence=0, cycle=10),
                    _stream_event(2, [0, 0, 0, 1], sequence=1, cycle=11, last=True),
                ],
            )
            unexpected = analyze_protocol_files(*paths)
        self.assertIn(
            "unexpected_output_transfer",
            {item.rule_id for item in unexpected.findings},
        )

    def test_cli_writes_machine_readable_protocol_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_pair(
                directory,
                [_event(1, [0, 0, 0, 0])],
                [_stream_event(1, [0, 0, 0, 0], sequence=0, cycle=10, last=False)],
            )
            report_path = Path(directory) / "protocol.json"
            output = StringIO()
            with redirect_stdout(output):
                status = main(
                    [
                        "trace",
                        "diagnose-protocol",
                        str(paths[0]),
                        str(paths[1]),
                        "--report",
                        str(report_path),
                    ]
                )
            data = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(status, 2)
        self.assertIn("Protocol result: FAIL", output.getvalue())
        self.assertFalse(data["protocol_ok"])


if __name__ == "__main__":
    unittest.main()
