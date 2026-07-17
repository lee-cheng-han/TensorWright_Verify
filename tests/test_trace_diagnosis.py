from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from tensorwright.cli import main
from tensorwright.trace import (
    ComparisonReport,
    Divergence,
    diagnose_comparison,
    write_trace,
)
from tests.test_trace_comparison import _event


def _comparison(
    *,
    kind: str = "value_mismatch",
    trace_point: str = "operation_output",
    reference: int | None = 10,
    candidate: int | None = 12,
) -> ComparisonReport:
    return ComparisonReport(
        reference_backend="tensorwright.python_reference",
        candidate_backend="tensorwright.verilator_rtl",
        model_id="tiny_conv",
        matched_values=3,
        reference_values=4,
        candidate_values=4,
        first_divergence=Divergence(
            kind=kind,
            source_operation_id="onnx:Conv_0",
            compiled_operation_id="compiled:op_0000",
            tensor_name="output",
            trace_point=trace_point,
            coordinate=[0, 0, 0, 3],
            reference_value=reference,
            candidate_value=candidate,
            reference_cycle=None,
            candidate_cycle=42,
        ),
    )


class TraceDiagnosisTest(unittest.TestCase):
    def test_stage_rules_are_deterministic(self) -> None:
        cases = [
            ("accumulator", 10, 12, "accumulation_arithmetic_mismatch", "high"),
            ("post_bias", 10, 12, "bias_application_mismatch", "high"),
            ("post_activation", 10, -2, "activation_mismatch", "high"),
            (
                "post_requantization",
                10,
                11,
                "requantization_rounding_mismatch",
                "medium",
            ),
            (
                "post_requantization",
                10,
                20,
                "requantization_parameter_mismatch",
                "medium",
            ),
            (
                "operation_output",
                126,
                127,
                "saturation_boundary_mismatch",
                "medium",
            ),
            (
                "operation_output",
                10,
                20,
                "output_numerical_mismatch",
                "low",
            ),
        ]
        for trace_point, reference, candidate, rule, confidence in cases:
            with self.subTest(rule=rule):
                report = diagnose_comparison(
                    _comparison(
                        trace_point=trace_point,
                        reference=reference,
                        candidate=candidate,
                    )
                )
                assert report.diagnosis is not None
                self.assertEqual(report.diagnosis.rule_id, rule)
                self.assertEqual(report.diagnosis.confidence, confidence)

    def test_structural_divergence_does_not_claim_numerical_cause(self) -> None:
        report = diagnose_comparison(
            _comparison(kind="missing_candidate_value", reference=10, candidate=None)
        )
        assert report.diagnosis is not None
        self.assertEqual(report.diagnosis.rule_id, "insufficient_numerical_evidence")
        self.assertIn(
            "automatically generated protocol findings",
            report.diagnosis.recommended_checks[1],
        )

    def test_matching_comparison_needs_no_diagnosis(self) -> None:
        comparison = ComparisonReport(
            "tensorwright.python_reference",
            "tensorwright.verilator_rtl",
            "tiny_conv",
            4,
            4,
            4,
            None,
        )
        report = diagnose_comparison(comparison)
        self.assertTrue(report.matched)
        self.assertIsNone(report.diagnosis)

    def test_cli_prints_and_writes_diagnosis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = write_trace(
                root / "reference.jsonl", [_event(10, [0, 0, 0, 0])]
            )
            candidate_path = write_trace(
                root / "candidate.jsonl",
                [_event(11, [0, 0, 0, 0], backend="custom.rtl", cycle=9)],
            )
            report_path = root / "diagnosis.json"
            output = StringIO()
            with redirect_stdout(output):
                status = main(
                    [
                        "trace",
                        "diagnose",
                        str(reference_path),
                        str(candidate_path),
                        "--report",
                        str(report_path),
                    ]
                )
            data = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(status, 2)
        self.assertIn(
            "Likely cause: Likely requantization rounding mismatch", output.getvalue()
        )
        self.assertEqual(
            data["diagnosis"]["rule_id"], "requantization_rounding_mismatch"
        )


if __name__ == "__main__":
    unittest.main()
