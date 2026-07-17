"""Deterministic numerical diagnosis rules for first divergences."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from tensorwright.trace.compare import ComparisonReport, Divergence

DIAGNOSIS_RULESET_VERSION = 1


@dataclass(frozen=True)
class Diagnosis:
    """One deterministic likely-cause classification and its evidence."""

    rule_id: str
    title: str
    confidence: str
    evidence: list[str]
    recommended_checks: list[str]


@dataclass(frozen=True)
class DiagnosisReport:
    """A comparison report enriched with a numerical diagnosis."""

    ruleset_version: int
    comparison: ComparisonReport
    diagnosis: Diagnosis | None

    @property
    def matched(self) -> bool:
        return self.comparison.matched

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["comparison"]["matched"] = self.comparison.matched
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def diagnose_comparison(comparison: ComparisonReport) -> DiagnosisReport:
    """Apply the first matching numerical rule without statistical guessing."""
    divergence = comparison.first_divergence
    diagnosis = None if divergence is None else _classify(divergence)
    return DiagnosisReport(DIAGNOSIS_RULESET_VERSION, comparison, diagnosis)


def _classify(divergence: Divergence) -> Diagnosis:
    if divergence.kind != "value_mismatch":
        return Diagnosis(
            rule_id="insufficient_numerical_evidence",
            title="No numerical cause can be assigned",
            confidence="high",
            evidence=[
                f"The first divergence is {divergence.kind}, not unequal values."
            ],
            recommended_checks=[
                "Check trace coverage and semantic mappings.",
                "Use Milestone 15 protocol analysis for missing or extra transfers.",
            ],
        )

    reference = divergence.reference_value
    candidate = divergence.candidate_value
    assert reference is not None and candidate is not None
    delta = candidate - reference
    evidence = [
        f"First unequal value is at {divergence.trace_point}.",
        f"Reference={reference}, candidate={candidate}, delta={delta}.",
    ]

    if divergence.trace_point == "accumulator":
        return Diagnosis(
            "accumulation_arithmetic_mismatch",
            "Accumulator arithmetic mismatch",
            "high",
            evidence,
            [
                "Check signed INT8 multiplication and accumulator width.",
                "Check kernel traversal, channel order, and overflow behavior.",
            ],
        )
    if divergence.trace_point == "post_bias":
        return Diagnosis(
            "bias_application_mismatch",
            "Bias application mismatch",
            "high",
            evidence,
            [
                "Compare the packed INT32 bias for this output channel.",
                "Check bias sign extension and addition width.",
            ],
        )
    if divergence.trace_point == "post_activation":
        return Diagnosis(
            "activation_mismatch",
            "Activation behavior mismatch",
            "high",
            evidence,
            [
                "Check activation enable and operation ordering.",
                "Check the activation boundary and signed comparison.",
            ],
        )
    if (reference in {-128, 127}) != (candidate in {-128, 127}):
        return Diagnosis(
            "saturation_boundary_mismatch",
            "Saturation boundary mismatch",
            "medium",
            [*evidence, "Exactly one value is at a signed INT8 saturation boundary."],
            [
                "Check clamp ordering and signed INT8 limits.",
                "Capture the value immediately before saturation.",
            ],
        )
    if abs(delta) == 1 and divergence.trace_point in {
        "post_requantization",
        "operation_output",
    }:
        return Diagnosis(
            "requantization_rounding_mismatch",
            "Likely requantization rounding mismatch",
            "medium",
            [*evidence, "The outputs differ by exactly one quantized unit."],
            [
                "Compare tie handling and right-shift rounding for negative values.",
                "Compare multiplier, shift, and zero-point application order.",
            ],
        )
    if divergence.trace_point == "post_requantization":
        return Diagnosis(
            "requantization_parameter_mismatch",
            "Likely requantization parameter mismatch",
            "medium",
            evidence,
            [
                "Compare per-channel multiplier, shift, scale, and zero point.",
                "Check intermediate width before the rounding shift.",
            ],
        )
    return Diagnosis(
        "output_numerical_mismatch",
        "Output mismatch needs an earlier trace point",
        "low",
        evidence,
        [
            "Capture accumulator, post-bias, and post-requantization values.",
            "Re-run diagnosis at the earliest unequal internal trace point.",
        ],
    )
