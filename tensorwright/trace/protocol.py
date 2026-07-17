"""Deterministic streaming and protocol diagnosis for canonical RTL traces."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tensorwright.trace.compare import (
    AlignmentError,
    ComparisonReport,
    compare_trace_files,
)
from tensorwright.trace.schema import TraceEvent, read_trace

PROTOCOL_RULESET_VERSION = 1


@dataclass(frozen=True)
class ProtocolFinding:
    """One deterministic protocol-rule finding."""

    rule_id: str
    title: str
    severity: str
    event_index: int | None
    cycle: int | None
    evidence: str
    recommended_check: str


@dataclass(frozen=True)
class ProtocolReport:
    """Protocol findings plus the related semantic comparison, when available."""

    ruleset_version: int
    candidate_backend: str
    model_id: str
    stream_events: int
    findings: list[ProtocolFinding]
    comparison: ComparisonReport | None
    alignment_error: str | None = None

    @property
    def protocol_ok(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["protocol_ok"] = self.protocol_ok
        if self.comparison is not None:
            result["comparison"]["matched"] = self.comparison.matched
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def analyze_protocol_files(
    reference_path: str | Path, candidate_path: str | Path
) -> ProtocolReport:
    """Analyze candidate stream events and correlate transfer-count divergences."""
    candidate = read_trace(candidate_path)
    stream_events = [
        event for event in candidate.events if event.trace_point == "stream_transfer"
    ]
    findings = _inspect_stream_events(stream_events)
    comparison: ComparisonReport | None = None
    alignment_error: str | None = None
    try:
        comparison = compare_trace_files(reference_path, candidate_path)
    except AlignmentError as error:
        alignment_error = str(error)
        findings.append(
            ProtocolFinding(
                "ambiguous_stream_alignment",
                "Stream trace cannot be aligned uniquely",
                "error",
                None,
                None,
                str(error),
                "Check duplicate coordinates, sequence metadata, and source mappings.",
            )
        )
    if comparison is not None and comparison.first_divergence is not None:
        kind = comparison.first_divergence.kind
        if kind == "missing_candidate_value":
            findings.append(
                ProtocolFinding(
                    "missing_output_transfer",
                    "Expected output transfer is missing",
                    "error",
                    comparison.matched_values,
                    comparison.first_divergence.candidate_cycle,
                    "The reference has a semantic coordinate absent from the "
                    "RTL trace.",
                    "Hold each output valid until ready is high, and advance the "
                    "output coordinate only after an accepted handshake.",
                )
            )
        elif kind == "unexpected_candidate_value":
            findings.append(
                ProtocolFinding(
                    "unexpected_output_transfer",
                    "Unexpected output transfer was observed",
                    "error",
                    comparison.matched_values,
                    comparison.first_divergence.candidate_cycle,
                    "The RTL trace has a semantic coordinate absent from the "
                    "reference.",
                    "Advance the output counter exactly once per accepted handshake "
                    "and suppress duplicate emission after packet restart.",
                )
            )
    identity = candidate.events[0]
    return ProtocolReport(
        PROTOCOL_RULESET_VERSION,
        identity.source_backend,
        identity.model_id,
        len(stream_events),
        findings,
        comparison,
        alignment_error,
    )


def _inspect_stream_events(events: list[TraceEvent]) -> list[ProtocolFinding]:
    findings: list[ProtocolFinding] = []
    groups: dict[tuple[str, str], list[tuple[int, TraceEvent]]] = defaultdict(list)
    for event_index, event in enumerate(events):
        groups[(event.source_operation_id, event.tensor_name)].append(
            (event_index, event)
        )
    for group in groups.values():
        previous_cycle: int | None = None
        expected_sequence = 0
        for sequence, (event_index, event) in enumerate(group):
            metadata = event.metadata
            if metadata.get("valid") is not True or metadata.get("ready") is not True:
                findings.append(
                    ProtocolFinding(
                        "unaccepted_transfer_recorded",
                        "Trace contains a transfer without a valid/ready handshake",
                        "error",
                        event_index,
                        event.cycle,
                        f"valid={metadata.get('valid')}, ready={metadata.get('ready')}",
                        "Record stream transfers only on cycles where valid and ready "
                        "are both high.",
                    )
                )
            observed_sequence = metadata.get("sequence")
            if not isinstance(observed_sequence, int) or isinstance(
                observed_sequence, bool
            ):
                findings.append(
                    ProtocolFinding(
                        "missing_sequence_metadata",
                        "Transfer sequence metadata is missing or malformed",
                        "warning",
                        event_index,
                        event.cycle,
                        f"sequence={observed_sequence!r}",
                        "Emit a zero-based accepted-transfer sequence number.",
                    )
                )
            elif observed_sequence != expected_sequence:
                findings.append(
                    ProtocolFinding(
                        "transfer_sequence_discontinuity",
                        "Transfer sequence is discontinuous",
                        "error",
                        event_index,
                        event.cycle,
                        f"expected sequence {expected_sequence}, "
                        f"observed {observed_sequence}",
                        "Increment the sequence counter only on accepted transfers; "
                        "never advance past an output that was not presented.",
                    )
                )
            if isinstance(observed_sequence, int) and not isinstance(
                observed_sequence, bool
            ):
                expected_sequence = observed_sequence + 1
            if event.cycle is None:
                findings.append(
                    ProtocolFinding(
                        "missing_cycle_metadata",
                        "Stream transfer has no simulator cycle",
                        "warning",
                        event_index,
                        None,
                        "cycle is absent",
                        "Capture the simulator cycle at each accepted transfer.",
                    )
                )
            elif previous_cycle is not None and event.cycle <= previous_cycle:
                findings.append(
                    ProtocolFinding(
                        "non_monotonic_transfer_cycle",
                        "Transfer cycles are not strictly increasing",
                        "error",
                        event_index,
                        event.cycle,
                        f"previous cycle {previous_cycle}, current cycle {event.cycle}",
                        "Check monitor sampling order and multiple-counting of a "
                        "handshake.",
                    )
                )
            if event.cycle is not None:
                previous_cycle = event.cycle
            observed_last = metadata.get("tlast")
            expected_last = sequence == len(group) - 1
            if not isinstance(observed_last, bool):
                findings.append(
                    ProtocolFinding(
                        "missing_tlast_metadata",
                        "TLAST metadata is missing or malformed",
                        "warning",
                        event_index,
                        event.cycle,
                        f"tlast={observed_last!r}",
                        "Capture TLAST on every accepted transfer.",
                    )
                )
            elif observed_last != expected_last:
                rule_id = "early_tlast" if observed_last else "missing_final_tlast"
                title = (
                    "TLAST asserted before the final transfer"
                    if observed_last
                    else "Final transfer is missing TLAST"
                )
                findings.append(
                    ProtocolFinding(
                        rule_id,
                        title,
                        "error",
                        event_index,
                        event.cycle,
                        f"sequence {sequence} of {len(group)} has "
                        f"tlast={observed_last}",
                        "Check packet length accounting and final-transfer generation.",
                    )
                )
    return findings
