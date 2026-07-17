"""Semantic alignment and first-divergence detection for canonical traces."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tensorwright.trace.schema import TraceError, TraceEvent, read_trace


class AlignmentError(TraceError):
    """Raised when two traces cannot be aligned without ambiguity."""


@dataclass(frozen=True)
class AlignedValue:
    """One scalar value normalized from a scalar or chunk event."""

    event: TraceEvent
    coordinate: tuple[int, ...]
    value: int | float


@dataclass(frozen=True)
class Divergence:
    """The earliest semantic mismatch between two traces."""

    kind: str
    source_operation_id: str
    compiled_operation_id: str
    tensor_name: str
    trace_point: str
    coordinate: list[int]
    reference_value: int | float | None
    candidate_value: int | float | None
    reference_cycle: int | None
    candidate_cycle: int | None


@dataclass(frozen=True)
class ComparisonReport:
    """Machine-readable result of a deterministic trace comparison."""

    reference_backend: str
    candidate_backend: str
    model_id: str
    matched_values: int
    reference_values: int
    candidate_values: int
    first_divergence: Divergence | None

    @property
    def matched(self) -> bool:
        return self.first_divergence is None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["matched"] = self.matched
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


SemanticKey = tuple[tuple[str, ...], str, str, tuple[int, ...]]


def compare_trace_files(
    reference_path: str | Path, candidate_path: str | Path
) -> ComparisonReport:
    """Align two trace files semantically and return their first divergence."""
    reference_source = Path(reference_path)
    candidate_source = Path(candidate_path)
    reference = read_trace(reference_source)
    candidate = read_trace(candidate_source)
    reference_identity = reference.events[0]
    candidate_identity = candidate.events[0]
    if reference_identity.model_id != candidate_identity.model_id:
        raise AlignmentError(
            "Trace model IDs differ: "
            f"{reference_identity.model_id} != {candidate_identity.model_id}"
        )

    reference_values = _index_values(reference.events, reference_source.parent)
    candidate_values = _index_values(candidate.events, candidate_source.parent)
    ordered_keys = list(reference_values)
    ordered_keys.extend(key for key in candidate_values if key not in reference_values)
    matched_values = 0
    divergence: Divergence | None = None
    for key in ordered_keys:
        expected = reference_values.get(key)
        actual = candidate_values.get(key)
        if expected is None:
            divergence = _divergence("unexpected_candidate_value", None, actual)
            break
        if actual is None:
            divergence = _divergence("missing_candidate_value", expected, None)
            break
        if not _metadata_compatible(expected.event, actual.event):
            divergence = _divergence("metadata_mismatch", expected, actual)
            break
        if expected.value != actual.value:
            divergence = _divergence("value_mismatch", expected, actual)
            break
        matched_values += 1

    return ComparisonReport(
        reference_backend=reference_identity.source_backend,
        candidate_backend=candidate_identity.source_backend,
        model_id=reference_identity.model_id,
        matched_values=matched_values,
        reference_values=len(reference_values),
        candidate_values=len(candidate_values),
        first_divergence=divergence,
    )


def _index_values(
    events: list[TraceEvent], payload_directory: Path
) -> dict[SemanticKey, AlignedValue]:
    indexed: dict[SemanticKey, AlignedValue] = {}
    for event in events:
        for value in _expand_event(event, payload_directory):
            key = _semantic_key(value)
            if key in indexed:
                coordinate = list(value.coordinate)
                raise AlignmentError(
                    "Ambiguous duplicate trace value for "
                    f"{event.source_operation_id}/{event.tensor_name} at {coordinate}"
                )
            indexed[key] = value
    return indexed


def _expand_event(event: TraceEvent, payload_directory: Path) -> list[AlignedValue]:
    if event.event_type == "scalar":
        assert event.coordinate is not None and event.value is not None
        return [AlignedValue(event, tuple(event.coordinate), event.value)]
    assert event.data_file is not None
    assert event.start_coordinate is not None
    assert event.chunk_shape is not None
    payload = np.load(payload_directory / event.data_file, allow_pickle=False)
    if list(payload.shape) != event.chunk_shape:
        raise AlignmentError(
            f"Tensor payload shape {list(payload.shape)} does not match "
            f"declared chunk shape {event.chunk_shape}"
        )
    if str(payload.dtype) != event.dtype:
        raise AlignmentError(
            f"Tensor payload dtype {payload.dtype} does not match declared dtype "
            f"{event.dtype}"
        )
    values: list[AlignedValue] = []
    for local_coordinate in np.ndindex(payload.shape):
        coordinate = tuple(
            start + offset
            for start, offset in zip(
                event.start_coordinate, local_coordinate, strict=True
            )
        )
        values.append(AlignedValue(event, coordinate, payload[local_coordinate].item()))
    return values


def _semantic_key(value: AlignedValue) -> SemanticKey:
    event = value.event
    lineage = (event.source_operation_id,)
    trace_point = (
        "operation_output"
        if event.trace_point == "stream_transfer"
        else event.trace_point
    )
    return lineage, event.tensor_name, trace_point, value.coordinate


def _metadata_compatible(reference: TraceEvent, candidate: TraceEvent) -> bool:
    return (
        reference.shape == candidate.shape
        and reference.layout == candidate.layout
        and reference.dtype == candidate.dtype
        and reference.operation_type == candidate.operation_type
    )


def _divergence(
    kind: str,
    reference: AlignedValue | None,
    candidate: AlignedValue | None,
) -> Divergence:
    selected = reference or candidate
    assert selected is not None
    reference_event = reference.event if reference is not None else None
    candidate_event = candidate.event if candidate is not None else None
    event = reference_event or candidate_event
    assert event is not None
    return Divergence(
        kind=kind,
        source_operation_id=event.source_operation_id,
        compiled_operation_id=event.compiled_operation_id,
        tensor_name=event.tensor_name,
        trace_point=(
            "operation_output"
            if event.trace_point == "stream_transfer"
            else event.trace_point
        ),
        coordinate=list(selected.coordinate),
        reference_value=reference.value if reference is not None else None,
        candidate_value=candidate.value if candidate is not None else None,
        reference_cycle=reference_event.cycle if reference_event is not None else None,
        candidate_cycle=candidate_event.cycle if candidate_event is not None else None,
    )
