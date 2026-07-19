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

    chunk_report = _compare_chunk_events(
        reference.events,
        candidate.events,
        reference_source.parent,
        candidate_source.parent,
    )
    if chunk_report is not None:
        return ComparisonReport(
            reference_backend=reference_identity.source_backend,
            candidate_backend=candidate_identity.source_backend,
            model_id=reference_identity.model_id,
            **chunk_report,
        )

    mixed_report = _compare_mixed_events(
        reference.events,
        candidate.events,
        reference_source.parent,
        candidate_source.parent,
    )
    if mixed_report is not None:
        return ComparisonReport(
            reference_backend=reference_identity.source_backend,
            candidate_backend=candidate_identity.source_backend,
            model_id=reference_identity.model_id,
            **mixed_report,
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


def _compare_chunk_events(
    reference_events: list[TraceEvent],
    candidate_events: list[TraceEvent],
    reference_directory: Path,
    candidate_directory: Path,
) -> dict[str, Any] | None:
    all_events = reference_events + candidate_events
    if any(event.event_type != "tensor_chunk" for event in all_events):
        return None
    reference_index = _index_chunks(reference_events)
    candidate_index = _index_chunks(candidate_events)
    ordered_keys = list(reference_index)
    ordered_keys.extend(key for key in candidate_index if key not in reference_index)
    matched_values = 0
    divergence: Divergence | None = None
    for key in ordered_keys:
        reference = reference_index.get(key)
        candidate = candidate_index.get(key)
        if reference is None:
            assert candidate is not None
            actual = _first_chunk_value(candidate, candidate_directory)
            divergence = _divergence("unexpected_candidate_value", None, actual)
            break
        if candidate is None:
            expected = _first_chunk_value(reference, reference_directory)
            divergence = _divergence("missing_candidate_value", expected, None)
            break
        expected_payload = _load_chunk(reference, reference_directory)
        actual_payload = _load_chunk(candidate, candidate_directory)
        if not _metadata_compatible(reference, candidate):
            expected = _array_value(
                reference, expected_payload, (0,) * expected_payload.ndim
            )
            actual = _array_value(candidate, actual_payload, (0,) * actual_payload.ndim)
            divergence = _divergence("metadata_mismatch", expected, actual)
            break
        mismatch = _first_array_mismatch(expected_payload, actual_payload)
        if mismatch is not None:
            matched_values += int(
                np.ravel_multi_index(mismatch, expected_payload.shape)
            )
            expected = _array_value(reference, expected_payload, mismatch)
            actual = _array_value(candidate, actual_payload, mismatch)
            divergence = _divergence("value_mismatch", expected, actual)
            break
        matched_values += expected_payload.size
    return {
        "matched_values": matched_values,
        "reference_values": sum(_chunk_size(event) for event in reference_events),
        "candidate_values": sum(_chunk_size(event) for event in candidate_events),
        "first_divergence": divergence,
    }


def _compare_mixed_events(
    reference_events: list[TraceEvent],
    candidate_events: list[TraceEvent],
    reference_directory: Path,
    candidate_directory: Path,
) -> dict[str, Any] | None:
    """Compare homogeneous chunk/scalar traces without expanding chunk payloads."""
    reference_types = {event.event_type for event in reference_events}
    candidate_types = {event.event_type for event in candidate_events}
    if {frozenset(reference_types), frozenset(candidate_types)} != {
        frozenset({"scalar"}),
        frozenset({"tensor_chunk"}),
    }:
        return None
    chunks_are_reference = reference_types == {"tensor_chunk"}
    chunks = reference_events if chunks_are_reference else candidate_events
    scalars = candidate_events if chunks_are_reference else reference_events
    chunk_directory = (
        reference_directory if chunks_are_reference else candidate_directory
    )
    total_chunk_values = sum(_chunk_size(event) for event in chunks)
    matched_values = 0
    divergence: Divergence | None = None
    for scalar_event in scalars:
        assert scalar_event.coordinate is not None and scalar_event.value is not None
        scalar = AlignedValue(
            scalar_event, tuple(scalar_event.coordinate), scalar_event.value
        )
        chunk_value = _value_from_chunks(chunks, chunk_directory, scalar)
        if chunk_value is None:
            divergence = _divergence(
                "unexpected_candidate_value"
                if chunks_are_reference
                else "missing_candidate_value",
                scalar if not chunks_are_reference else None,
                scalar if chunks_are_reference else None,
            )
            break
        expected = chunk_value if chunks_are_reference else scalar
        actual = scalar if chunks_are_reference else chunk_value
        if not _metadata_compatible(expected.event, actual.event):
            divergence = _divergence("metadata_mismatch", expected, actual)
            break
        if expected.value != actual.value:
            divergence = _divergence("value_mismatch", expected, actual)
            break
        matched_values += 1
    if divergence is None and len(scalars) < total_chunk_values:
        missing = _chunk_value_at_offset(chunks, chunk_directory, len(scalars))
        divergence = _divergence(
            "missing_candidate_value"
            if chunks_are_reference
            else "unexpected_candidate_value",
            missing if chunks_are_reference else None,
            missing if not chunks_are_reference else None,
        )
    return {
        "matched_values": matched_values,
        "reference_values": total_chunk_values
        if chunks_are_reference
        else len(scalars),
        "candidate_values": len(scalars)
        if chunks_are_reference
        else total_chunk_values,
        "first_divergence": divergence,
    }


def _value_from_chunks(
    chunks: list[TraceEvent], directory: Path, scalar: AlignedValue
) -> AlignedValue | None:
    scalar_key = _semantic_key(scalar)[:3]
    for chunk in chunks:
        assert chunk.start_coordinate is not None and chunk.chunk_shape is not None
        probe = AlignedValue(chunk, tuple(chunk.start_coordinate), 0)
        if _semantic_key(probe)[:3] != scalar_key:
            continue
        if all(
            start <= coordinate < start + size
            for coordinate, start, size in zip(
                scalar.coordinate,
                chunk.start_coordinate,
                chunk.chunk_shape,
                strict=True,
            )
        ):
            local = tuple(
                coordinate - start
                for coordinate, start in zip(
                    scalar.coordinate, chunk.start_coordinate, strict=True
                )
            )
            return _array_value(chunk, _load_chunk(chunk, directory), local)
    return None


def _chunk_value_at_offset(
    chunks: list[TraceEvent], directory: Path, offset: int
) -> AlignedValue:
    for chunk in chunks:
        size = _chunk_size(chunk)
        if offset < size:
            assert chunk.chunk_shape is not None
            local = np.unravel_index(offset, tuple(chunk.chunk_shape))
            return _array_value(
                chunk,
                _load_chunk(chunk, directory),
                tuple(int(value) for value in local),
            )
        offset -= size
    raise AlignmentError("Mixed trace offset exceeds chunk payloads")


def _index_chunks(events: list[TraceEvent]) -> dict[SemanticKey, TraceEvent]:
    indexed: dict[SemanticKey, TraceEvent] = {}
    for event in events:
        assert event.start_coordinate is not None
        value = AlignedValue(event, tuple(event.start_coordinate), 0)
        key = _semantic_key(value)
        if key in indexed:
            raise AlignmentError(
                f"Ambiguous duplicate tensor chunk for {event.tensor_name}"
            )
        indexed[key] = event
    return indexed


def _load_chunk(event: TraceEvent, directory: Path) -> np.ndarray[Any, Any]:
    assert event.data_file is not None and event.chunk_shape is not None
    payload = np.load(directory / event.data_file, mmap_mode="r", allow_pickle=False)
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
    return payload


def _first_array_mismatch(
    reference: np.ndarray[Any, Any], candidate: np.ndarray[Any, Any]
) -> tuple[int, ...] | None:
    if reference.shape != candidate.shape:
        return (0,) * reference.ndim
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    block_size = 65_536
    for start in range(0, reference_flat.size, block_size):
        stop = min(reference_flat.size, start + block_size)
        unequal = np.flatnonzero(
            reference_flat[start:stop] != candidate_flat[start:stop]
        )
        if unequal.size:
            coordinate = np.unravel_index(start + int(unequal[0]), reference.shape)
            return tuple(int(index) for index in coordinate)
    return None


def _array_value(
    event: TraceEvent,
    payload: np.ndarray[Any, Any],
    local_coordinate: tuple[int, ...],
) -> AlignedValue:
    assert event.start_coordinate is not None
    coordinate = tuple(
        start + offset
        for start, offset in zip(event.start_coordinate, local_coordinate, strict=True)
    )
    return AlignedValue(event, coordinate, payload[local_coordinate].item())


def _first_chunk_value(event: TraceEvent, directory: Path) -> AlignedValue:
    payload = _load_chunk(event, directory)
    return _array_value(event, payload, (0,) * payload.ndim)


def _chunk_size(event: TraceEvent) -> int:
    assert event.chunk_shape is not None
    return int(np.prod(event.chunk_shape))


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
