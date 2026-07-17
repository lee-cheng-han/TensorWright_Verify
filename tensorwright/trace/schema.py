"""Version-2 canonical JSON Lines trace schema."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TRACE_VERSION = 2
BACKEND_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_-]*)+$")
TRACE_POINTS = {
    "operation_input",
    "accumulator",
    "post_bias",
    "post_requantization",
    "post_activation",
    "operation_output",
    "stream_transfer",
}


class TraceError(ValueError):
    """Raised when canonical trace data is invalid or incompatible."""


@dataclass(frozen=True)
class QuantizationMetadata:
    scale: float | list[float]
    zero_point: int = 0
    axis: int | None = None

    def validate(self) -> None:
        scales = self.scale if isinstance(self.scale, list) else [self.scale]
        if not scales or any(
            not math.isfinite(value) or value <= 0 for value in scales
        ):
            raise TraceError("Quantization scales must be finite and positive")
        if self.axis is not None and self.axis < 0:
            raise TraceError("Quantization axis must be non-negative")


@dataclass(frozen=True)
class TraceEvent:
    trace_version: int
    event_type: str
    run_id: str
    source_backend: str
    model_id: str
    source_operation_id: str
    compiled_operation_id: str
    fused_source_operation_ids: list[str]
    graph_stage: str
    operation_name: str
    operation_type: str
    hardware_stage: str
    trace_point: str
    tensor_name: str
    shape: list[int]
    layout: str
    dtype: str
    value: int | float | None = None
    coordinate: list[int] | None = None
    start_coordinate: list[int] | None = None
    chunk_shape: list[int] | None = None
    data_file: str | None = None
    cycle: int | None = None
    quantization: QuantizationMetadata | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.trace_version != TRACE_VERSION:
            raise TraceError(f"Unsupported trace version: {self.trace_version}")
        required = {
            "run_id": self.run_id,
            "source_backend": self.source_backend,
            "model_id": self.model_id,
            "source_operation_id": self.source_operation_id,
            "compiled_operation_id": self.compiled_operation_id,
            "graph_stage": self.graph_stage,
            "operation_name": self.operation_name,
            "operation_type": self.operation_type,
            "hardware_stage": self.hardware_stage,
            "trace_point": self.trace_point,
            "tensor_name": self.tensor_name,
            "layout": self.layout,
            "dtype": self.dtype,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise TraceError(
                f"Trace event has empty required fields: {', '.join(missing)}"
            )
        if BACKEND_PATTERN.fullmatch(self.source_backend) is None:
            raise TraceError(f"Malformed trace source backend: {self.source_backend}")
        if self.trace_point not in TRACE_POINTS:
            raise TraceError(f"Unsupported trace point: {self.trace_point}")
        if not self.shape or any(value <= 0 for value in self.shape):
            raise TraceError("Trace shape dimensions must be positive")
        if self.event_type == "scalar":
            self._validate_scalar()
        elif self.event_type == "tensor_chunk":
            self._validate_chunk()
        else:
            raise TraceError(f"Unsupported trace event type: {self.event_type}")
        if self.cycle is not None and self.cycle < 0:
            raise TraceError("Trace cycle must be non-negative")
        if self.quantization is not None:
            self.quantization.validate()

    def _validate_scalar(self) -> None:
        if self.coordinate is None or len(self.coordinate) != len(self.shape):
            raise TraceError("Coordinate rank does not match tensor shape")
        if any(
            index < 0 or index >= size
            for index, size in zip(self.coordinate, self.shape, strict=True)
        ):
            raise TraceError("Trace coordinate is outside the tensor shape")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TraceError("Scalar trace value must be numeric")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise TraceError("Trace value must be finite")
        if any(
            value is not None
            for value in (self.start_coordinate, self.chunk_shape, self.data_file)
        ):
            raise TraceError("Scalar event contains tensor-chunk fields")

    def _validate_chunk(self) -> None:
        if self.value is not None or self.coordinate is not None:
            raise TraceError("Tensor-chunk event contains scalar fields")
        if (
            self.start_coordinate is None
            or self.chunk_shape is None
            or not self.data_file
        ):
            raise TraceError("Tensor-chunk event is missing payload fields")
        if len(self.start_coordinate) != len(self.shape) or len(
            self.chunk_shape
        ) != len(self.shape):
            raise TraceError("Tensor-chunk rank does not match tensor shape")
        if any(size <= 0 for size in self.chunk_shape):
            raise TraceError("Tensor-chunk dimensions must be positive")
        if any(
            start < 0 or start + size > total
            for start, size, total in zip(
                self.start_coordinate, self.chunk_shape, self.shape, strict=True
            )
        ):
            raise TraceError("Tensor chunk is outside the tensor shape")
        payload = Path(self.data_file)
        if payload.is_absolute() or ".." in payload.parts:
            raise TraceError("Tensor payload path must be relative and contained")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceEvent:
        try:
            value = dict(data)
            quantization = value.get("quantization")
            if quantization is not None:
                value["quantization"] = QuantizationMetadata(**quantization)
            event = cls(**value)
        except TypeError as error:
            raise TraceError(f"Invalid trace event structure: {error}") from error
        event.validate()
        return event


@dataclass(frozen=True)
class TraceSet:
    events: list[TraceEvent]

    def validate(self) -> None:
        if not self.events:
            raise TraceError("Trace contains no events")
        for event in self.events:
            event.validate()
        identities = {
            (event.run_id, event.source_backend, event.model_id)
            for event in self.events
        }
        if len(identities) != 1:
            raise TraceError("Trace events do not share one run, backend, and model")


def write_trace(path: str | Path, events: list[TraceEvent]) -> Path:
    trace = TraceSet(events)
    trace.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(event.to_dict(), sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    return destination


def read_trace(path: str | Path) -> TraceSet:
    source = Path(path)
    events: list[TraceEvent] = []
    try:
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise TraceError(f"Invalid JSON on trace line {line_number}") from error
            if not isinstance(data, dict):
                raise TraceError(f"Trace line {line_number} is not an object")
            event = TraceEvent.from_dict(data)
            if event.event_type == "tensor_chunk":
                payload = source.parent / str(event.data_file)
                if not payload.is_file():
                    raise TraceError(f"Missing tensor payload: {event.data_file}")
            events.append(event)
    except OSError as error:
        raise TraceError(f"Could not read trace: {error}") from error
    trace = TraceSet(events)
    trace.validate()
    return trace
