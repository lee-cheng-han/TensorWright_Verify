"""Version-1 canonical JSON Lines trace schema."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TRACE_VERSION = 1
SUPPORTED_BACKENDS = {"python_reference", "cocotb_rtl", "custom_rtl"}


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
    run_id: str
    source_backend: str
    model_id: str
    operation_id: str
    operation_name: str
    operation_type: str
    hardware_stage: str
    tensor_name: str
    coordinate: list[int]
    shape: list[int]
    layout: str
    dtype: str
    value: int | float
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
            "operation_id": self.operation_id,
            "operation_name": self.operation_name,
            "operation_type": self.operation_type,
            "hardware_stage": self.hardware_stage,
            "tensor_name": self.tensor_name,
            "layout": self.layout,
            "dtype": self.dtype,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise TraceError(
                f"Trace event has empty required fields: {', '.join(missing)}"
            )
        if self.source_backend not in SUPPORTED_BACKENDS:
            raise TraceError(f"Unsupported trace source backend: {self.source_backend}")
        if not self.shape or any(value <= 0 for value in self.shape):
            raise TraceError("Trace shape dimensions must be positive")
        if len(self.coordinate) != len(self.shape):
            raise TraceError("Coordinate rank does not match tensor shape")
        if any(
            index < 0 or index >= size
            for index, size in zip(self.coordinate, self.shape, strict=True)
        ):
            raise TraceError("Trace coordinate is outside the tensor shape")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TraceError("Trace value must be numeric")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise TraceError("Trace value must be finite")
        if self.cycle is not None and self.cycle < 0:
            raise TraceError("Trace cycle must be non-negative")
        if self.quantization is not None:
            self.quantization.validate()

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
        except (TypeError, KeyError) as error:
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
    events: list[TraceEvent] = []
    try:
        for line_number, line in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise TraceError(f"Invalid JSON on trace line {line_number}") from error
            if not isinstance(data, dict):
                raise TraceError(f"Trace line {line_number} is not an object")
            events.append(TraceEvent.from_dict(data))
    except OSError as error:
        raise TraceError(f"Could not read trace: {error}") from error
    trace = TraceSet(events)
    trace.validate()
    return trace
