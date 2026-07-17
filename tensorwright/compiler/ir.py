"""Typed, serializable intermediate representation owned by TensorWright."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


@dataclass
class Tensor:
    """Tensor metadata used throughout the compiler pipeline."""

    name: str
    shape: list[int]
    original_dtype: str
    compiled_dtype: str
    layout: str
    quantization_scale: float | list[float] | None = None
    zero_point: int = 0
    producer: str | None = None
    consumers: list[str] = field(default_factory=list)
    is_constant: bool = False
    constant_data: JsonValue = None
    memory_region: str | None = None
    memory_offset: int | None = None
    lifetime_start: int | None = None
    lifetime_end: int | None = None


@dataclass
class Operation:
    """One normalized operation in the imported graph."""

    name: str
    operation_type: str
    inputs: list[str]
    outputs: list[str]
    attributes: dict[str, JsonValue]
    hardware_supported: bool
    assigned_backend: str
    fused_operations: list[str] = field(default_factory=list)
    estimated_operation_count: int | None = None


@dataclass
class LayerSchedule:
    """Planned hardware schedule fields; populated by a later milestone."""

    operation_id: str
    input_tile_height: int
    input_tile_width: int
    input_channel_tile: int
    output_channel_tile: int
    output_channel_parallelism: int
    input_buffer: str
    output_buffer: str
    estimated_compute_cycles: int
    estimated_transfer_cycles: int


@dataclass
class Graph:
    """A complete imported graph independent of ONNX protocol objects."""

    name: str
    opset_imports: dict[str, int]
    inputs: list[str]
    outputs: list[str]
    tensors: dict[str, Tensor]
    operations: list[Operation]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""
        return {
            "name": self.name,
            "opset_imports": dict(sorted(self.opset_imports.items())),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "tensors": {
                name: asdict(tensor) for name, tensor in sorted(self.tensors.items())
            },
            "operations": [asdict(operation) for operation in self.operations],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the graph deterministically for diagnostics and tests."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"

    def write_json(self, path: str | Path) -> None:
        """Write serialized IR to a UTF-8 JSON file."""
        Path(path).write_text(self.to_json(), encoding="utf-8")
