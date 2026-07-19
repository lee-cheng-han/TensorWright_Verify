"""Versioned trace-adapter contract, registry, and built-in adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from tensorwright.trace.adapters.rtl import RtlTraceCapture, read_transfer_log
from tensorwright.trace.schema import (
    BACKEND_PATTERN,
    TRACE_POINTS,
    TRACE_VERSION,
    TraceEvent,
    read_trace,
    write_trace,
)

ADAPTER_API_VERSION = 1
ENTRY_POINT_GROUP = "tensorwright.trace_adapters"
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


class AdapterError(RuntimeError):
    """Raised for malformed, unavailable, or failed trace adapters."""


@dataclass(frozen=True)
class AdapterDescriptor:
    """Stable metadata advertised without executing a conversion."""

    name: str
    version: str
    api_version: int
    input_formats: tuple[str, ...]
    trace_points: tuple[str, ...]
    description: str

    def validate(self) -> None:
        if BACKEND_PATTERN.fullmatch(self.name) is None:
            raise AdapterError(f"Malformed adapter name: {self.name}")
        if VERSION_PATTERN.fullmatch(self.version) is None:
            raise AdapterError(f"Adapter {self.name} has invalid semantic version")
        if self.api_version != ADAPTER_API_VERSION:
            raise AdapterError(
                f"Adapter {self.name} API version {self.api_version} is incompatible "
                f"with {ADAPTER_API_VERSION}"
            )
        if not self.input_formats or any(not value for value in self.input_formats):
            raise AdapterError(f"Adapter {self.name} declares no input formats")
        if not self.trace_points or any(
            value not in TRACE_POINTS for value in self.trace_points
        ):
            raise AdapterError(f"Adapter {self.name} declares invalid trace points")
        if not self.description:
            raise AdapterError(f"Adapter {self.name} has no description")


@dataclass(frozen=True)
class AdapterRequest:
    """One source-to-canonical conversion request."""

    source: Path
    destination: Path
    options: dict[str, Any]


@runtime_checkable
class TraceAdapter(Protocol):
    """Contract implemented by built-in and third-party trace adapters."""

    descriptor: AdapterDescriptor

    def convert(self, request: AdapterRequest) -> Path:
        """Convert a source artifact to validated canonical JSONL."""
        ...


class AdapterRegistry:
    """Deterministic registry with explicit third-party discovery."""

    def __init__(self) -> None:
        self._adapters: dict[str, TraceAdapter] = {}

    def register(self, adapter: TraceAdapter) -> None:
        if not isinstance(adapter, TraceAdapter):
            raise AdapterError("Registered object does not implement TraceAdapter")
        adapter.descriptor.validate()
        name = adapter.descriptor.name
        if name in self._adapters:
            raise AdapterError(f"Duplicate trace adapter: {name}")
        self._adapters[name] = adapter

    def get(self, name: str) -> TraceAdapter:
        try:
            return self._adapters[name]
        except KeyError as error:
            available = ", ".join(self.names()) or "none"
            raise AdapterError(
                f"Unknown trace adapter {name!r}; available: {available}"
            ) from error

    def names(self) -> list[str]:
        return sorted(self._adapters)

    def descriptors(self) -> list[AdapterDescriptor]:
        return [self._adapters[name].descriptor for name in self.names()]

    def discover(self) -> None:
        """Load installed adapters from the documented Python entry-point group."""
        try:
            entry_points = metadata.entry_points(group=ENTRY_POINT_GROUP)
        except Exception as error:
            raise AdapterError(
                f"Could not enumerate adapter entry points: {error}"
            ) from error
        for entry_point in sorted(entry_points, key=lambda item: item.name):
            try:
                loaded = entry_point.load()
                adapter = loaded() if isinstance(loaded, type) else loaded
                self.register(adapter)
            except Exception as error:
                raise AdapterError(
                    f"Could not load trace adapter entry point {entry_point.name!r}: "
                    f"{error}"
                ) from error

    def convert(self, name: str, request: AdapterRequest) -> Path:
        adapter = self.get(name)
        try:
            output = adapter.convert(request)
            trace = read_trace(output)
        except AdapterError:
            raise
        except Exception as error:
            raise AdapterError(f"Adapter {name} conversion failed: {error}") from error
        backend = trace.events[0].source_backend
        if backend != adapter.descriptor.name:
            raise AdapterError(
                f"Adapter {name} emitted backend {backend!r}, expected {name!r}"
            )
        return output


class VerilatorTransferLogAdapter:
    """Built-in converter for TensorWright's compact Verilator transfer log."""

    descriptor = AdapterDescriptor(
        name="tensorwright.verilator_rtl",
        version="1.0.0",
        api_version=ADAPTER_API_VERSION,
        input_formats=("tensorwright-transfer-log-v1",),
        trace_points=("stream_transfer",),
        description="Convert compact accepted-transfer logs to canonical RTL traces.",
    )

    def convert(self, request: AdapterRequest) -> Path:
        allowed = {
            "run_id",
            "model_id",
            "source_operation_id",
            "compiled_operation_id",
            "operation_name",
            "tensor_name",
            "shape",
            "fused_source_operation_ids",
            "graph_stage",
            "operation_type",
            "hardware_stage",
            "layout",
            "dtype",
        }
        unknown = sorted(request.options.keys() - allowed)
        if unknown:
            raise AdapterError(f"Unknown adapter options: {', '.join(unknown)}")
        required = {
            "run_id",
            "model_id",
            "source_operation_id",
            "compiled_operation_id",
            "operation_name",
            "tensor_name",
            "shape",
        }
        missing = sorted(required - request.options.keys())
        if missing:
            raise AdapterError(f"Missing adapter options: {', '.join(missing)}")
        shape = request.options["shape"]
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in shape
            )
        ):
            raise AdapterError("Adapter option shape must be a positive integer array")
        strings = {
            name: _string_option(request.options, name) for name in required - {"shape"}
        }
        fused = request.options.get("fused_source_operation_ids", [])
        if not isinstance(fused, list) or any(
            not isinstance(value, str) or not value for value in fused
        ):
            raise AdapterError("fused_source_operation_ids must be a string array")
        capture = RtlTraceCapture(
            enabled=True,
            shape=shape,
            source_backend=self.descriptor.name,
            fused_source_operation_ids=fused,
            graph_stage=_optional_string(
                request.options, "graph_stage", "rtl_execution"
            ),
            operation_type=_optional_string(request.options, "operation_type", "Conv"),
            hardware_stage=_optional_string(
                request.options, "hardware_stage", "convolution_output_stream"
            ),
            layout=_optional_string(request.options, "layout", "NCHW"),
            dtype=_optional_string(request.options, "dtype", "int8"),
            **strings,
        )
        for transfer in read_transfer_log(request.source):
            capture.record(transfer)
        output = capture.write(request.destination)
        assert output is not None
        return output


class FinnExecutionContextAdapter:
    """Convert FINN's saved full execution context into canonical traces."""

    artifact_label = "FINN execution context"
    default_graph_stage = "finn_execution"
    default_hardware_stage = "finn_operation_output"
    metadata_key = "finn_context_key"
    payload_prefix = "finn"

    descriptor = AdapterDescriptor(
        name="finn.dataflow",
        version="1.0.0",
        api_version=ADAPTER_API_VERSION,
        input_formats=("finn-full-execution-context-npz",),
        trace_points=("operation_output",),
        description="Convert a FINN full execution-context NPZ to canonical traces.",
    )

    def convert(self, request: AdapterRequest) -> Path:
        allowed = {
            "run_id",
            "model_id",
            "graph_stage",
            "scalar_event_limit",
            "tensors",
        }
        unknown = sorted(request.options.keys() - allowed)
        if unknown:
            raise AdapterError(f"Unknown adapter options: {', '.join(unknown)}")
        run_id = _string_option(request.options, "run_id")
        model_id = _string_option(request.options, "model_id")
        graph_stage = _optional_string(
            request.options, "graph_stage", self.default_graph_stage
        )
        limit = request.options.get("scalar_event_limit", 4096)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise AdapterError("scalar_event_limit must be a non-negative integer")
        mappings = request.options.get("tensors")
        if not isinstance(mappings, list) or not mappings:
            raise AdapterError("Adapter option tensors must be a non-empty array")

        try:
            archive = np.load(request.source, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise AdapterError(
                f"Could not read {self.artifact_label}: {error}"
            ) from error

        events: list[TraceEvent] = []
        seen: set[str] = set()
        try:
            for index, raw_mapping in enumerate(mappings):
                mapping = _npz_tensor_mapping(raw_mapping, index)
                tensor_name = mapping["tensor_name"]
                if tensor_name in seen:
                    raise AdapterError(f"Duplicate tensor mapping: {tensor_name}")
                seen.add(tensor_name)
                if tensor_name not in archive.files:
                    raise AdapterError(
                        f"{self.artifact_label} has no tensor {tensor_name!r}"
                    )
                value = np.asarray(archive[tensor_name])
                if value.ndim == 0:
                    value = value.reshape(1)
                if value.size == 0 or any(size <= 0 for size in value.shape):
                    raise AdapterError(f"Tensor {tensor_name!r} is empty")
                if value.dtype.kind not in "iuf":
                    raise AdapterError(
                        f"Tensor {tensor_name!r} has non-numeric dtype {value.dtype}"
                    )
                common: dict[str, Any] = {
                    "trace_version": TRACE_VERSION,
                    "run_id": run_id,
                    "source_backend": self.descriptor.name,
                    "model_id": model_id,
                    "source_operation_id": mapping["source_operation_id"],
                    "compiled_operation_id": mapping["compiled_operation_id"],
                    "fused_source_operation_ids": mapping.get(
                        "fused_source_operation_ids", []
                    ),
                    "graph_stage": graph_stage,
                    "operation_name": mapping["operation_name"],
                    "operation_type": mapping["operation_type"],
                    "hardware_stage": mapping.get(
                        "hardware_stage", self.default_hardware_stage
                    ),
                    "trace_point": "operation_output",
                    "tensor_name": tensor_name,
                    "shape": list(value.shape),
                    "layout": mapping.get("layout", "unknown"),
                    "dtype": str(value.dtype),
                    "metadata": {self.metadata_key: tensor_name},
                }
                if value.size > limit:
                    payload_dir = request.destination.parent / "tensors"
                    payload_dir.mkdir(parents=True, exist_ok=True)
                    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", tensor_name)
                    payload_name = f"{self.payload_prefix}_{index:04d}_{safe_name}.npy"
                    np.save(payload_dir / payload_name, value, allow_pickle=False)
                    events.append(
                        TraceEvent(
                            event_type="tensor_chunk",
                            start_coordinate=[0] * value.ndim,
                            chunk_shape=list(value.shape),
                            data_file=f"tensors/{payload_name}",
                            **common,
                        )
                    )
                else:
                    for coordinate in np.ndindex(value.shape):
                        events.append(
                            TraceEvent(
                                event_type="scalar",
                                coordinate=list(coordinate),
                                value=value[coordinate].item(),
                                **common,
                            )
                        )
        finally:
            archive.close()
        return write_trace(request.destination, events)


class Hls4mlCsimTraceAdapter(FinnExecutionContextAdapter):
    """Convert hls4ml ModelGraph.trace arrays into canonical traces."""

    artifact_label = "hls4ml C-simulation trace"
    default_graph_stage = "hls4ml_optimized_graph"
    default_hardware_stage = "hls4ml_layer_output"
    metadata_key = "hls4ml_trace_key"
    payload_prefix = "hls4ml"
    descriptor = AdapterDescriptor(
        name="hls4ml.csim",
        version="1.0.0",
        api_version=ADAPTER_API_VERSION,
        input_formats=("hls4ml-modelgraph-trace-npz",),
        trace_points=("operation_output",),
        description="Convert hls4ml C-simulation layer traces to canonical traces.",
    )


def default_adapter_registry(*, discover: bool = False) -> AdapterRegistry:
    """Create an isolated registry containing maintained built-in adapters."""
    registry = AdapterRegistry()
    registry.register(FinnExecutionContextAdapter())
    registry.register(Hls4mlCsimTraceAdapter())
    registry.register(VerilatorTransferLogAdapter())
    if discover:
        registry.discover()
    return registry


def _string_option(options: dict[str, Any], name: str) -> str:
    value = options[name]
    if not isinstance(value, str) or not value:
        raise AdapterError(f"Adapter option {name} must be a non-empty string")
    return value


def _optional_string(options: dict[str, Any], name: str, default: str) -> str:
    if name not in options:
        return default
    return _string_option(options, name)


def _npz_tensor_mapping(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AdapterError(f"tensors[{index}] must be an object")
    allowed = {
        "tensor_name",
        "source_operation_id",
        "compiled_operation_id",
        "fused_source_operation_ids",
        "operation_name",
        "operation_type",
        "hardware_stage",
        "layout",
    }
    unknown = sorted(value.keys() - allowed)
    if unknown:
        raise AdapterError(f"Unknown tensors[{index}] options: {', '.join(unknown)}")
    required = {
        "tensor_name",
        "source_operation_id",
        "compiled_operation_id",
        "operation_name",
        "operation_type",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise AdapterError(f"Missing tensors[{index}] options: {', '.join(missing)}")
    for name in required | ({"hardware_stage", "layout"} & value.keys()):
        if not isinstance(value[name], str) or not value[name]:
            raise AdapterError(f"tensors[{index}].{name} must be a non-empty string")
    fused = value.get("fused_source_operation_ids", [])
    if not isinstance(fused, list) or any(
        not isinstance(item, str) or not item for item in fused
    ):
        raise AdapterError(
            f"tensors[{index}].fused_source_operation_ids must be a string array"
        )
    return value
