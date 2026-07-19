"""Trace adapter for the bit-accurate quantized Python reference."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tensorwright.compiler.ir import Graph
from tensorwright.compiler.quantization import execute_quantized
from tensorwright.trace.schema import (
    TRACE_VERSION,
    QuantizationMetadata,
    TraceEvent,
    write_trace,
)


def write_reference_trace(
    graph: Graph,
    inputs: dict[str, np.ndarray],
    output_path: str | Path,
    *,
    run_id: str = "run_000001",
    scalar_event_limit: int = 4096,
    operation_ids: set[str] | None = None,
    tensor_names: set[str] | None = None,
    max_payload_bytes: int | None = None,
) -> Path:
    """Execute a graph and export selected outputs with bounded payload storage."""
    values = execute_quantized(graph, inputs, capture_all=True)
    if scalar_event_limit < 0:
        raise ValueError("scalar_event_limit must be non-negative")
    if max_payload_bytes is not None and max_payload_bytes <= 0:
        raise ValueError("max_payload_bytes must be positive")
    destination = Path(output_path)
    events: list[TraceEvent] = []
    for operation_index, operation in enumerate(graph.operations):
        for tensor_name in operation.outputs:
            source_id = operation.source_operation_id or f"synthetic:{operation.name}"
            compiled_id = f"compiled:op_{operation_index:04d}"
            if operation_ids is not None and not {
                source_id,
                compiled_id,
            }.intersection(operation_ids):
                continue
            if tensor_names is not None and tensor_name not in tensor_names:
                continue
            tensor = graph.tensors[tensor_name]
            value = values[tensor_name]
            if max_payload_bytes is not None and value.nbytes > max_payload_bytes:
                raise ValueError(
                    f"Trace tensor {tensor_name} requires {value.nbytes} bytes, "
                    f"exceeding max_payload_bytes={max_payload_bytes}"
                )
            quantization = None
            if tensor.quantization_scale is not None:
                scale = tensor.quantization_scale
                quantization = QuantizationMetadata(
                    scale=scale,
                    zero_point=tensor.zero_point,
                    axis=0 if isinstance(scale, list) else None,
                )
            common = {
                "trace_version": TRACE_VERSION,
                "run_id": run_id,
                "source_backend": "tensorwright.python_reference",
                "model_id": graph.name,
                "source_operation_id": source_id,
                "compiled_operation_id": compiled_id,
                "fused_source_operation_ids": operation.fused_source_operation_ids,
                "graph_stage": "post_quantization",
                "operation_name": operation.name,
                "operation_type": operation.operation_type,
                "hardware_stage": "software_operation_output",
                "trace_point": "operation_output",
                "tensor_name": tensor_name,
                "shape": list(value.shape),
                "layout": tensor.layout,
                "dtype": str(value.dtype),
                "quantization": quantization,
                "metadata": {
                    "fused_operations": operation.fused_operations,
                    "assigned_backend": operation.assigned_backend,
                },
            }
            if value.size > scalar_event_limit:
                payload_directory = destination.parent / "tensors"
                payload_directory.mkdir(parents=True, exist_ok=True)
                payload_name = (
                    f"op_{operation_index:04d}_{tensor_name.replace('/', '_')}.npy"
                )
                np.save(payload_directory / payload_name, value, allow_pickle=False)
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
    if not events:
        raise ValueError("Trace selection produced no operation outputs")
    return write_trace(destination, events)
