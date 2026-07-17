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
) -> Path:
    """Execute a graph unchanged and optionally export operation-output values."""
    values = execute_quantized(graph, inputs, capture_all=True)
    events: list[TraceEvent] = []
    for operation_index, operation in enumerate(graph.operations):
        for tensor_name in operation.outputs:
            tensor = graph.tensors[tensor_name]
            value = values[tensor_name]
            quantization = None
            if tensor.quantization_scale is not None:
                scale = tensor.quantization_scale
                quantization = QuantizationMetadata(
                    scale=scale,
                    zero_point=tensor.zero_point,
                    axis=0 if isinstance(scale, list) else None,
                )
            for coordinate in np.ndindex(value.shape):
                scalar = value[coordinate].item()
                events.append(
                    TraceEvent(
                        trace_version=TRACE_VERSION,
                        run_id=run_id,
                        source_backend="python_reference",
                        model_id=graph.name,
                        operation_id=f"op_{operation_index:04d}:{operation.name}",
                        operation_name=operation.name,
                        operation_type=operation.operation_type,
                        hardware_stage="software_operation_output",
                        tensor_name=tensor_name,
                        coordinate=list(coordinate),
                        shape=list(value.shape),
                        layout=tensor.layout,
                        dtype=str(value.dtype),
                        value=scalar,
                        quantization=quantization,
                        metadata={
                            "fused_operations": operation.fused_operations,
                            "assigned_backend": operation.assigned_backend,
                        },
                    )
                )
    return write_trace(output_path, events)
