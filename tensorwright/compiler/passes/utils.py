"""Shared graph-transformation utilities."""

from __future__ import annotations

from copy import deepcopy

from tensorwright.compiler.errors import OptimizationError
from tensorwright.compiler.ir import Graph


def copy_graph(graph: Graph) -> Graph:
    """Deep-copy a graph so passes never mutate their caller's object."""
    return deepcopy(graph)


def rebuild_links(graph: Graph) -> None:
    """Recompute all producer-consumer links and validate tensor references."""
    for tensor in graph.tensors.values():
        tensor.producer = None
        tensor.consumers = []
    for operation in graph.operations:
        for input_name in operation.inputs:
            if input_name not in graph.tensors:
                raise OptimizationError(
                    f'Operation "{operation.name}" references missing tensor '
                    f'"{input_name}"'
                )
            graph.tensors[input_name].consumers.append(operation.name)
        for output_name in operation.outputs:
            if output_name not in graph.tensors:
                raise OptimizationError(
                    f'Operation "{operation.name}" produces missing tensor '
                    f'"{output_name}"'
                )
            tensor = graph.tensors[output_name]
            if tensor.producer is not None:
                raise OptimizationError(
                    f'Tensor "{output_name}" has multiple producers'
                )
            tensor.producer = operation.name


def unique_tensor_name(graph: Graph, base: str) -> str:
    """Return a deterministic tensor name not already present in the graph."""
    if base not in graph.tensors:
        return base
    suffix = 1
    while f"{base}_{suffix}" in graph.tensors:
        suffix += 1
    return f"{base}_{suffix}"
