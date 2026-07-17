"""Canonical metadata-only shape operations."""

from __future__ import annotations

from tensorwright.compiler.ir import Graph
from tensorwright.compiler.passes.utils import copy_graph, rebuild_links


class CanonicalizeShapeOperations:
    """Canonicalize static Flatten and Reshape operations into View."""

    name = "canonicalize_shape_operations"

    def run(self, graph: Graph) -> Graph:
        result = copy_graph(graph)
        for operation in result.operations:
            if operation.operation_type not in {"Flatten", "Reshape"}:
                continue
            if len(operation.outputs) != 1 or not operation.inputs:
                continue
            operation.operation_type = "View"
            operation.inputs = [operation.inputs[0]]
            operation.attributes = {
                "target_shape": list(result.tensors[operation.outputs[0]].shape)
            }
            operation.hardware_supported = False
            operation.assigned_backend = "metadata"
        rebuild_links(result)
        return result
