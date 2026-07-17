"""Dead operation and tensor elimination."""

from __future__ import annotations

from tensorwright.compiler.ir import Graph
from tensorwright.compiler.passes.utils import copy_graph, rebuild_links


class EliminateDeadCode:
    """Remove pure operations and tensors not reachable from graph outputs."""

    name = "eliminate_dead_code"

    def run(self, graph: Graph) -> Graph:
        result = copy_graph(graph)
        live_tensors = set(result.outputs)
        live_operations: set[str] = set()
        for operation in reversed(result.operations):
            if any(output in live_tensors for output in operation.outputs):
                live_operations.add(operation.name)
                live_tensors.update(operation.inputs)
                live_tensors.update(operation.outputs)
        result.operations = [
            operation
            for operation in result.operations
            if operation.name in live_operations
        ]
        live_tensors.update(result.inputs)
        result.tensors = {
            name: tensor
            for name, tensor in result.tensors.items()
            if name in live_tensors
        }
        rebuild_links(result)
        return result
