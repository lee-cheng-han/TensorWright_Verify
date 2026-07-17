"""Deterministic backend assignment."""

from __future__ import annotations

from tensorwright.compiler.ir import Graph
from tensorwright.compiler.passes.utils import copy_graph, rebuild_links

_BACKENDS = {
    "Add": "compiler",
    "BatchNormalization": "compiler",
    "Constant": "compiler",
    "Conv": "fpga",
    "Gemm": "arm",
    "MaxPool": "arm",
    "Relu": "fpga",
    "Softmax": "arm",
    "View": "metadata",
}


class AssignBackends:
    """Annotate every known operation with its planned execution backend."""

    name = "assign_backends"

    def run(self, graph: Graph) -> Graph:
        result = copy_graph(graph)
        for operation in result.operations:
            operation.assigned_backend = _BACKENDS.get(
                operation.operation_type, "unsupported"
            )
            operation.hardware_supported = operation.operation_type in {"Conv", "Relu"}
        rebuild_links(result)
        return result
