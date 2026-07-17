"""Constant-expression folding."""

from __future__ import annotations

import numpy as np

from tensorwright.compiler.ir import Graph, JsonValue, Operation
from tensorwright.compiler.passes.utils import copy_graph, rebuild_links


class FoldConstants:
    """Fold imported Constant nodes and constant Add operations."""

    name = "fold_constants"

    def run(self, graph: Graph) -> Graph:
        result = copy_graph(graph)
        retained = []
        for operation in result.operations:
            if operation.operation_type == "Constant" and all(
                result.tensors[name].is_constant for name in operation.outputs
            ):
                continue
            if operation.operation_type == "Add" and self._fold_add(result, operation):
                continue
            retained.append(operation)
        result.operations = retained
        rebuild_links(result)
        return result

    @staticmethod
    def _fold_add(graph: Graph, operation: Operation) -> bool:
        if len(operation.inputs) != 2 or len(operation.outputs) != 1:
            return False
        lhs = graph.tensors[operation.inputs[0]]
        rhs = graph.tensors[operation.inputs[1]]
        if not lhs.is_constant or not rhs.is_constant:
            return False
        try:
            value = np.add(np.asarray(lhs.constant_data), np.asarray(rhs.constant_data))
            expected_shape = tuple(graph.tensors[operation.outputs[0]].shape)
            if value.shape != expected_shape:
                value = np.broadcast_to(value, expected_shape)
        except (TypeError, ValueError):
            return False
        output = graph.tensors[operation.outputs[0]]
        output.is_constant = True
        output.constant_data = _json_value(value.tolist())
        return True


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    raise TypeError(f"constant result contains unsupported {type(value).__name__}")
