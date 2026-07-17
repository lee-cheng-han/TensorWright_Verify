"""Batch-normalization folding into convolution parameters."""

from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

from tensorwright.compiler.ir import Graph, Operation
from tensorwright.compiler.passes.utils import (
    copy_graph,
    rebuild_links,
    unique_tensor_name,
)


class FoldBatchNormalization:
    """Fold eligible inference-mode BatchNormalization nodes into Conv."""

    name = "fold_batch_normalization"

    def run(self, graph: Graph) -> Graph:
        result = copy_graph(graph)
        rebuild_links(result)
        operations_by_name = {
            operation.name: operation for operation in result.operations
        }
        removed: set[str] = set()
        for batch_norm in result.operations:
            if batch_norm.operation_type != "BatchNormalization":
                continue
            convolution = self._producer(result, operations_by_name, batch_norm)
            if convolution is None or not self._fold(result, convolution, batch_norm):
                continue
            removed.add(batch_norm.name)
        result.operations = [
            operation
            for operation in result.operations
            if operation.name not in removed
        ]
        rebuild_links(result)
        return result

    @staticmethod
    def _producer(
        graph: Graph,
        operations_by_name: dict[str, Operation],
        batch_norm: Operation,
    ) -> Operation | None:
        if len(batch_norm.inputs) != 5 or len(batch_norm.outputs) != 1:
            return None
        input_tensor = graph.tensors[batch_norm.inputs[0]]
        if input_tensor.producer is None or len(input_tensor.consumers) != 1:
            return None
        if input_tensor.name in graph.outputs:
            return None
        producer = operations_by_name[input_tensor.producer]
        return producer if producer.operation_type == "Conv" else None

    @staticmethod
    def _fold(graph: Graph, convolution: Operation, batch_norm: Operation) -> bool:
        if len(convolution.inputs) not in {2, 3} or len(convolution.outputs) != 1:
            return False
        parameters = [graph.tensors[name] for name in batch_norm.inputs[1:]]
        weights = graph.tensors[convolution.inputs[1]]
        if not weights.is_constant or any(
            not tensor.is_constant for tensor in parameters
        ):
            return False
        try:
            weight_values = np.asarray(weights.constant_data, dtype=np.float64)
            scale, offset, mean, variance = (
                np.asarray(tensor.constant_data, dtype=np.float64)
                for tensor in parameters
            )
            channels = weight_values.shape[0]
            if any(
                value.shape != (channels,) for value in (scale, offset, mean, variance)
            ):
                return False
            if len(convolution.inputs) == 3:
                old_bias_tensor = graph.tensors[convolution.inputs[2]]
                if not old_bias_tensor.is_constant:
                    return False
                old_bias = np.asarray(old_bias_tensor.constant_data, dtype=np.float64)
            else:
                old_bias = np.zeros(channels, dtype=np.float64)
            epsilon_value = batch_norm.attributes.get("epsilon", 1e-5)
            if not isinstance(epsilon_value, (int, float)) or not math.isfinite(
                epsilon_value
            ):
                return False
            denominator = variance + float(epsilon_value)
            values = (weight_values, scale, offset, mean, denominator, old_bias)
            if any(not np.all(np.isfinite(value)) for value in values):
                return False
            if np.any(denominator <= 0):
                return False
            factor = scale / np.sqrt(denominator)
            folded_weights = weight_values * factor.reshape(
                (-1,) + (1,) * (weight_values.ndim - 1)
            )
            folded_bias = (old_bias - mean) * factor + offset
        except (TypeError, ValueError, FloatingPointError):
            return False

        weight_name = unique_tensor_name(graph, f"{convolution.name}__bn_weights")
        bias_name = unique_tensor_name(graph, f"{convolution.name}__bn_bias")
        new_weights = deepcopy(weights)
        new_weights.name = weight_name
        new_weights.constant_data = folded_weights.tolist()
        new_weights.producer = None
        new_weights.consumers = []
        new_bias = deepcopy(parameters[1])
        new_bias.name = bias_name
        new_bias.shape = [int(channels)]
        new_bias.constant_data = folded_bias.tolist()
        new_bias.producer = None
        new_bias.consumers = []
        graph.tensors[weight_name] = new_weights
        graph.tensors[bias_name] = new_bias
        convolution.inputs = [convolution.inputs[0], weight_name, bias_name]
        convolution.outputs = list(batch_norm.outputs)
        convolution.fused_operations.append(batch_norm.name)
        return True
