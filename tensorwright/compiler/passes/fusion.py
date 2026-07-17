"""Convolution, bias, and activation fusion."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from tensorwright.compiler.ir import Graph, Operation
from tensorwright.compiler.passes.utils import (
    copy_graph,
    rebuild_links,
    unique_tensor_name,
)


class FuseConvBiasRelu:
    """Fuse constant channel bias additions and single-consumer ReLUs into Conv."""

    name = "fuse_conv_bias_relu"

    def run(self, graph: Graph) -> Graph:
        result = copy_graph(graph)
        rebuild_links(result)
        for convolution in list(result.operations):
            if convolution.operation_type != "Conv":
                continue
            add = self._single_consumer(result, convolution, "Add")
            if add is not None and self._fuse_bias(result, convolution, add):
                result.operations.remove(add)
                rebuild_links(result)
            relu = self._single_consumer(result, convolution, "Relu")
            if relu is not None:
                convolution.outputs = list(relu.outputs)
                convolution.attributes["relu"] = True
                convolution.fused_operations.append(relu.name)
                if relu.source_operation_id is not None:
                    convolution.fused_source_operation_ids.append(
                        relu.source_operation_id
                    )
                result.operations.remove(relu)
                rebuild_links(result)
        return result

    @staticmethod
    def _single_consumer(
        graph: Graph, producer: Operation, operation_type: str
    ) -> Operation | None:
        if len(producer.outputs) != 1 or producer.outputs[0] in graph.outputs:
            return None
        tensor = graph.tensors[producer.outputs[0]]
        if len(tensor.consumers) != 1:
            return None
        consumer_name = tensor.consumers[0]
        consumer = next(
            operation
            for operation in graph.operations
            if operation.name == consumer_name
        )
        return consumer if consumer.operation_type == operation_type else None

    @staticmethod
    def _fuse_bias(graph: Graph, convolution: Operation, add: Operation) -> bool:
        if len(add.inputs) != 2 or len(add.outputs) != 1:
            return False
        conv_output = convolution.outputs[0]
        bias_inputs = [name for name in add.inputs if name != conv_output]
        if len(bias_inputs) != 1:
            return False
        added_bias_tensor = graph.tensors[bias_inputs[0]]
        if not added_bias_tensor.is_constant:
            return False
        output_shape = graph.tensors[add.outputs[0]].shape
        if len(output_shape) != 4:
            return False
        output_channels = output_shape[1]
        added_bias = _channel_bias(added_bias_tensor.constant_data, output_channels)
        if added_bias is None:
            return False
        if len(convolution.inputs) == 3:
            original_bias_tensor = graph.tensors[convolution.inputs[2]]
            if not original_bias_tensor.is_constant:
                return False
            original_bias = _channel_bias(
                original_bias_tensor.constant_data, output_channels
            )
            if original_bias is None:
                return False
            fused_bias = original_bias + added_bias
            template = original_bias_tensor
        elif len(convolution.inputs) == 2:
            fused_bias = added_bias
            template = added_bias_tensor
        else:
            return False

        bias_name = unique_tensor_name(graph, f"{convolution.name}__fused_bias")
        new_bias = deepcopy(template)
        new_bias.name = bias_name
        new_bias.shape = [output_channels]
        new_bias.constant_data = fused_bias.tolist()
        new_bias.producer = None
        new_bias.consumers = []
        graph.tensors[bias_name] = new_bias
        convolution.inputs = [convolution.inputs[0], convolution.inputs[1], bias_name]
        convolution.outputs = list(add.outputs)
        convolution.fused_operations.append(add.name)
        if add.source_operation_id is not None:
            convolution.fused_source_operation_ids.append(add.source_operation_id)
        return True


def _channel_bias(value: object, channels: int) -> np.ndarray[object, object] | None:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.shape == (channels,):
        return array
    if array.shape == (1, channels, 1, 1):
        return array.reshape(channels)
    return None
