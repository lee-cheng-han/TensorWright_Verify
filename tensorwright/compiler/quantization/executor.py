"""Float and quantized software execution for the optimized MVP graph."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from tensorwright.compiler.errors import QuantizationError
from tensorwright.compiler.ir import Graph, JsonValue, Operation
from tensorwright.reference import Conv2DConfig, conv2d_int8, requantize_int32

ArrayMap = dict[str, np.ndarray]


def execute_float(
    graph: Graph, inputs: Mapping[str, np.ndarray], *, capture_all: bool = False
) -> ArrayMap:
    """Execute an optimized graph in FP32 for calibration and comparison."""
    values = _initial_values(graph, inputs, quantized=False)
    for operation in graph.operations:
        operation_inputs = [values[name] for name in operation.inputs]
        values.update(_execute_float_operation(graph, operation, operation_inputs))
    names = values.keys() if capture_all else graph.outputs
    return {name: values[name].copy() for name in names}


def execute_quantized(
    graph: Graph, inputs: Mapping[str, np.ndarray], *, capture_all: bool = False
) -> ArrayMap:
    """Execute a quantized graph with integer Conv/Gemm and explicit ARM fallbacks."""
    values = _initial_values(graph, inputs, quantized=True)
    for operation in graph.operations:
        operation_inputs = [values[name] for name in operation.inputs]
        values.update(_execute_quantized_operation(graph, operation, operation_inputs))
    names = values.keys() if capture_all else graph.outputs
    return {name: values[name].copy() for name in names}


def _initial_values(
    graph: Graph, inputs: Mapping[str, np.ndarray], *, quantized: bool
) -> ArrayMap:
    values: ArrayMap = {}
    for name, tensor in graph.tensors.items():
        if tensor.is_constant and tensor.constant_data is not None:
            dtype = np.int32 if tensor.compiled_dtype == "int32" else None
            if tensor.compiled_dtype == "int8":
                dtype = np.int8
            values[name] = np.asarray(tensor.constant_data, dtype=dtype)
    for name in graph.inputs:
        if name not in inputs:
            raise QuantizationError(f'Missing model input "{name}"')
        value = np.asarray(inputs[name])
        expected_shape = tuple(graph.tensors[name].shape)
        if value.shape != expected_shape:
            raise QuantizationError(
                f'Input "{name}" has shape {value.shape}, expected {expected_shape}'
            )
        if quantized:
            scale = _tensor_scale(graph, name)
            value = _quantize_array(value, scale)
        else:
            value = value.astype(np.float32)
        values[name] = value
    return values


def _execute_float_operation(
    graph: Graph, operation: Operation, inputs: list[np.ndarray]
) -> ArrayMap:
    if operation.operation_type == "Conv":
        output = _conv_float(inputs, operation)
    elif operation.operation_type == "MaxPool":
        output = _max_pool(inputs[0], operation)
    elif operation.operation_type == "View":
        output = inputs[0].reshape(graph.tensors[operation.outputs[0]].shape)
    elif operation.operation_type == "Gemm":
        output = _gemm_float(inputs, operation)
    elif operation.operation_type == "Softmax":
        output = _softmax(inputs[0], operation)
    elif operation.operation_type == "Relu":
        output = np.maximum(inputs[0], 0)
    else:
        raise QuantizationError(
            f'Float execution does not support operation "{operation.operation_type}"'
        )
    return {operation.outputs[0]: output}


def _execute_quantized_operation(
    graph: Graph, operation: Operation, inputs: list[np.ndarray]
) -> ArrayMap:
    if operation.operation_type == "Conv":
        output = _conv_quantized(inputs, operation)
    elif operation.operation_type == "Gemm":
        output = _gemm_quantized(inputs, operation)
    elif operation.operation_type == "MaxPool":
        output = _max_pool(inputs[0], operation).astype(np.int8)
    elif operation.operation_type == "View":
        output = inputs[0].reshape(graph.tensors[operation.outputs[0]].shape)
    elif operation.operation_type == "Softmax":
        input_scale = _tensor_scale(graph, operation.inputs[0])
        output = _softmax(inputs[0].astype(np.float64) * input_scale, operation)
    else:
        operation_type = operation.operation_type
        raise QuantizationError(
            f'Quantized execution does not support operation "{operation_type}"'
        )
    return {operation.outputs[0]: output}


def _conv_float(inputs: list[np.ndarray], operation: Operation) -> np.ndarray:
    activations, weights = inputs[:2]
    bias = inputs[2] if len(inputs) == 3 else np.zeros(weights.shape[0])
    strides, pads = _spatial_attributes(operation)
    top, left, bottom, right = pads
    padded = np.pad(activations, ((0, 0), (0, 0), (top, bottom), (left, right)))
    batch, _, height, width = padded.shape
    output_channels, input_channels, kernel_height, kernel_width = weights.shape
    output_height = (height - kernel_height) // strides[0] + 1
    output_width = (width - kernel_width) // strides[1] + 1
    output = np.empty(
        (batch, output_channels, output_height, output_width), dtype=np.float32
    )
    for n in range(batch):
        for oc in range(output_channels):
            for y in range(output_height):
                for x in range(output_width):
                    window = padded[
                        n,
                        :input_channels,
                        y * strides[0] : y * strides[0] + kernel_height,
                        x * strides[1] : x * strides[1] + kernel_width,
                    ]
                    output[n, oc, y, x] = np.sum(window * weights[oc]) + bias[oc]
    if operation.attributes.get("relu") is True:
        output = np.maximum(output, 0)
    return output


def _conv_quantized(inputs: list[np.ndarray], operation: Operation) -> np.ndarray:
    activations, weights = inputs[:2]
    bias = inputs[2] if len(inputs) == 3 else np.zeros(weights.shape[0], dtype=np.int32)
    if activations.shape[0] != 1:
        raise QuantizationError("Quantized Conv supports batch size one")
    strides, pads = _spatial_attributes(operation)
    top, left, bottom, right = pads
    multipliers = _integer_list(operation.attributes, "requantization_multipliers")
    shifts = _integer_list(operation.attributes, "requantization_shifts")
    output = conv2d_int8(
        activations[0].astype(np.int8).tolist(),
        weights.astype(np.int8).tolist(),
        bias.astype(np.int32).tolist(),
        multipliers,
        shifts,
        Conv2DConfig(
            stride_height=strides[0],
            stride_width=strides[1],
            padding=(top, bottom, left, right),
            relu=operation.attributes.get("relu") is True,
        ),
    )
    return np.asarray([output], dtype=np.int8)


def _gemm_float(inputs: list[np.ndarray], operation: Operation) -> np.ndarray:
    _validate_gemm_attributes(operation)
    output = inputs[0] @ inputs[1]
    if len(inputs) == 3:
        output = output + inputs[2]
    return output


def _gemm_quantized(inputs: list[np.ndarray], operation: Operation) -> np.ndarray:
    _validate_gemm_attributes(operation)
    activations = inputs[0].astype(np.int64)
    weights = inputs[1].astype(np.int64)
    accumulator = activations @ weights
    if len(inputs) == 3:
        accumulator = accumulator + inputs[2].astype(np.int64)
    if np.any(accumulator < -(1 << 31)) or np.any(accumulator > (1 << 31) - 1):
        raise QuantizationError("Gemm accumulator exceeds signed INT32")
    multipliers = _integer_list(operation.attributes, "requantization_multipliers")
    shifts = _integer_list(operation.attributes, "requantization_shifts")
    output = np.empty(accumulator.shape, dtype=np.int8)
    for row in range(accumulator.shape[0]):
        for channel in range(accumulator.shape[1]):
            output[row, channel] = requantize_int32(
                int(accumulator[row, channel]),
                0,
                multipliers[channel],
                shifts[channel],
            )
    return output


def _max_pool(value: np.ndarray, operation: Operation) -> np.ndarray:
    kernel = _integer_attribute(operation.attributes, "kernel_shape", required=True)
    strides = _integer_attribute(operation.attributes, "strides", default=kernel)
    pads = _integer_attribute(operation.attributes, "pads", default=[0, 0, 0, 0])
    if len(kernel) != 2 or len(strides) != 2 or len(pads) != 4:
        raise QuantizationError("MaxPool requires 2D kernel, strides, and padding")
    top, left, bottom, right = pads
    minimum = -128 if np.issubdtype(value.dtype, np.integer) else -np.inf
    padded = np.pad(
        value, ((0, 0), (0, 0), (top, bottom), (left, right)), constant_values=minimum
    )
    output_height = (padded.shape[2] - kernel[0]) // strides[0] + 1
    output_width = (padded.shape[3] - kernel[1]) // strides[1] + 1
    output = np.empty(
        (value.shape[0], value.shape[1], output_height, output_width),
        dtype=value.dtype,
    )
    for y in range(output_height):
        for x in range(output_width):
            window = padded[
                :,
                :,
                y * strides[0] : y * strides[0] + kernel[0],
                x * strides[1] : x * strides[1] + kernel[1],
            ]
            output[:, :, y, x] = np.max(window, axis=(2, 3))
    return output.astype(value.dtype)


def _softmax(value: np.ndarray, operation: Operation) -> np.ndarray:
    axis_value = operation.attributes.get("axis", -1)
    if not isinstance(axis_value, int):
        raise QuantizationError("Softmax axis must be an integer")
    shifted = value - np.max(value, axis=axis_value, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=axis_value, keepdims=True)


def _spatial_attributes(operation: Operation) -> tuple[list[int], list[int]]:
    strides = _integer_attribute(operation.attributes, "strides", default=[1, 1])
    pads = _integer_attribute(operation.attributes, "pads", default=[0, 0, 0, 0])
    if len(strides) != 2 or len(pads) != 4:
        raise QuantizationError("Conv requires 2D strides and padding")
    return strides, pads


def _validate_gemm_attributes(operation: Operation) -> None:
    supported = {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}
    if any(
        operation.attributes.get(name, value) != value
        for name, value in supported.items()
    ):
        raise QuantizationError("Gemm supports only default alpha, beta, and transpose")


def _integer_attribute(
    attributes: Mapping[str, JsonValue],
    name: str,
    *,
    default: list[int] | None = None,
    required: bool = False,
) -> list[int]:
    value = attributes.get(name)
    if value is None:
        if required or default is None:
            raise QuantizationError(f'Missing integer-list attribute "{name}"')
        return default
    if not isinstance(value, list):
        raise QuantizationError(f'Attribute "{name}" must be an integer list')
    result: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise QuantizationError(f'Attribute "{name}" must be an integer list')
        result.append(item)
    return result


def _integer_list(attributes: Mapping[str, JsonValue], name: str) -> list[int]:
    return _integer_attribute(attributes, name, required=True)


def _tensor_scale(graph: Graph, name: str) -> float:
    scale = graph.tensors[name].quantization_scale
    if not isinstance(scale, float) or scale <= 0:
        raise QuantizationError(f'Tensor "{name}" has no positive per-tensor scale')
    return scale


def _quantize_array(value: np.ndarray, scale: float) -> np.ndarray:
    magnitude = np.floor(np.abs(value.astype(np.float64) / scale) + 0.5)
    rounded = np.copysign(magnitude, value)
    return np.clip(rounded, -128, 127).astype(np.int8)
