"""Calibration and graph quantization for the MVP execution path."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tensorwright.compiler.errors import QuantizationError
from tensorwright.compiler.ir import Graph, Tensor
from tensorwright.compiler.passes.utils import rebuild_links, unique_tensor_name
from tensorwright.compiler.quantization.executor import execute_float, execute_quantized


@dataclass(frozen=True)
class CalibrationRange:
    """Observed finite minimum and maximum for one tensor."""

    minimum: float
    maximum: float

    @property
    def scale(self) -> float:
        magnitude = max(abs(self.minimum), abs(self.maximum))
        return magnitude / 127.0 if magnitude > 0.0 else 1.0


@dataclass(frozen=True)
class CompilationResult:
    """Quantized graph plus measured, serializable compilation report."""

    graph: Graph
    report: dict[str, Any]

    def report_json(self, *, indent: int = 2) -> str:
        """Serialize the measured compilation report deterministically."""
        return json.dumps(self.report, indent=indent, sort_keys=True) + "\n"

    def write_report(self, path: str | Path) -> None:
        """Write compilation-report content to a UTF-8 JSON file."""
        Path(path).write_text(self.report_json(), encoding="utf-8")


def compile_quantized(
    graph: Graph,
    calibration_samples: Sequence[Mapping[str, np.ndarray]],
    *,
    labels: Sequence[int] | None = None,
) -> CompilationResult:
    """Calibrate, quantize, execute, and compare an optimized graph."""
    if not calibration_samples:
        raise QuantizationError("At least one calibration sample is required")
    if labels is not None and len(labels) != len(calibration_samples):
        raise QuantizationError("Labels must match the calibration sample count")
    ranges = _calibrate(graph, calibration_samples)
    quantized = _quantize_graph(graph, ranges)
    metrics = _compare(graph, quantized, calibration_samples, labels)
    report: dict[str, Any] = {
        "format_version": 1,
        "calibration_sample_count": len(calibration_samples),
        "calibration_ranges": {
            name: {
                "minimum": value.minimum,
                "maximum": value.maximum,
                "scale": value.scale,
            }
            for name, value in sorted(ranges.items())
        },
        "quantized_tensor_count": sum(
            tensor.compiled_dtype in {"int8", "int32"}
            for tensor in quantized.tensors.values()
        ),
        "comparison": metrics,
    }
    return CompilationResult(quantized, report)


def _calibrate(
    graph: Graph, samples: Sequence[Mapping[str, np.ndarray]]
) -> dict[str, CalibrationRange]:
    extrema: dict[str, tuple[float, float]] = {}
    for sample in samples:
        values = execute_float(graph, sample, capture_all=True)
        for name, value in values.items():
            if graph.tensors[name].is_constant:
                continue
            if not np.all(np.isfinite(value)):
                raise QuantizationError(f'Calibration tensor "{name}" is not finite')
            minimum = float(np.min(value))
            maximum = float(np.max(value))
            previous = extrema.get(name, (minimum, maximum))
            extrema[name] = (min(previous[0], minimum), max(previous[1], maximum))
    return {
        name: CalibrationRange(minimum, maximum)
        for name, (minimum, maximum) in extrema.items()
    }


def _quantize_graph(graph: Graph, ranges: dict[str, CalibrationRange]) -> Graph:
    result = deepcopy(graph)
    for name in result.inputs:
        _set_activation(result.tensors[name], ranges[name].scale)
    for operation in result.operations:
        if operation.operation_type in {"Conv", "Gemm"}:
            _quantize_linear_operation(result, operation.name, ranges)
        elif operation.operation_type in {"MaxPool", "View"}:
            input_scale = _per_tensor_scale(result.tensors[operation.inputs[0]])
            _set_activation(result.tensors[operation.outputs[0]], input_scale)
        elif operation.operation_type == "Softmax":
            result.tensors[operation.outputs[0]].compiled_dtype = "float32"
        else:
            raise QuantizationError(
                f'Quantization does not support operation "{operation.operation_type}"'
            )
    rebuild_links(result)
    return result


def _quantize_linear_operation(
    graph: Graph, operation_name: str, ranges: dict[str, CalibrationRange]
) -> None:
    operation = next(item for item in graph.operations if item.name == operation_name)
    input_scale = _per_tensor_scale(graph.tensors[operation.inputs[0]])
    output_name = operation.outputs[0]
    output_scale = ranges[output_name].scale
    weights = graph.tensors[operation.inputs[1]]
    if not weights.is_constant:
        raise QuantizationError(
            f'Operation "{operation.name}" requires constant weights'
        )
    weight_values = np.asarray(weights.constant_data, dtype=np.float64)
    output_axis = 0 if operation.operation_type == "Conv" else 1
    reduction_axes = tuple(
        index for index in range(weight_values.ndim) if index != output_axis
    )
    maximum = np.max(np.abs(weight_values), axis=reduction_axes)
    weight_scales = np.where(maximum > 0.0, maximum / 127.0, 1.0)
    reshape = [1] * weight_values.ndim
    reshape[output_axis] = weight_scales.shape[0]
    quantized_weights = _quantize(weight_values, weight_scales.reshape(reshape))

    weight_name = unique_tensor_name(graph, f"{operation.name}__int8_weights")
    graph.tensors[weight_name] = Tensor(
        name=weight_name,
        shape=list(weights.shape),
        original_dtype=weights.original_dtype,
        compiled_dtype="int8",
        layout=weights.layout,
        quantization_scale=weight_scales.tolist(),
        is_constant=True,
        constant_data=quantized_weights.tolist(),
    )
    operation.inputs[1] = weight_name

    channel_count = weight_scales.shape[0]
    bias_values = np.zeros(channel_count, dtype=np.float64)
    if len(operation.inputs) == 3:
        bias_tensor = graph.tensors[operation.inputs[2]]
        if not bias_tensor.is_constant:
            raise QuantizationError(
                f'Operation "{operation.name}" requires constant bias'
            )
        bias_values = np.asarray(bias_tensor.constant_data, dtype=np.float64)
    bias_scales = input_scale * weight_scales
    quantized_bias = _round_away(bias_values / bias_scales).astype(np.int64)
    if np.any(quantized_bias < -(1 << 31)) or np.any(quantized_bias > (1 << 31) - 1):
        raise QuantizationError(f'Operation "{operation.name}" bias exceeds INT32')
    bias_name = unique_tensor_name(graph, f"{operation.name}__int32_bias")
    graph.tensors[bias_name] = Tensor(
        name=bias_name,
        shape=[int(channel_count)],
        original_dtype="float",
        compiled_dtype="int32",
        layout="UNSPECIFIED",
        quantization_scale=bias_scales.tolist(),
        is_constant=True,
        constant_data=quantized_bias.astype(np.int32).tolist(),
    )
    operation.inputs = [operation.inputs[0], weight_name, bias_name]

    multiplier, shift = _fixed_point_parameters(bias_scales / output_scale)
    operation.attributes["requantization_multipliers"] = multiplier.tolist()
    operation.attributes["requantization_shifts"] = shift.tolist()
    _set_activation(graph.tensors[output_name], output_scale)


def _fixed_point_parameters(
    real_multiplier: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    multipliers = np.empty(real_multiplier.shape, dtype=np.int64)
    shifts = np.empty(real_multiplier.shape, dtype=np.int64)
    for index, value in np.ndenumerate(real_multiplier):
        ratio = float(value)
        if not math.isfinite(ratio) or ratio < 0.0:
            raise QuantizationError(
                "Requantization multiplier must be finite and non-negative"
            )
        if ratio == 0.0:
            multipliers[index] = 0
            shifts[index] = 0
            continue
        mantissa, exponent = math.frexp(ratio)
        multiplier = int(math.floor(mantissa * (1 << 31) + 0.5))
        if multiplier == 1 << 31:
            multiplier //= 2
            exponent += 1
        shift = 31 - exponent
        if shift < 0 or shift > 127:
            raise QuantizationError(
                "Requantization shift does not fit the RTL interface"
            )
        multipliers[index] = multiplier
        shifts[index] = shift
    return multipliers, shifts


def _set_activation(tensor: Tensor, scale: float) -> None:
    tensor.compiled_dtype = "int8"
    tensor.quantization_scale = float(scale)
    tensor.zero_point = 0


def _per_tensor_scale(tensor: Tensor) -> float:
    if not isinstance(tensor.quantization_scale, float):
        raise QuantizationError(f'Tensor "{tensor.name}" needs a per-tensor scale')
    return tensor.quantization_scale


def _quantize(value: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip(_round_away(value / scale), -128, 127).astype(np.int8)


def _round_away(value: np.ndarray) -> np.ndarray:
    return np.copysign(np.floor(np.abs(value) + 0.5), value)


def _compare(
    float_graph: Graph,
    quantized_graph: Graph,
    samples: Sequence[Mapping[str, np.ndarray]],
    labels: Sequence[int] | None,
) -> dict[str, Any]:
    absolute_errors: list[np.ndarray] = []
    agreements = 0
    comparisons = 0
    float_correct = 0
    quantized_correct = 0
    for sample_index, sample in enumerate(samples):
        float_outputs = execute_float(float_graph, sample)
        quantized_outputs = execute_quantized(quantized_graph, sample)
        for name in float_graph.outputs:
            expected = float_outputs[name].astype(np.float64)
            actual = quantized_outputs[name].astype(np.float64)
            if quantized_graph.tensors[name].compiled_dtype == "int8":
                actual *= _per_tensor_scale(quantized_graph.tensors[name])
            absolute_errors.append(np.abs(expected - actual).reshape(-1))
            if expected.ndim >= 1 and expected.shape[-1] > 1:
                agreements += int(np.argmax(expected) == np.argmax(actual))
                comparisons += 1
                if labels is not None and name == float_graph.outputs[0]:
                    float_correct += int(np.argmax(expected) == labels[sample_index])
                    quantized_correct += int(np.argmax(actual) == labels[sample_index])
    combined = np.concatenate(absolute_errors)
    return {
        "max_absolute_error": float(np.max(combined)),
        "mean_absolute_error": float(np.mean(combined)),
        "top1_agreement": float(agreements / comparisons) if comparisons else None,
        "float_top1_accuracy": (
            float(float_correct / len(samples)) if labels is not None else None
        ),
        "quantized_top1_accuracy": (
            float(quantized_correct / len(samples)) if labels is not None else None
        ),
        "compared_output_count": len(absolute_errors),
    }
