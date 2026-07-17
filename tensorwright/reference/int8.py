"""Integer-only numerical contract for future TensorWright RTL.

Tensor containers are plain nested sequences in CHW and OIHW order. Python integers
make intermediate arithmetic explicit; architectural INT32 boundaries are checked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

INT8_MIN = -128
INT8_MAX = 127
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1

Tensor3D: TypeAlias = list[list[list[int]]]
Tensor4D: TypeAlias = list[list[list[list[int]]]]


@dataclass(frozen=True)
class Conv2DConfig:
    """Static convolution geometry for one NCHW batch.

    Padding order is top, bottom, left, right. Only zero padding is supported.
    """

    stride_height: int = 1
    stride_width: int = 1
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    relu: bool = False

    def __post_init__(self) -> None:
        _check_non_negative("stride_height", self.stride_height)
        _check_non_negative("stride_width", self.stride_width)
        if self.stride_height <= 0 or self.stride_width <= 0:
            raise ValueError("convolution strides must be positive")
        if len(self.padding) != 4:
            raise ValueError("padding must contain four non-negative values")
        for value in self.padding:
            _check_non_negative("padding", value)


def multiply_int8(lhs: int, rhs: int) -> int:
    """Multiply two signed INT8 values, producing an exact signed INT16 value."""
    _check_range("lhs", lhs, INT8_MIN, INT8_MAX)
    _check_range("rhs", rhs, INT8_MIN, INT8_MAX)
    return lhs * rhs


def saturate_int8(value: int) -> int:
    """Clamp an integer to the signed INT8 range."""
    return min(INT8_MAX, max(INT8_MIN, value))


def quantize_symmetric(real_value: float, scale: float) -> int:
    """Quantize one finite value with symmetric, ties-away-from-zero rounding."""
    if not math.isfinite(real_value):
        raise ValueError("real_value must be finite")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")

    scaled = real_value / scale
    if scaled >= INT8_MAX + 0.5:
        return INT8_MAX
    if scaled <= INT8_MIN + 0.5:
        return INT8_MIN
    magnitude = math.floor(abs(scaled) + 0.5)
    rounded = magnitude if scaled >= 0.0 else -magnitude
    return saturate_int8(rounded)


def round_shift_right(value: int, shift: int) -> int:
    """Divide by 2**shift, rounding halfway cases away from zero.

    A shift of zero is an identity. Negative shifts are rejected so that left shifts
    cannot be confused with requantization right shifts.
    """
    _check_non_negative("shift", shift)
    if shift == 0:
        return value

    magnitude = abs(value)
    rounded_magnitude = (magnitude + (1 << (shift - 1))) >> shift
    return rounded_magnitude if value >= 0 else -rounded_magnitude


def requantize_int32(
    accumulator: int,
    bias: int,
    multiplier: int,
    shift: int,
    *,
    relu: bool = False,
) -> int:
    """Apply bias, fixed-point scaling, optional ReLU, and INT8 saturation.

    The bias sum is checked as INT32. The multiplier product is intentionally kept at
    arbitrary precision here; the eventual RTL interface must select a width capable
    of reproducing this result for compiler-generated multipliers.
    """
    _check_range("accumulator", accumulator, INT32_MIN, INT32_MAX)
    _check_range("bias", bias, INT32_MIN, INT32_MAX)
    biased = accumulator + bias
    _check_range("biased accumulator", biased, INT32_MIN, INT32_MAX, OverflowError)
    _check_non_negative("multiplier", multiplier)

    scaled = round_shift_right(biased * multiplier, shift)
    if relu:
        scaled = max(0, scaled)
    return saturate_int8(scaled)


def conv2d_int8(
    inputs: Tensor3D,
    weights: Tensor4D,
    biases: list[int],
    multipliers: list[int],
    shifts: list[int],
    config: Conv2DConfig | None = None,
) -> Tensor3D:
    """Execute batch-one, multi-channel INT8 convolution in CHW/OIHW order."""
    if config is None:
        config = Conv2DConfig()
    input_channels, input_height, input_width = _shape_3d("inputs", inputs)
    output_channels, weight_channels, kernel_height, kernel_width = _shape_4d(
        "weights", weights
    )
    if input_channels != weight_channels:
        raise ValueError("input and weight channel counts must match")
    _check_channel_parameters(output_channels, biases, multipliers, shifts)
    _check_values("inputs", inputs, INT8_MIN, INT8_MAX)
    _check_values("weights", weights, INT8_MIN, INT8_MAX)

    pad_top, pad_bottom, pad_left, pad_right = config.padding
    padded_height = input_height + pad_top + pad_bottom
    padded_width = input_width + pad_left + pad_right
    if kernel_height > padded_height or kernel_width > padded_width:
        raise ValueError("kernel must fit within the padded input")
    output_height = (padded_height - kernel_height) // config.stride_height + 1
    output_width = (padded_width - kernel_width) // config.stride_width + 1

    outputs: Tensor3D = []
    for output_channel in range(output_channels):
        plane: list[list[int]] = []
        for output_y in range(output_height):
            row: list[int] = []
            for output_x in range(output_width):
                accumulator = 0
                for input_channel in range(input_channels):
                    for kernel_y in range(kernel_height):
                        input_y = (
                            output_y * config.stride_height + kernel_y - pad_top
                        )
                        for kernel_x in range(kernel_width):
                            input_x = (
                                output_x * config.stride_width + kernel_x - pad_left
                            )
                            input_is_valid = 0 <= input_y < input_height
                            input_is_valid &= 0 <= input_x < input_width
                            if input_is_valid:
                                product = multiply_int8(
                                    inputs[input_channel][input_y][input_x],
                                    weights[output_channel][input_channel][kernel_y][
                                        kernel_x
                                    ],
                                )
                                accumulator += product
                                _check_range(
                                    "convolution accumulator",
                                    accumulator,
                                    INT32_MIN,
                                    INT32_MAX,
                                    OverflowError,
                                )
                row.append(
                    requantize_int32(
                        accumulator,
                        biases[output_channel],
                        multipliers[output_channel],
                        shifts[output_channel],
                        relu=config.relu,
                    )
                )
            plane.append(row)
        outputs.append(plane)
    return outputs


def _check_channel_parameters(
    output_channels: int,
    biases: list[int],
    multipliers: list[int],
    shifts: list[int],
) -> None:
    if not (
        len(biases) == len(multipliers) == len(shifts) == output_channels
    ):
        raise ValueError("biases, multipliers, and shifts need one value per output")
    for bias in biases:
        _check_range("bias", bias, INT32_MIN, INT32_MAX)
    for multiplier in multipliers:
        _check_non_negative("multiplier", multiplier)
    for shift in shifts:
        _check_non_negative("shift", shift)


def _shape_3d(name: str, values: Tensor3D) -> tuple[int, int, int]:
    if not values or not values[0] or not values[0][0]:
        raise ValueError(f"{name} must be a non-empty rectangular 3D tensor")
    height = len(values[0])
    width = len(values[0][0])
    if any(len(plane) != height for plane in values) or any(
        len(row) != width for plane in values for row in plane
    ):
        raise ValueError(f"{name} must be rectangular")
    return len(values), height, width


def _shape_4d(name: str, values: Tensor4D) -> tuple[int, int, int, int]:
    if not values or not values[0] or not values[0][0] or not values[0][0][0]:
        raise ValueError(f"{name} must be a non-empty rectangular 4D tensor")
    channels = len(values[0])
    height = len(values[0][0])
    width = len(values[0][0][0])
    if any(len(output) != channels for output in values) or any(
        len(kernel) != height for output in values for kernel in output
    ) or any(
        len(row) != width
        for output in values
        for kernel in output
        for row in kernel
    ):
        raise ValueError(f"{name} must be rectangular")
    return len(values), channels, height, width


def _check_values(name: str, values: object, minimum: int, maximum: int) -> None:
    if isinstance(values, list):
        for value in values:
            _check_values(name, value, minimum, maximum)
        return
    if not isinstance(values, int) or isinstance(values, bool):
        raise TypeError(f"{name} values must be integers")
    _check_range(name, values, minimum, maximum)


def _check_range(
    name: str,
    value: int,
    minimum: int,
    maximum: int,
    error_type: type[Exception] = ValueError,
) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise error_type(f"{name} must be in [{minimum}, {maximum}]")


def _check_non_negative(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
