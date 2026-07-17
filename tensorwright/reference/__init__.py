"""Bit-accurate software reference operations for TensorWright."""

from tensorwright.reference.int8 import (
    Conv2DConfig,
    conv2d_int8,
    multiply_int8,
    quantize_symmetric,
    requantize_int32,
    round_shift_right,
    saturate_int8,
)

__all__ = [
    "Conv2DConfig",
    "conv2d_int8",
    "multiply_int8",
    "quantize_symmetric",
    "requantize_int32",
    "round_shift_right",
    "saturate_int8",
]
