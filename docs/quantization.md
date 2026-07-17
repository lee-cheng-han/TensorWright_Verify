# Quantization contract

The MVP uses signed symmetric quantization with zero point zero. Inputs, weights, and
outputs are INT8; products are INT16; accumulation and bias are INT32. Activations use
per-tensor scales initially, while weights should use per-output-channel scales.

```text
q = round_ties_away_from_zero(real_value / scale)
q = clamp(q, -128, 127)
```

The compiler will approximate `(input_scale * weight_scale) / output_scale` with a
fixed-point multiplier and shift. Post-processing order is INT32 bias addition,
fixed-point multiply, rounded arithmetic shift, optional ReLU, then signed INT8
saturation.

For right shifts, TensorWright divides the magnitude by `2**shift`, rounds halfway
cases upward, and restores the original sign. This is round-to-nearest with ties away
from zero. Examples are `3 >> 1 = 2`, `-3 >> 1 = -2`, `1 >> 1 = 1`, and
`-1 >> 1 = -1`. A zero shift is an identity and negative shifts are invalid.

The integer reference checks input and weight ranges and requires every convolution
partial sum and bias-added accumulator to remain within signed INT32. Overflow raises
an error rather than silently wrapping. Fixed-point multiplication uses an unbounded
software intermediate. Milestone 5 fixes the RTL contract to an unsigned 31-bit
multiplier, a 7-bit non-negative shift, and a signed 64-bit product. The compiler rejects
fixed-point parameters that do not fit this interface.

The bit-accurate reference implementation is in `tensorwright.reference`. Convolution
uses batch-one CHW activations and OIHW weights, explicit zero padding, positive integer
strides, and per-output-channel bias, multiplier, and shift values.

Milestone 4 calibration and graph quantization are specified in
`quantized_compilation.md`. Calibration is data-dependent: reported ranges, errors, and
accuracy values always come from the samples supplied to that compilation run.
