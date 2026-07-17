# Quantized model compilation

Milestone 4 compiles an already optimized graph using representative, user-supplied
calibration samples. It does not download data, select a dataset, or claim accuracy for
an unmeasured model.

The float executor records finite minima and maxima for every non-constant tensor.
Activation scales use signed symmetric per-tensor quantization:

```text
scale = max(abs(minimum), abs(maximum)) / 127
zero_point = 0
```

An all-zero range uses scale `1.0`. Conv and Gemm weights use a separate symmetric scale
for every output channel. Bias uses INT32 with scale `input_scale * weight_scale`.
MaxPool and metadata-only View preserve the input activation scale. Softmax explicitly
dequantizes its input and executes in floating point as an ARM fallback.

Requantization represents each positive real multiplier with `frexp`: a normalized
31-bit integer multiplier and a non-negative right shift. Rounding uses the same
ties-away-from-zero rule as the integer reference. Gemm accumulates through INT64 in
software and rejects results outside signed INT32; Conv uses the checked integer
reference implementation.

`compile_quantized` returns a quantized graph and a versioned report containing observed
ranges, scales, tensor counts, maximum and mean absolute output error, top-1 agreement
with the float graph, and—only when labels are supplied—measured float and quantized
top-1 accuracy. `CompilationResult.write_report` writes this data as deterministic JSON.
