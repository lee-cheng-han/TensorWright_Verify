# Verification plan

TensorWright uses exact differential checks across four boundaries: framework FP32
versus ONNX FP32, ONNX FP32 versus quantized software, quantized software versus RTL,
and RTL versus FPGA hardware. Results from later boundaries are not inferred from an
earlier pass.

Milestone 0 tests installation-facing CLI behavior. Milestone 1 includes directed
arithmetic edge cases, small multi-channel convolution, validation failures, and
deterministic randomized requantization and convolution tests. Seeds are fixed and
included in failing subtest context. RTL milestones add
unit and integration Cocotb tests plus SystemVerilog assertions for reset, command
acceptance, FIFO safety, transfer counts, errors, and stable AXI Stream data under
backpressure. Failed randomized tests must report their seed.

Claims about timing, utilization, accuracy, throughput, power, or ASIC results require
actual tool reports or measurements. Missing tools and skipped validation are reported
explicitly.

Milestone 2 tests ONNX file loading, checker failures, strict static-shape enforcement,
opset capture, attribute and initializer extraction, producer-consumer relationships,
explicit unsupported-operation errors, and deterministic IR serialization. Models are
constructed in memory by the tests so fixtures remain small and reviewable.

Milestone 3 adds before-and-after tests for constant folding, BatchNormalization
folding, Conv/bias/ReLU fusion, shared-intermediate safety, static view
canonicalization, backward-liveness cleanup, backend assignment, and the complete pass
pipeline. Numerical checks directly compare unfused and folded arithmetic. The
recommended two-convolution CNN is also imported and optimized end to end.

Milestone 4 tests observed calibration ranges, activation and per-output-channel weight
metadata, INT32 biases, bounded fixed-point multipliers, complete Conv/MaxPool/View/Gemm/
Softmax execution, deterministic JSON reports, measured output error, optional labeled
accuracy, and rejection of empty, non-finite, or incorrectly shaped samples.
