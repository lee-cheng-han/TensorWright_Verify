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
