# Verification plan

TensorWright uses exact differential checks across the first-release boundaries:
framework FP32 versus ONNX FP32, ONNX FP32 versus quantized software, and quantized
software versus RTL simulation. Future board deployment adds RTL simulation versus FPGA
hardware. Results from one boundary are never inferred for another.

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

The no-board verification path will use cocotb to reset and identify the device, program
registers, drive ready/valid streams under deterministic or seeded randomized
backpressure, enforce timeouts, collect counters and outputs, and report the first exact
mismatch with layer, tensor coordinate, arithmetic context, seed, and simulation cycle.
Protocol assertions and a compact waveform/trace artifact complement differential
checks; they do not replace them.

Milestone 5 exhaustively tests all signed INT8 multiplier inputs and differentially
tests directed and seeded-random post-processing and multi-cycle dot products against
the Python reference. Verilator lint and inline reset/input/overflow assertions are part
of the regression. Cocotb sources are present but could not execute in the current
Python 3.14 environment because cocotb 2.0.1 supports Python only through 3.13.

Milestone 6 drives the activation FIFO, packetized weight buffer, and 3x3 line/window
buffer with a fixed-seed pseudo-random pattern of source gaps and sink backpressure.
The self-checking Verilator test proves ordered delivery of 40 FIFO items, addressable
delivery of 16 weights, and exact lane packing and `tlast` placement for all nine
windows of a 5x5 raster. Inline assertions enforce stable stalled outputs, bounded FIFO
occupancy, known accepted inputs, and correctly placed image `tlast`.

Milestone 7 performs register transactions through the AXI4-Lite channels and checks
identity/version reads, undefined-address `DECERR`, invalid configuration errors,
write-one-to-clear IRQ state, soft reset, normal and premature completion, and exact
activity/transfer/MAC counters. Completion is rejected unless the final configured
output transfer has occurred.

Milestone 8 differentially tests complete 5x5, three-input-channel, two-output-channel
layers against Python-generated integer-reference vectors. Twenty fixed-seed randomized
layers vary all activations, weights, biases, multipliers, shifts, and ReLU controls;
randomized gaps and output backpressure exercise all three stream boundaries. Every one
of the 360 output values and each packet-final marker must match exactly.

Milestone 9 builds the same quantized graph twice and requires byte-identical bundle
contents. Tests unpack command fields, inspect tensor and quantization sizes, load the
validated metadata, and reject checksum corruption, incompatible versions, and invalid
bundle paths. The validator also enforces required files, command-record boundaries,
64-byte memory alignment, JSON schemas, and matching graph/schedule layer counts.

Milestone 10 executes bundles solely from decoded commands and schedules, cross-checks
packed weights and biases, allocates bounded simulated memory, records public register
transactions, and verifies final bytes against the packaged reference. Tests cover a
mixed FPGA/ARM/metadata graph, deterministic no-stall counters, reproducible seeded
backpressure, timeout failures containing their seed, and CLI success/failure behavior.
These are contract-model counters; simulator-derived trace measurements begin in the
revised Milestone 12.

The revised verification plan adds canonical trace schema and round-trip tests,
reference and RTL trace capture, semantic alignment, first-divergence localization,
deterministic diagnosis-rule fixtures, explicit fault injection, minimizer preservation
tests, and generated regression portability. Milestone 11 implements only schema
validation, Python-reference trace generation, and CLI inspection. Later capabilities
must not be inferred from the presence of schema fields.
