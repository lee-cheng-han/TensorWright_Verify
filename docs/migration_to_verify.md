# Migration to TensorWright Verify

## Repository assessment

The original Milestones 0–10 produced working ONNX validation, typed graph IR,
deterministic optimization, calibrated INT8 execution, exact integer arithmetic, RTL
compute/stream/control blocks, an integrated convolution engine, deterministic
`.twmodel` bundles, and a command-driven simulation host. Stable operation names and
intermediate quantized tensors are accessible through `capture_all=True`.

Reusable components are the frontend, IR, quantization metadata and arithmetic,
deployment schemas, tensor layouts, register/counter definitions, RTL, verification
benches, and runtime orchestration. They become trace metadata, golden execution,
hardware integration targets, and trace producers. Compiler scheduling and FPGA design
space work become supporting rather than primary product workflows. No working subsystem
is removed.

## Baseline recorded before Milestone 11 changes

On the bare environment on 2026-07-17:

- `make lint`: failed because Ruff was not installed.
- `make type-check`: failed because mypy was not installed.
- `make test` and `make test-python`: 13 reference tests passed; six modules failed to
  import because NumPy/ONNX were not installed.
- `make test-rtl`: passed 65,536 multiplier cases, 508 postprocess vectors, 150
  arithmetic-core vectors, streaming/control tests, and 20 randomized convolution layers.
- `make regression`: target does not exist.

Using the prepared dependency environment before migration, Ruff and all 50 Python tests
passed. This distinction prevents missing dependencies from being reported as product
regressions.

## Adaptation and compatibility

The CLI keeps `--version`, `--help`, and `simulate`. Milestone 11 adds the nonbreaking
`trace inspect` command. Compiler-oriented APIs and `.twmodel` version 1 remain supported.
The existing bundle already contains graph, quantization, schedule, layout, and reference
data; future bundle versions may add explicit trace-point and hardware-stage mappings.
Version 1 remains readable, and fields will not be removed silently.

Reference trace insertion occurs after each quantized operation output. Large payloads
use `.npy` tensor chunks while small or selected regions retain scalar events. Proposed RTL
insertion points are window output, per-channel accumulator, bias/requantization output,
final stream handshake, and control completion/counters. M12 will select and implement
the smallest stable set without exposing internal state for control.

Compatibility risks include fused software/RTL stage granularity, layout differences,
missing cycles in software traces, current Python 3.14 Cocotb incompatibility, and bundle
v1 lacking explicit alignment records. Later alignment must reject ambiguity rather than
matching by line number.

## Preserved tests and deferred work

All existing Python and RTL tests remain in the normal targets. New trace tests are
additive. Alignment, first-divergence reporting, diagnosis rules, fault injection,
minimization, generated regressions, dashboard work, and FINN/hls4ml adapters are
explicitly deferred to their revised milestones.
