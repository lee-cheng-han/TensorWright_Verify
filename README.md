# TensorWright Verify

> Find the first hardware/software mismatch—not the ten thousand errors that follow it.

TensorWright Verify is a cross-layer debugging platform for quantized AI accelerators.
It aligns execution traces from software reference models and RTL or HLS simulations,
locates the first divergence, identifies likely numerical or streaming causes, minimizes
failing inputs, and generates reproducible regression tests.

The project is migrating from its original ONNX-to-FPGA compiler focus. The compiler,
bit-accurate INT8 reference, `.twmodel` bundles, custom SystemVerilog accelerator,
register/stream interfaces, and simulation runtime remain verification infrastructure.
TensorWright does not position itself as a replacement for FINN or hls4ml.

## Currently implemented

- Validated ONNX frontend, typed IR, optimization, and quantization
- Bit-accurate integer reference and multi-channel convolution RTL
- AXI-style streams, control registers, errors, interrupts, and counters
- Versioned `.twmodel` bundles and command-driven simulation host
- Canonical trace schema version 2 with strict validation and chunked tensor payloads
- Optional operation-output traces from the quantized Python reference
- Optional RTL output-transfer traces with cycle, ready/valid, sequence, and TLAST metadata
- `tensorwright trace inspect` trace summaries
- Semantic trace alignment and deterministic first-divergence reports
- Versioned numerical diagnosis rules with evidence and confidence

The supported canonical trace sources are the quantized Python reference and the custom
TensorWright RTL output stream. The self-checking Verilator regression produces a real RTL
trace; the simulator-independent capture API is also suitable for Cocotb monitors. The
comparison engine aligns scalar and chunked payloads across these backends and reports the
first missing, unexpected, structurally incompatible, or unequal value. Deterministic rules
classify supported numerical patterns without claiming protocol causes. FINN and hls4ml
adapters are planned only; no compatibility is claimed.

## Planned verification workflow

```text
ONNX/QONNX -> quantized Python reference -> canonical reference trace
custom RTL -> Cocotb simulation          -> canonical hardware trace
                                             |
                                             v
                           alignment -> first divergence -> diagnosis
```

A future failure report will emphasize the first causal mismatch:

```text
First divergence: conv2_requant, coordinate [0, 5, 7, 11]
Reference: -38    Hardware: -36    Cycle: 18,302
Likely cause: requantization rounding mismatch
```

Numerical and protocol diagnosis, minimization, regression generation, and dashboards are
planned milestones, not current product claims.

## Development

```bash
make setup
tensorwright --version
tensorwright --help
make lint
make type-check
make test
make lint-rtl
make test-rtl
```

Existing simulation compatibility is preserved:

```bash
tensorwright simulate model_name.twmodel --seed 32325
tensorwright trace inspect traces/reference.jsonl
tensorwright trace compare traces/reference.jsonl traces/rtl.jsonl --report report.json
tensorwright trace diagnose traces/reference.jsonl traces/rtl.jsonl --report diagnosis.json
```

See the [migration assessment](docs/migration_to_verify.md),
[architecture](docs/architecture.md), [trace format](docs/trace_format.md),
[trace comparison](docs/trace_comparison.md), [numerical diagnosis](docs/numerical_diagnosis.md),
and [roadmap](docs/roadmap.md).

## License

TensorWright is available under the MIT License.
