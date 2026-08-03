# TensorWright

[![Verification](https://github.com/lee-cheng-han/TensorWright_Verify/actions/workflows/verification.yml/badge.svg)](https://github.com/lee-cheng-han/TensorWright_Verify/actions/workflows/verification.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![SystemVerilog](https://img.shields.io/badge/RTL-SystemVerilog-5C2D91)](rtl/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A hardware-aware compiler, FPGA inference engine, and cross-layer verification platform
for quantized neural networks.**

TensorWright imports ONNX models, optimizes and quantizes them to INT8, produces scheduled
`.twmodel` deployment bundles, and executes supported workloads on a reusable SystemVerilog
accelerator. Its bit-accurate software reference and cycle-aware RTL traces locate the first
numerical or streaming-protocol divergence, explain the likely cause, recommend fixes, and
generate a reproducible regression and offline dashboard.

The accelerator targets the Xilinx Zynq-7020 on the Zybo Z7-20. The IP meets a 100 MHz
timing target after Vivado placement and routing. Compiler-to-RTL execution, simulation,
verification, fault diagnosis, and timing closure are reproducible without a physical board;
on-board integration and measurement are described in [Hardware status](#hardware-status).

## Why TensorWright

AI accelerator failures cross abstraction boundaries. A wrong output may originate in graph
lowering, quantization, arithmetic, control logic, or a ready/valid handshake. TensorWright
preserves operation identity from ONNX through RTL and aligns software and hardware traces at
semantic trace points, allowing it to report the first causal mismatch instead of thousands
of downstream symptoms.

```text
ONNX + calibration data
          |
          v
 import -> optimize -> quantize -> partition -> schedule
          |                                      |
          v                                      v
 bit-accurate reference                 .twmodel deployment bundle
          |                                      |
          v                                      v
 software trace                         Verilator / FPGA trace
          \                                      /
           +---------- semantic alignment -------+
                              |
                              v
             first divergence -> diagnosis -> regression -> dashboard
```

## Capabilities

### Compiler and runtime

- ONNX import, typed intermediate representation, and stable source-operation identities
- Constant folding, dead-code elimination, fusion, and shape validation
- Calibration-driven INT8 quantization and bit-accurate reference execution
- Hardware/software partitioning, memory planning, and deterministic scheduling
- Versioned `.twmodel` bundles containing commands, parameters, reference vectors, and
  compilation metadata
- CLI workflows for compilation, inspection, simulation, benchmarking, and trace analysis

### RTL accelerator

- Pipelined signed INT8 3×3 convolution datapath
- Multi-channel multiply-accumulate, bias, fixed-point requantization, activation, and
  saturation
- Activation, weight, window, and stream buffering
- AXI-style streaming interfaces and memory-mapped control/status registers
- Interrupts, error reporting, and performance counters
- Direct execution of compiler-emitted weights, biases, quantization records, and inputs
  under Verilator

### Verification and debugging

- Versioned canonical traces with scalable `.npy`/`.npz` tensor payloads
- Python reference, Verilator RTL, Cocotb, FINN, hls4ml, and third-party adapter support
- Deterministic semantic alignment and first-divergence localization
- Numerical diagnosis with evidence, confidence, and recommended fixes
- Protocol checks for ready/valid acceptance, transfer counts, order, and packet boundaries
- Failure-preserving input minimization and automatic regression generation
- Self-contained HTML reports for local review, CI artifacts, and demonstrations

## Quick start

### Requirements

- Python 3.10 or newer
- GNU Make
- Verilator for RTL linting, simulation, and hardware/software comparison
- AMD Vivado for Zynq-7020 synthesis and implementation

Vivado is optional unless reproducing FPGA implementation results.

### Install

```bash
git clone https://github.com/lee-cheng-han/TensorWright_Verify.git
cd TensorWright_Verify
make setup

tensorwright --version
tensorwright --help
```

### Compile and simulate a model

```bash
tensorwright compile model.onnx calibration.npz model.twmodel
tensorwright inspect-bundle model.twmodel
tensorwright simulate model.twmodel --seed 32325
tensorwright benchmark model.twmodel --runs 20
```

Calibration data is supplied as an NPZ archive. The compiler emits a directory with the
`.twmodel` extension containing the graph, command stream, weights, biases, quantization
parameters, constants, memory plan, schedule, reference vectors, labels, and compilation
report. See the [deployment bundle specification](docs/deployment_bundle.md).

### Compare software and RTL traces

```bash
tensorwright trace inspect traces/reference.jsonl
tensorwright trace compare traces/reference.jsonl traces/rtl.jsonl --report report.json
tensorwright trace diagnose traces/reference.jsonl traces/rtl.jsonl --report diagnosis.json
tensorwright trace diagnose-protocol traces/reference.jsonl traces/rtl.jsonl
tensorwright dashboard traces/reference.jsonl traces/rtl.jsonl report.html
```

Reduce a failure and turn it into a reusable regression:

```bash
tensorwright minimize failing.npz minimal.npz --oracle python verify_failure.py
tensorwright generate-regression minimal.npz minimal.report.json reference.jsonl \
  regressions/conv_rounding --name conv_rounding
```

## Demonstrations

| Command | Demonstration |
|---|---|
| `make demo-model` | ONNX import, INT8 compilation, bundle simulation, and ten-class inference |
| `make demo-bundle-rtl` | Compiler-generated `.twmodel` data executed by the real Verilator RTL |
| `make demo` | Clean baseline, numerical fault, protocol fault, diagnosis, fixes, and dashboards |
| `make demo-clean` | Clean software-versus-RTL comparison only |
| `make demo-numerical-fault` | Internal requantization-rounding defect and generated regression |
| `make demo-protocol-fault` | Dropped-transfer fault and protocol diagnosis |

The presentation demo creates:

- `build/demo/index.html` — numerical divergence dashboard
- `build/demo/protocol.html` — streaming-protocol dashboard
- `build/demo/presentation.html` — single-screen presentation view
- `build/demo/tensorwright-demo-report.zip` — portable report archive

See the [video demo guide](docs/demo.md) for the recommended recording sequence.

## Verification

Run the complete board-independent release gate:

```bash
make release-check
```

This runs Python formatting and linting, strict type checking, unit tests, Verilator lint and
regressions, the model workflow, compiler-bundle RTL execution, and the diagnosis demo.

| Validated check | Result |
|---|---:|
| Python tests | 115 passed |
| Exhaustive INT8 multiplier regression | 65,536 passed |
| Post-processing vectors | 508 passed |
| Arithmetic-core vectors | 150 passed |
| Randomized convolution layers | 20 passed |
| Recognizable model classifications | 10/10 correct |
| Compiler bundle → Verilator RTL | 18/18 outputs matched |
| Python lint, formatting, and type checking | Passed |
| Verilator lint | Passed |

GitHub Actions runs the software and RTL verification suites on every push and pull request
and publishes the demonstration report as a CI artifact.

## FPGA results

Reproduce out-of-context synthesis and implementation with Vivado:

```bash
make synth
make implement
```

Reports are written under `build/synthesis/` and `build/implementation/`.

| Routed accelerator IP | Result |
|---|---:|
| Target device | `xc7z020clg400-1` |
| Target clock | 100 MHz |
| Worst negative slack | +1.302 ns |
| Total negative slack | 0 ns |
| Routed nets | 4,117/4,117; 0 routing errors |
| LUTs | 2,555 |
| Flip-flops | 1,623 |
| DSP blocks | 4 |
| Block RAM | 0 |
| Vectorless power estimate | 0.137 W |

These are reproducible out-of-context implementation results for the accelerator IP, not
measurements from a programmed board. See the [implementation report](docs/fpga_implementation.md)
and [performance methodology](docs/performance.md) for interpretation.

## Supported hardware scope

The native bundle-to-RTL path executes an INT8, valid 3×3, stride-one convolution with a
`1×3×5×5` input, `2×3×3×3` weights, and `1×2×3×3` output. The compiler and software
runtime support a broader graph subset and assign unsupported hardware operations to the
software partition. The exact operator and constraint matrix is maintained in
[supported operators](docs/supported_operators.md).

## Hardware status

The compiler, bundle format, software reference, native RTL arithmetic, Verilator runtime,
trace pipeline, diagnosis engine, dashboards, regressions, and 100 MHz routed accelerator IP
are validated.

Completing deployment on a physical Zybo Z7-20 requires:

- Zynq Processing System, DMA, and AXI interconnect integration
- Board clocks, resets, interrupts, address map, and constraints
- Bitstream generation and hardware export
- Linux userspace or driver transport for `.twmodel` data
- On-board correctness testing and measured latency, throughput, power, and temperature

See the [board-independent release report](docs/board_independent_release.md) and
[release checklist](docs/release_checklist.md) for the evidence boundary.

## Repository layout

```text
TensorWright/
├── tensorwright/       # Compiler, runtime, CLI, and trace analysis
├── rtl/                # SystemVerilog accelerator and interfaces
├── verification/       # Cocotb and SystemVerilog testbenches
├── fpga/               # FPGA integration assets
├── asic/               # ASIC-oriented integration area
├── dashboard/          # Offline dashboard assets
├── models/             # Example models and calibration data
├── benchmarks/         # Performance workloads
├── tests/              # Python tests
├── scripts/            # Demo, simulation, synthesis, and implementation drivers
└── docs/               # Specifications, methodology, and user guides
```

## Documentation

| Area | Documents |
|---|---|
| Design | [Architecture](docs/architecture.md) · [Compiler IR](docs/compiler_ir.md) · [Quantization](docs/quantization.md) · [Supported operators](docs/supported_operators.md) |
| Deployment | [Bundle format](docs/deployment_bundle.md) · [Simulation runtime](docs/simulation_runtime.md) · [Performance](docs/performance.md) |
| Hardware | [Convolution engine](docs/convolution_engine.md) · [RTL arithmetic](docs/rtl_arithmetic.md) · [Synthesis](docs/synthesis_methodology.md) · [FPGA implementation](docs/fpga_implementation.md) |
| Verification | [Verification plan](docs/verification_plan.md) · [Trace format](docs/trace_format.md) · [Trace comparison](docs/trace_comparison.md) · [Numerical diagnosis](docs/numerical_diagnosis.md) · [Protocol diagnosis](docs/protocol_diagnosis.md) |
| Debugging | [Dashboard](docs/dashboard.md) · [Input minimization](docs/input_minimization.md) · [Regression generation](docs/regression_generation.md) · [Adapter API](docs/trace_adapter_api.md) |
| Project status | [Board-independent release](docs/board_independent_release.md) · [Release checklist](docs/release_checklist.md) · [Roadmap](docs/roadmap.md) |

## Development

```bash
make format
make lint
make type-check
make test
make lint-rtl
make test-rtl
```

Changes should preserve bit-accurate agreement between the reference implementation and
RTL, include tests for new behavior, and pass `make release-check` before submission.

## License

TensorWright is released under the [MIT License](LICENSE).
