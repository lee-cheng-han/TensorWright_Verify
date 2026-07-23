# TensorWright

> Compile quantized neural networks, execute them in RTL, and find the first
> hardware/software mismatch—not the thousands of errors that follow it.

TensorWright is an end-to-end hardware-aware machine-learning compiler and FPGA inference
platform for quantized AI accelerators. It imports ONNX models, performs graph optimization
and INT8 quantization, generates hardware execution schedules and versioned `.twmodel`
deployment bundles, and executes supported convolution workloads on a reusable SystemVerilog
accelerator. A bit-accurate Python reference, cycle-aware RTL tracing, automated diagnosis,
and offline dashboards connect model compilation to hardware verification.

The primary hardware target is the Xilinx Zynq-7020 on the Zybo Z7-20. The accelerator IP
has been synthesized, placed, and routed for that device at 100 MHz. All compiler, simulation,
RTL, verification, timing-closure, and documentation work that does not require physical
hardware is included in this repository.

## Features

- ONNX import with stable source-operation identities and a typed compiler IR
- Constant folding, dead-code elimination, operator fusion, and shape validation
- Calibration-driven INT8 quantization with bit-accurate integer reference execution
- Hardware/software partitioning and deterministic execution scheduling
- Versioned `.twmodel` bundles containing commands, parameters, memory plans, schedules,
  reference vectors, and compilation reports
- Pipelined multi-channel 3×3 convolution RTL with signed multiply-accumulate, bias,
  requantization, activation, saturation, buffering, control, and streaming interfaces
- Direct execution of compiler-generated bundle data on the Verilator RTL model
- Canonical trace format with scalar events and scalable NumPy tensor payloads
- Semantic alignment across software, RTL, FINN, hls4ml, and registered third-party adapters
- First-divergence localization for numerical and streaming-protocol failures
- Evidence-based diagnosis with confidence levels and recommended fixes
- Ready/valid, ordering, transfer-count, and packet-boundary protocol analysis
- Failure-preserving input minimization and deterministic regression generation
- Self-contained HTML dashboards suitable for local debugging and CI artifacts
- Reproducible benchmarking, Vivado synthesis, placement, routing, and release validation

## System flow

```text
ONNX model + calibration data
             |
             v
    import -> optimize -> quantize -> partition -> schedule
             |                                  |
             v                                  v
   bit-accurate reference              .twmodel deployment bundle
             |                                  |
             v                                  v
     software trace                    Verilator / FPGA runtime
             |                                  |
             +------------> alignment <---------+
                                |
                                v
                    first divergence and cause
                                |
                                v
                  recommendations, regression,
                       and offline dashboard
```

TensorWright reports the first causal mismatch rather than presenting every downstream
error. A typical diagnosis identifies the operation, tensor coordinate, trace point,
reference and candidate values, hardware cycle, likely cause, supporting evidence, and
recommended checks.

## Installation

TensorWright requires Python 3.10 or newer. Verilator is required for RTL execution, and
Vivado is required only for Xilinx synthesis and implementation.

```bash
git clone https://github.com/<owner>/tensorwright.git
cd tensorwright
make setup

tensorwright --version
tensorwright --help
```

## Compiler and runtime

Compile an ONNX model using calibration samples stored in an NPZ archive:

```bash
tensorwright compile model.onnx calibration.npz model.twmodel
tensorwright inspect-bundle model.twmodel
tensorwright simulate model.twmodel --seed 32325
tensorwright benchmark model.twmodel --runs 20
```

A deployment bundle is a directory with the `.twmodel` extension:

```text
model.twmodel/
├── manifest.json
├── graph.json
├── commands.bin
├── weights.bin
├── biases.bin
├── quantization.bin
├── constants.bin
├── memory_plan.json
├── schedule.json
├── labels.txt
├── reference_input.bin
├── reference_output.bin
└── compilation_report.json
```

The native RTL path accepts a scheduled 3×3, stride-one INT8 convolution workload with
fixed demonstration dimensions of `1×3×5×5` input, `2×3×3×3` weights, and `1×2×3×3`
output. Operations outside the native hardware subset remain executable through the
software partition.

## Verification and diagnosis

```bash
tensorwright trace inspect traces/reference.jsonl
tensorwright trace compare traces/reference.jsonl traces/rtl.jsonl --report report.json
tensorwright trace diagnose traces/reference.jsonl traces/rtl.jsonl \
  --report diagnosis.json
tensorwright trace diagnose-protocol traces/reference.jsonl traces/rtl.jsonl
tensorwright dashboard traces/reference.jsonl traces/rtl.jsonl report.html
```

Large tensors are stored in `.npy` or `.npz` payloads and referenced by compact JSONL
metadata. Coordinate-level events remain available for selected trace points, small tests,
and the region surrounding a failure.

Failures can be reduced and converted into reusable regression packages:

```bash
tensorwright minimize failing.npz minimal.npz --oracle python verify_failure.py
tensorwright generate-regression minimal.npz minimal.report.json reference.jsonl \
  regressions/conv_rounding --name conv_rounding
```

Trace adapters may be inspected or used to convert external backend output:

```bash
tensorwright trace adapters
tensorwright trace convert transfers.txt rtl.jsonl \
  --adapter tensorwright.verilator_rtl --options @adapter-options.json
```

## Demonstrations

Run a recognizable ten-class model through ONNX import, graph optimization, INT8
compilation, bundle generation, simulation, and classification:

```bash
make demo-model
```

Run a compiler-generated `.twmodel` directly against the real Verilator convolution RTL
and compare all hardware results with the software reference:

```bash
make demo-bundle-rtl
```

Run the presentation-oriented verification demo:

```bash
make demo
```

This demo produces:

- A clean software-versus-RTL baseline
- A genuine internal requantization-rounding defect
- A dropped-transfer streaming-protocol defect
- Cycle-aware first-divergence reports
- Automated diagnoses and recommended fixes
- A generated regression that reproduces the numerical failure
- Self-contained dashboards at `build/demo/index.html`,
  `build/demo/protocol.html`, and `build/demo/presentation.html`
- A portable report archive at `build/demo/tensorwright-demo-report.zip`

Individual recording segments are available through `make demo-clean`,
`make demo-numerical-fault`, and `make demo-protocol-fault`. See the
[video demonstration guide](docs/demo.md) for the recording sequence.

## Validation

Run the complete board-independent release gate:

```bash
make release-check
```

The gate checks Python and RTL formatting, linting, static typing, unit tests, Verilator
regressions, the recognizable model workflow, compiler-bundle RTL execution, and the full
fault-diagnosis demo.

The validated release results are:

| Check | Result |
|---|---:|
| Python tests | 115 passed |
| Multiplier regression | 65,536 cases passed |
| Post-processing regression | 508 cases passed |
| Arithmetic-core regression | 150 cases passed |
| Randomized convolution layers | 20 passed |
| Recognizable model demo | 10/10 classifications |
| Compiler bundle → Verilator RTL | 18/18 outputs matched |
| Python lint and formatting | Passed |
| Python static typing | Passed |
| Verilator lint | Passed |

## FPGA implementation

With Vivado installed, reproduce synthesis and out-of-context implementation for the
Zynq-7020:

```bash
make synth
make implement
```

Reports are written to `build/synthesis/` and `build/implementation/`.

| Routed result | Value |
|---|---:|
| Device | `xc7z020clg400-1` |
| Target clock | 100 MHz |
| Worst negative slack | +1.302 ns |
| Total negative slack | 0 ns |
| Routing | 4,117/4,117 nets, 0 errors |
| LUTs | 2,555 |
| Flip-flops | 1,623 |
| DSP blocks | 4 |
| Block RAM | 0 |
| Vectorless power estimate | 0.137 W |

These figures describe the routed accelerator IP in an out-of-context implementation.
They are timing and resource evidence, not measurements from a programmed board.

## Physical-board deployment

Deployment on the Zybo Z7-20 requires the board-specific shell around the validated
accelerator IP:

- Zynq Processing System and DMA integration
- AXI interconnect, clocks, resets, interrupts, and address map
- Board-level timing and pin constraints
- Bitstream generation and hardware export
- Linux userspace or driver transport for `.twmodel` data
- On-board correctness, latency, throughput, power, temperature, and stability measurements

These tasks require access to the physical board and do not alter the compiler, bundle
format, reference model, RTL arithmetic, tracing, or diagnosis architecture.

## Repository layout

```text
tensorwright/
├── tensorwright/        # Python compiler, runtime, CLI, and trace analysis
├── rtl/                 # Reusable SystemVerilog accelerator
├── verification/        # Cocotb and SystemVerilog testbenches
├── fpga/                # FPGA integration assets
├── asic/                # ASIC-oriented integration area
├── dashboard/           # Offline dashboard assets
├── models/              # Model and calibration examples
├── benchmarks/          # Performance workloads
├── tests/               # Python test suite
├── scripts/             # Demo, simulation, synthesis, and implementation drivers
└── docs/                # Architecture, formats, methodology, and guides
```

## Documentation

- [Architecture](docs/architecture.md)
- [Compiler IR](docs/compiler_ir.md)
- [Quantization](docs/quantization.md)
- [Supported operators](docs/supported_operators.md)
- [Deployment bundles](docs/deployment_bundle.md)
- [Convolution engine](docs/convolution_engine.md)
- [RTL arithmetic](docs/rtl_arithmetic.md)
- [Simulation runtime](docs/simulation_runtime.md)
- [Trace format](docs/trace_format.md)
- [Trace comparison](docs/trace_comparison.md)
- [Numerical diagnosis](docs/numerical_diagnosis.md)
- [Protocol diagnosis](docs/protocol_diagnosis.md)
- [Input minimization](docs/input_minimization.md)
- [Regression generation](docs/regression_generation.md)
- [Dashboard](docs/dashboard.md)
- [Trace adapter API](docs/trace_adapter_api.md)
- [FINN adapter](docs/finn_adapter.md)
- [hls4ml adapter](docs/hls4ml_adapter.md)
- [Performance evidence](docs/performance.md)
- [Synthesis methodology](docs/synthesis_methodology.md)
- [FPGA implementation](docs/fpga_implementation.md)
- [Verification plan](docs/verification_plan.md)
- [Board-independent release](docs/board_independent_release.md)
- [Release checklist](docs/release_checklist.md)
- [Roadmap](docs/roadmap.md)

## License

TensorWright is available under the [MIT License](LICENSE).
