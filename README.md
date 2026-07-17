# TensorWright

TensorWright is an ONNX-to-accelerator compilation and verification platform that
performs graph optimization, INT8 quantization, hardware scheduling, simulated runtime
execution, bit-accurate RTL verification, and FPGA synthesis analysis.

The first complete release is simulation-first and requires no physical FPGA board. It
targets a reusable SystemVerilog accelerator for the Zynq-7020 used by the Zybo Z7-20,
while reserving physical board deployment and measurements for a future extension.

## Status

Milestone 5 provides the project specification, installable Python package, compiler,
bit-accurate INT8 reference, quantized software execution, and a synthesizable nine-lane
RTL arithmetic core with exact Verilator differential tests. Streaming, integrated
accelerator control, runtime, deployment bundles, and hardware execution are not yet
implemented.

The current implementation does not claim RTL-simulated execution, physical FPGA
execution, measured hardware speedup, board power, DMA latency, or ARM-to-FPGA
bring-up. See the [simulation-first architecture](docs/architecture.md),
[roadmap](docs/roadmap.md), and [metric classifications](docs/performance_model.md).

## Development

Install the project in editable mode and inspect the command-line interface:

```bash
make setup
tensorwright --version
tensorwright --help
make lint
make type-check
make test
```

Compiled deployment bundles use the `.twmodel` directory extension. See [the bundle format](docs/deployment_bundle.md) for the initial layout.

## Current repository layout

```text
tensorwright/
├── README.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── compiler/
├── tensorwright/
├── rtl/
├── verification/
├── scripts/
├── tests/
└── docs/
```

Directories for the runtime, FPGA, ASIC, and dashboard will be introduced only
when their milestones add functional code.

## License

TensorWright is available under the MIT License.
