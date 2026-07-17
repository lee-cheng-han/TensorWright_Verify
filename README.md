# TensorWright

TensorWright is a hardware-aware machine-learning compiler and FPGA inference platform that imports small ONNX neural networks, performs graph optimization and INT8 quantization, generates hardware execution schedules, and deploys supported operations to a reusable SystemVerilog accelerator on a Zybo Z7-20. The system compares FPGA results against a bit-accurate software reference and reports correctness, latency, throughput, FPGA utilization, and hardware-software bottlenecks.

## Status

Milestone 1 provides the project specification, installable Python package, CLI,
documentation contracts, and a tested bit-accurate INT8 software reference for
quantization, requantization, saturation, and batch-one convolution. Compiler, runtime,
and RTL behavior are intentionally not implemented yet.

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
├── tests/
└── docs/
```

Directories for the runtime, RTL, FPGA, ASIC, and dashboard will be introduced only
when their milestones add functional code.

## License

TensorWright is available under the MIT License.
