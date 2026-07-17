# Simulation-first roadmap

The first complete TensorWright demonstration requires no physical board. Milestones
0–16 produce a compiler-to-RTL-simulation and synthesis-analysis flow targeted for the
Zybo Z7-20. Physical deployment is isolated as future Milestone 17.

| Milestone | Scope | Status |
| --- | --- | --- |
| 0 | Specification, package, CLI, and repository foundation | Complete |
| 1 | Bit-accurate INT8 integer reference | Complete |
| 2 | ONNX frontend and typed compiler IR | Complete |
| 3 | Deterministic graph optimizations | Complete |
| 4 | Calibrated quantized full-model software backend | Complete |
| 5 | RTL arithmetic core and exact differential unit tests | Complete |
| 6 | AXI-style streaming, FIFOs, buffering, and backpressure | Complete |
| 7 | Register interface, control, errors, interrupts, and counters | Complete |
| 8 | Integrated reusable RTL convolution engine | Complete |
| 9 | Memory planning, scheduling, commands, and `.twmodel` bundle | Complete |
| 10 | Bundle-driven simulation runtime | Next |
| 11 | Full-model RTL-assisted simulation with CPU fallbacks | Planned |
| 12 | Simulator-derived cycle, stall, and utilization analysis | Planned |
| 13 | Zynq-7020 Vivado synthesis and report parsing | Planned |
| 14 | No-board CLI/dashboard demonstration | Planned |
| 15 | Verification closure and CI-compatible regression | Planned |
| 16 | Simulation/synthesis FPGA design-space exploration | Planned |
| 17 | Future Zybo Z7-20 physical deployment | Deferred until hardware is available |

## First-release acceptance

The no-board release is complete when a supported ONNX CNN compiles into a validated
`.twmodel`; a generic simulation runtime programs the documented register model and
stream interfaces from compiler-generated commands; at least two convolutions execute
in RTL simulation; all RTL tensors exactly match the integer reference; CPU fallbacks
complete the prediction; simulator counters and clearly labeled clock-derived latency
are reported; and actual Vivado reports for `xc7z020clg400-1` are parsed without
fabricated values.

## Next implementation

Milestone 10 is the next step. It loads validated `.twmodel` bundles, creates simulated
memory, programs the stable register contract, drives the stable streams with timeouts
and backpressure, executes CPU fallbacks, and collects complete outputs.

Milestone 17 replaces the simulation transport with an ARM runtime, AXI DMA, physical
accelerator control, interrupts, and real measurements. It must not require redesign of
the compiler, bundle, command, tensor-layout, register, or RTL interfaces.
