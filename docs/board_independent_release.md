# Board-independent release status

TensorWright's host-side compiler, reference execution, RTL verification, diagnosis, and
reporting workflows are complete and runnable without an FPGA board.

## Complete without a board

- Import statically shaped ONNX models, optimize their graphs, calibrate and compile INT8
  execution, and package validated `.twmodel` bundles.
- Inspect, simulate, and benchmark bundles through the public CLI.
- Run a recognizable ten-class seven-segment classifier with `make demo-model`.
- Verify the custom convolution accelerator with Verilator, including exhaustive multiplier
  coverage, randomized stream backpressure, numerical stage traces, and protocol traces.
- Compile a fresh ONNX convolution and execute its exact `.twmodel` binary data on the
  Verilator RTL with 18/18 matching outputs using `make demo-bundle-rtl`.
- Compare software and RTL results, identify the first divergence, diagnose numerical and
  streaming faults, recommend checks and fixes, minimize failures, and generate regressions.
- Produce self-contained dashboards and a video-oriented presentation with `make demo`.
- Synthesize with `make synth`, then run out-of-context optimization, placement, and routing
  with `make implement`. The fully routed accelerator IP meets 100 MHz with +1.302 ns WNS.
- Exercise linting, static typing, Python tests, RTL tests, and demos in CI where the required
  open-source tools are available.

The recognizable classifier deliberately includes ARM fallback operations (`Gemm` and
`Softmax`) and proves the mixed-backend compiler/runtime path. The bundle-driven Verilator
demo proves the compiler/native-RTL path. Results labeled as simulated cycles, routed timing,
or vectorless power estimates are not presented as measurements from physical hardware.

## Requires a physical board

- Generate and validate the board-level Vivado block design, bitstream, clocks/resets,
  and processing-system integration for a specific Zybo Z7-20 setup.
- Implement and validate the Linux driver/DMA transport against the physical accelerator.
- Run end-to-end `.twmodel` inference on the device.
- Measure wall-clock latency, sustained throughput, power, temperature, and
  hardware stability.

These are deployment-validation tasks rather than hidden host-side milestones. They cannot be
truthfully marked complete without access to the target board.
