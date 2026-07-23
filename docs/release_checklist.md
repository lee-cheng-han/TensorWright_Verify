# Board-ready release checklist

## Complete

- [x] ONNX import, validation, typed IR, optimization, calibration, and INT8 compilation
- [x] Validated, checksummed `.twmodel` format and public compile/inspect/simulate CLI
- [x] Bit-accurate software reference and mixed FPGA/ARM schedule model
- [x] Native pipelined INT8 convolution RTL with control, streams, errors, and counters
- [x] Compiler-generated `.twmodel` data executed on real Verilator RTL
- [x] Numerical and protocol differential diagnosis, minimization, regressions, dashboards
- [x] FINN and hls4ml trace adapters
- [x] Python tests, lint, static typing, RTL lint, and randomized RTL regressions
- [x] 100 MHz synthesis and fully routed out-of-context Zynq-7020 implementation
- [x] Recognizable software model demo and video-oriented RTL debugging demo
- [x] Reproducible board-independent release command: `make release-check`

## Requires the physical board integration

- [ ] Select and instantiate the Zynq processing-system and DMA architecture
- [ ] Connect clocks, resets, AXI interconnect, interrupt, and address map
- [ ] Add board-level constraints and generate the deployable bitstream
- [ ] Implement the Linux userspace/driver transport against real DMA and registers
- [ ] Run `.twmodel` inference on the Zybo Z7-20
- [ ] Measure wall-clock latency, throughput, power, temperature, and long-run stability

The unchecked items require board-level architectural choices or physical measurements.
They are not represented as completed by simulation or out-of-context implementation.
