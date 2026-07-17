# Future Zybo Z7-20 port

The eventual board target remains the Digilent Zybo Z7-20 with Zynq-7020. Future
Milestone 17 replaces the simulator adapter with a bare-metal or Linux ARM runtime,
AXI DMA, physical register access, interrupts, timeout recovery, and board measurements.

The following remain unchanged: compiler IR and passes, quantization rules, tensor and
weight layouts, `.twmodel` schemas, command encoding, hardware-interface versioning,
register offsets and semantics, AXI Stream ordering, error codes, RTL accelerator, and
reference vectors. A board-specific runtime translates the same logical transactions
to DMA and memory-mapped I/O.

The following are board-only concerns: device-tree or bare-metal discovery, physical
addresses, cache coherency, DMA-buffer allocation, interrupt plumbing, operating-system
permissions, and recovery from hardware faults. They must not leak into compiler output
or change model semantics.

Until hardware is available, TensorWright makes no claim of physical FPGA execution,
actual speedup, real DMA or ARM overhead, board power, hardware-validated throughput, or
real-time inference.
