# Architecture

TensorWright's first complete version is an explicitly staged, board-independent stack:

```text
ONNX model -> compiler IR -> optimization and INT8 quantization
           -> memory plan and schedule -> .twmodel bundle
           -> simulation runtime -> AXI transaction model
           -> SystemVerilog accelerator simulation
           -> verification results and simulated counters
```

The compiler owns graph validation, numerical lowering, backend partitioning, memory
planning, and command generation. The simulation runtime will validate bundle and
hardware-interface versions, allocate simulated memory, program the documented register
model, drive AXI-style streams, invoke RTL layers, and execute CPU fallbacks. The
reusable SystemVerilog accelerator performs configured layers rather than embedding a
particular network.

The future board path replaces only the runtime transport:

```text
First release: simulation runtime -> cocotb/RTL simulator
Future board:  ARM runtime -> AXI DMA -> FPGA accelerator
```

The compiler IR, `.twmodel` schema, command encoding, tensor layouts, register map,
stream ordering, error behavior, and RTL ports must be shared by both paths. Simulation
adapters may not introduce model-specific commands or bypass those interfaces.

Correctness advances in this order: specification, bit-accurate software reference,
compiler, RTL units, integrated RTL, and finally hardware. Performance work follows a
correct end-to-end path. The current compiler validates and imports static ONNX graphs,
then applies conservative, deterministic graph transformations. It does not yet emit a
schedule, deployment bundle, or accelerator commands. The quantized software path
executes Conv and Gemm with integer arithmetic, preserves integer data through MaxPool
and View, and makes the ARM floating-point Softmax boundary explicit.

Bit-accurate RTL simulation verifies the digital design and protocol behavior without a
board. Simulator cycle counts are observations of that simulation, but latency derived
from an assumed clock is an estimate. Vivado synthesis and implementation for
`xc7z020clg400-1` will provide tool-reported feasibility, utilization, and timing—not
physical performance or power.

The eventual hardware target remains the Digilent Zybo Z7-20 (Zynq-7020). Future board
integration uses AXI4-Lite for control, AXI Stream for payloads, and AXI DMA where
appropriate. See `future_board_port.md` for the replacement boundary.
