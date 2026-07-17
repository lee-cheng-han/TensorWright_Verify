# Architecture

TensorWright is organized as an explicitly staged hardware/software stack:

```text
ONNX model -> compiler IR -> optimization and INT8 quantization
           -> memory plan and schedule -> .twmodel bundle
           -> ARM runtime -> reusable FPGA layer engine
```

The compiler owns graph validation, numerical lowering, backend partitioning, memory
planning, and command generation. The ARM runtime validates bundles and hardware
versions, manages transfers, invokes FPGA layers, and executes supported fallbacks.
The reusable SystemVerilog accelerator performs configured layers rather than embedding
a particular network.

Correctness advances in this order: specification, bit-accurate software reference,
compiler, RTL units, integrated RTL, and finally hardware. Performance work follows a
correct end-to-end path. The current compiler validates and imports static ONNX graphs,
then applies conservative, deterministic graph transformations. It does not yet emit a
quantized model, schedule, deployment bundle, or accelerator commands.

The primary target is a Digilent Zybo Z7-20 (Zynq-7020). Planned board integration uses
AXI4-Lite for control, AXI Stream for payloads, and AXI DMA where appropriate. Linux on
WSL is the initial host development environment.
