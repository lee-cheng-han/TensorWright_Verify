# TensorWright Verify architecture

TensorWright Verify answers why quantized software and accelerator execution disagree:

```text
ONNX model -> typed IR -> optimization -> calibration/INT8 quantization
                                        |
                                        +-> .twmodel bundle
                                             |             |
                                             |             +-> ARM fallback
                                             +-> native RTL convolution
                                                  |
Python reference trace ----------------------+    +-> Verilator trace
                                             \   /
                                              alignment
                                                -> diagnosis
                                                -> minimization
                                                -> regression/dashboard
```

Milestones 11–19 implement the canonical trace contract, Python-reference writer,
accepted-transfer capture at the convolution RTL output, semantic alignment, and first-
divergence detection, deterministic numerical and protocol diagnosis, and failure-preserving
input minimization, portable Cocotb regression generation, offline debugging dashboards, and a
versioned trace-adapter plugin API, a tested FINN full execution-context adapter, and a
tested hls4ml C-simulation trace adapter.

## Preserved supporting infrastructure

The ONNX frontend and compiler IR provide stable operations, graph connectivity, tensor
shapes/layouts, fused groups, and quantization metadata. The integer backend supplies
golden values. `.twmodel` retains graphs, packed constants, schedules, reference vectors,
and interface versions. The runtime provides register/stream orchestration, seeded
backpressure, timeouts, CPU fallbacks, and a future insertion point for hardware events.
The custom SystemVerilog accelerator is the native FPGA backend and controlled
fault-injection target. The fixed-shape bundle runner decodes compiler-generated binary
constants and quantization records and executes them on this RTL.

Compilation, execution, and verification are equal parts of the product. FINN full-context
and hls4ml C-simulation traces are supported through adapters, while only the native
TensorWright convolution backend claims cycle-level RTL evidence.
