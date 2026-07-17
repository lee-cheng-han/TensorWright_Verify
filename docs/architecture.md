# TensorWright Verify architecture

TensorWright Verify answers why quantized software and accelerator execution disagree:

```text
ONNX or QONNX model
        |
        +-> quantized Python reference -> canonical reference trace --+
        |                                                       future |
        +-> custom RTL -> Cocotb simulation -> hardware trace --------+-> alignment
                                                                         -> first divergence
                                                                         -> diagnosis
                                                                         -> minimization
                                                                         -> regression
```

Milestones 11 and 12 implement the canonical trace contract, Python-reference writer,
and accepted-transfer capture at the convolution RTL output. Alignment and debugging
stages follow in Milestone 13 and later.

## Preserved supporting infrastructure

The ONNX frontend and compiler IR provide stable operations, graph connectivity, tensor
shapes/layouts, fused groups, and quantization metadata. The integer backend supplies
golden values. `.twmodel` retains graphs, packed constants, schedules, reference vectors,
and interface versions. The runtime provides register/stream orchestration, seeded
backpressure, timeouts, CPU fallbacks, and a future insertion point for hardware events.
The custom SystemVerilog accelerator remains the primary integration and controlled
fault-injection target.

Compilation is now a lower-level workload and metadata preparation mechanism, not the
primary product. Existing bundle and `simulate` workflows remain compatible. No FINN or
hls4ml adapter exists; future adapters must convert real tested traces into the same
canonical schema.
