# Supported operators

The MVP uses an explicit whitelist. Encountering any other ONNX operation is a
compile-time error that identifies the node and operation and suggests exporting a
supported graph or using a future CPU fallback.

| Operation | Planned MVP lowering |
| --- | --- |
| `Conv` | FPGA accelerator |
| `Relu` | Fuse into FPGA post-processing |
| `Add` used for bias | Fold or fuse during optimization |
| `BatchNormalization` | Fold into convolution weights and bias |
| `MaxPool` | ARM fallback initially |
| `Flatten` | Metadata-only transformation |
| `Reshape` | Metadata-only when static and valid |
| `Gemm` | ARM fallback initially |
| `Softmax` | ARM fallback |
| `Constant` | Extract into compiler-owned tensor data |

Initial graphs must have batch size one and static shapes. The planned convolution
subset uses 1x1 or 3x3 kernels, stride one or two, and static zero padding. Dynamic
shapes, recurrent networks, attention, transformers, training, and floating-point RTL
are outside MVP scope.

The frontend accepts the whitelist at import time. Milestone 3 folds an `Add` only when
it is a constant channel bias immediately following a single-consumer Conv, with bias
layout `[C]` or `[1,C,1,1]`. Other `Add` nodes remain visible and compiler-assigned; no
later execution support is claimed for them yet. BatchNormalization similarly remains
visible unless every parameter is constant and its Conv producer is safe to rewrite.

Operation-level validation such as convolution kernel and stride restrictions remains
future work. Nodes in custom domains are rejected even when their short operation name
matches the whitelist.

Under TensorWright Verify, operator support has three separate meanings: the Python
reference can produce semantic traces, the custom RTL can expose a corresponding stage,
and the future alignment engine can compare them. Current M11 trace generation covers
outputs executable by the quantized Python backend. This does not imply an RTL trace
point or cross-backend alignment exists yet.
