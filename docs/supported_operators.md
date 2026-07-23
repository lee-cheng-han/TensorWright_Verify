# Supported operators

The MVP uses an explicit whitelist. Encountering any other ONNX operation is a
compile-time error that identifies the node and operation. Supported non-RTL operations
are explicitly assigned to the ARM fallback.

| Operation | Current lowering |
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

Initial graphs must have batch size one and static shapes. The generic software convolution
path supports static parameters; the native RTL bundle runner currently supports valid
3x3, stride-one `1x3x5x5 -> 1x2x3x3` convolution. Dynamic
shapes, recurrent networks, attention, transformers, training, and floating-point RTL
are outside MVP scope.

The frontend accepts the whitelist at import time. Milestone 3 folds an `Add` only when
it is a constant channel bias immediately following a single-consumer Conv, with bias
layout `[C]` or `[1,C,1,1]`. Other `Add` nodes remain visible and compiler-assigned; no
later execution support is claimed for them yet. BatchNormalization similarly remains
visible unless every parameter is constant and its Conv producer is safe to rewrite.

Nodes in custom domains are rejected even when their short operation name matches the
whitelist. Native RTL execution rejects bundles outside its documented fixed-shape contract
before simulation.

Under TensorWright Verify, operator support has three separate meanings: the Python
reference can produce semantic traces, the custom RTL can expose a corresponding stage,
and the alignment engine can compare them. Current trace generation covers outputs
executable by the quantized Python backend and the custom convolution RTL output stream.
Other operator stages do not imply an RTL trace point or cross-backend alignment yet.
