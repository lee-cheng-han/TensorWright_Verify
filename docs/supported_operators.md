# Supported operators

The MVP uses an explicit whitelist. Encountering any other ONNX operation is a
compile-time error that identifies the node and operation and suggests exporting a
supported graph or using a future CPU fallback.

| Operation | Planned MVP lowering |
| --- | --- |
| `Conv` | FPGA accelerator |
| `Relu` | Fuse into FPGA post-processing |
| Bias add | Fuse into convolution |
| `BatchNormalization` | Fold into convolution weights and bias |
| `MaxPool` | ARM fallback initially |
| `Flatten` | Metadata-only transformation |
| `Reshape` | Metadata-only when static and valid |
| `Gemm` | ARM fallback initially |
| `Softmax` | ARM fallback |

Initial graphs must have batch size one and static shapes. The planned convolution
subset uses 1x1 or 3x3 kernels, stride one or two, and static zero padding. Dynamic
shapes, recurrent networks, attention, transformers, training, and floating-point RTL
are outside MVP scope.
