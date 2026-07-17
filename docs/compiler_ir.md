# Compiler IR contract

Milestone 2 will convert ONNX protocol objects into a TensorWright-owned graph rather
than transforming ONNX objects in place. Tensor records will carry identity, shape,
source and compiled dtypes, layout, quantization metadata, producer/consumer links,
constant data, memory placement, and lifetime. Operation records will carry identity,
type, inputs, outputs, attributes, hardware support, backend assignment, fusions, and
an operation-count estimate.

Schedule records will identify the operation, tile dimensions, channel parallelism,
buffers, and separately labeled compute- and transfer-cycle estimates. Fields should be
added only when consumed by a pass or backend. Serialization schemas are deferred until
the typed IR is implemented and tested.
