# Compiler IR contract

Milestone 2 converts ONNX protocol objects into a TensorWright-owned graph rather than
transforming ONNX objects in place. Tensor records carry identity, shape,
source and compiled dtypes, layout, quantization metadata, producer/consumer links,
constant data, memory placement, and lifetime. Operation records will carry identity,
type, inputs, outputs, attributes, hardware support, backend assignment, fusions, and
an operation-count estimate. Graph records contain their name, domain-specific opset
imports, ordered model inputs and outputs, tensors, and topologically ordered operations.

Schedule records identify the operation, tile dimensions, channel parallelism,
buffers, and separately labeled compute- and transfer-cycle estimates. Fields should be
added only when consumed by a pass or backend.

`Graph.to_dict()` and `Graph.to_json()` produce deterministic, JSON-compatible
diagnostic serialization. Tensor keys and opset domains are sorted, while operation,
input, output, and consumer ordering follows the ONNX graph. This representation is an
IR diagnostic format, not yet the versioned deployment `graph.json` schema.

The frontend runs the ONNX checker before and after strict shape inference. Every
tensor consumed or produced by a supported node must have a positive, fully static
shape. Initializers become constant tensors with JSON-compatible data. ONNX protocol
objects do not remain in the resulting IR.

Optimization passes preserve operation order, rebuild producer-consumer links, and
return deep-copied graphs. Fused operation names remain on the surviving operation for
diagnostics. Static Flatten and Reshape operations canonicalize to the internal `View`
operation; this internal type is not accepted directly from ONNX. See
`optimization_passes.md` for pass order and eligibility rules.
