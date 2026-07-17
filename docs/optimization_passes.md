# Optimization passes

TensorWright runs Milestone 3 passes in a fixed, deterministic order. Each pass accepts
a graph, deep-copies it, and returns the transformed graph; caller-owned IR is never
mutated. The pipeline emits operation and tensor counts through Python debug logging.

1. `FoldConstants` removes imported `Constant` operations whose output data was already
   extracted and evaluates `Add` only when both operands are constant and NumPy
   broadcasting produces the statically inferred output shape.
2. `FoldBatchNormalization` folds a single-consumer inference BatchNormalization into
   its Conv producer. Scale, offset, mean, variance, weights, and any existing Conv bias
   must be constant. New private weight and bias tensors prevent mutation of shared
   initializers.
3. `FuseConvBiasRelu` recognizes a single-consumer Conv followed by an optional constant
   channel-bias Add and optional ReLU. Bias layout must be `[C]` or `[1,C,1,1]`. Fusion
   stops when an intermediate is a graph output or has multiple consumers.
4. `CanonicalizeShapeOperations` converts statically inferred Flatten and Reshape nodes
   to metadata-only `View` operations with an explicit target shape.
5. `EliminateDeadCode` performs backward liveness from graph outputs and removes pure
   operations and tensors that cannot affect those outputs.
6. `AssignBackends` annotates Conv and remaining Relu operations for FPGA, View for
   metadata handling, MaxPool/Gemm/Softmax for ARM, compiler-time operations for the
   compiler, and unknown IR operations as unsupported.

Passes skip patterns whose safety conditions are not proven. A skipped fold is not an
error and remains visible in IR for diagnostics or later handling. Milestone 3 does not
evaluate arbitrary ONNX expressions, rewrite dynamic shapes, quantize values, or choose
hardware schedules.
