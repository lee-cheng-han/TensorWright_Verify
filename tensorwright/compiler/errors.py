"""User-facing compiler exceptions."""

from __future__ import annotations


class CompilerError(Exception):
    """Base class for expected TensorWright compilation failures."""


class ModelValidationError(CompilerError):
    """Raised when an ONNX model is missing or structurally invalid."""


class StaticShapeError(CompilerError):
    """Raised when a tensor does not have a fully known static shape."""


class UnsupportedOperatorError(CompilerError):
    """Raised when an ONNX node is outside the explicit MVP whitelist."""

    def __init__(self, node_name: str, operation_type: str) -> None:
        self.node_name = node_name
        self.operation_type = operation_type
        super().__init__(
            f'Compilation failed at node "{node_name}".\n\n'
            f"Operator: {operation_type}\n"
            "Reason: TensorWright MVP supports only Add, BatchNormalization, "
            "Constant, Conv, Flatten, Gemm, MaxPool, Relu, Reshape, and Softmax.\n\n"
            "Suggested action:\n"
            "- Remove or replace the operation.\n"
            "- Export a supported model.\n"
            "- Assign the operation to a future CPU fallback backend."
        )
