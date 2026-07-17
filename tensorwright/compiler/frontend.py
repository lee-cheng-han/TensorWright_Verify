"""ONNX validation, shape inference, and conversion to TensorWright IR."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import onnx
from onnx import AttributeProto, ModelProto, TensorProto, helper, numpy_helper

from tensorwright.compiler.errors import (
    ModelValidationError,
    StaticShapeError,
    UnsupportedOperatorError,
)
from tensorwright.compiler.ir import Graph, JsonValue, Operation, Tensor

SUPPORTED_OPERATORS = frozenset(
    {
        "Add",
        "BatchNormalization",
        "Constant",
        "Conv",
        "Flatten",
        "Gemm",
        "MaxPool",
        "Relu",
        "Reshape",
        "Softmax",
    }
)

_BACKENDS = {
    "Add": "compiler",
    "BatchNormalization": "compiler",
    "Constant": "compiler",
    "Conv": "fpga",
    "Flatten": "metadata",
    "Gemm": "arm",
    "MaxPool": "arm",
    "Relu": "fpga",
    "Reshape": "metadata",
    "Softmax": "arm",
}


def load_onnx(path: str | Path) -> Graph:
    """Load an ONNX file and convert it into validated TensorWright IR."""
    model_path = Path(path)
    if not model_path.is_file():
        raise ModelValidationError(f"ONNX model does not exist: {model_path}")
    try:
        model = onnx.load(model_path, load_external_data=True)
    except Exception as error:
        raise ModelValidationError(f"Could not load ONNX model: {error}") from error
    return import_onnx_model(model)


def import_onnx_model(model: ModelProto) -> Graph:
    """Validate, infer static shapes, and import an in-memory ONNX model."""
    try:
        onnx.checker.check_model(model, full_check=False)
        _validate_operator_whitelist(model)
        inferred = onnx.shape_inference.infer_shapes(
            model, check_type=True, strict_mode=True, data_prop=True
        )
        onnx.checker.check_model(inferred, full_check=True)
    except (onnx.checker.ValidationError, onnx.shape_inference.InferenceError) as error:
        raise ModelValidationError(f"Invalid ONNX model: {error}") from error

    opsets = _read_opsets(inferred)
    tensors = _read_tensors(inferred)
    operations = _read_operations(inferred)
    _mark_constant_outputs(tensors, operations)
    _link_graph(tensors, operations)
    return Graph(
        name=inferred.graph.name or "onnx_graph",
        opset_imports=opsets,
        inputs=[value.name for value in inferred.graph.input],
        outputs=[value.name for value in inferred.graph.output],
        tensors=tensors,
        operations=operations,
    )


def _read_opsets(model: ModelProto) -> dict[str, int]:
    opsets = {
        (item.domain or "ai.onnx"): int(item.version) for item in model.opset_import
    }
    if "ai.onnx" not in opsets:
        raise ModelValidationError("Model does not declare the default ONNX opset")
    if any(version <= 0 for version in opsets.values()):
        raise ModelValidationError("ONNX opset versions must be positive")
    return opsets


def _read_tensors(model: ModelProto) -> dict[str, Tensor]:
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    value_infos = (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    )
    tensors: dict[str, Tensor] = {}
    for value_info in value_infos:
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            raise StaticShapeError(f'Tensor "{value_info.name}" has no inferred shape')
        shape = _static_shape(value_info.name, tensor_type.shape.dim)
        dtype = _dtype_name(tensor_type.elem_type, value_info.name)
        initializer = initializers.get(value_info.name)
        tensors[value_info.name] = Tensor(
            name=value_info.name,
            shape=shape,
            original_dtype=dtype,
            compiled_dtype=dtype,
            layout=_default_layout(shape),
            is_constant=initializer is not None,
            constant_data=(
                _tensor_data(initializer) if initializer is not None else None
            ),
        )

    for name, initializer in initializers.items():
        if name not in tensors:
            dtype = _dtype_name(initializer.data_type, name)
            shape = [int(dimension) for dimension in initializer.dims]
            tensors[name] = Tensor(
                name=name,
                shape=shape,
                original_dtype=dtype,
                compiled_dtype=dtype,
                layout=_default_layout(shape),
                is_constant=True,
                constant_data=_tensor_data(initializer),
            )
    return tensors


def _read_operations(model: ModelProto) -> list[Operation]:
    operations: list[Operation] = []
    for index, node in enumerate(model.graph.node):
        name = node.name or f"{node.op_type.lower()}_{index}"
        if (
            node.domain not in {"", "ai.onnx"}
            or node.op_type not in SUPPORTED_OPERATORS
        ):
            qualified_type = (
                f"{node.domain}::{node.op_type}" if node.domain else node.op_type
            )
            raise UnsupportedOperatorError(name, qualified_type)
        operations.append(
            Operation(
                name=name,
                operation_type=node.op_type,
                inputs=[item for item in node.input if item],
                outputs=[item for item in node.output if item],
                attributes={
                    attribute.name: _attribute_value(attribute)
                    for attribute in node.attribute
                },
                hardware_supported=node.op_type in {"Conv", "Relu"},
                assigned_backend=_BACKENDS[node.op_type],
            )
        )
    return operations


def _validate_operator_whitelist(model: ModelProto) -> None:
    for index, node in enumerate(model.graph.node):
        name = node.name or f"{node.op_type.lower()}_{index}"
        if (
            node.domain not in {"", "ai.onnx"}
            or node.op_type not in SUPPORTED_OPERATORS
        ):
            qualified_type = (
                f"{node.domain}::{node.op_type}" if node.domain else node.op_type
            )
            raise UnsupportedOperatorError(name, qualified_type)


def _mark_constant_outputs(
    tensors: dict[str, Tensor], operations: list[Operation]
) -> None:
    constant_attributes = (
        "value",
        "value_float",
        "value_floats",
        "value_int",
        "value_ints",
        "value_string",
        "value_strings",
    )
    for operation in operations:
        if operation.operation_type != "Constant" or len(operation.outputs) != 1:
            continue
        output = tensors.get(operation.outputs[0])
        if output is None:
            continue
        for attribute_name in constant_attributes:
            if attribute_name in operation.attributes:
                output.is_constant = True
                output.constant_data = operation.attributes[attribute_name]
                break


def _link_graph(tensors: dict[str, Tensor], operations: list[Operation]) -> None:
    for operation in operations:
        for input_name in operation.inputs:
            if input_name not in tensors:
                raise ModelValidationError(
                    f'Node "{operation.name}" references unknown tensor "{input_name}"'
                )
            tensors[input_name].consumers.append(operation.name)
        for output_name in operation.outputs:
            if output_name not in tensors:
                raise StaticShapeError(
                    f'Tensor "{output_name}" has no statically inferred shape'
                )
            if tensors[output_name].producer is not None:
                raise ModelValidationError(
                    f'Tensor "{output_name}" has more than one producer'
                )
            tensors[output_name].producer = operation.name


def _static_shape(name: str, dimensions: Iterable[Any]) -> list[int]:
    shape: list[int] = []
    for dimension in dimensions:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            detail = dimension.dim_param or "unknown"
            raise StaticShapeError(
                f'Tensor "{name}" must have a positive static shape; found {detail}'
            )
        shape.append(int(dimension.dim_value))
    return shape


def _dtype_name(element_type: int, tensor_name: str) -> str:
    try:
        return str(TensorProto.DataType.Name(element_type)).lower()
    except ValueError as error:
        raise ModelValidationError(
            f'Tensor "{tensor_name}" has unknown ONNX dtype {element_type}'
        ) from error


def _default_layout(shape: list[int]) -> str:
    if len(shape) == 4:
        return "NCHW"
    if not shape:
        return "SCALAR"
    return "UNSPECIFIED"


def _tensor_data(tensor: TensorProto) -> JsonValue:
    return _json_value(numpy_helper.to_array(tensor).tolist())


def _attribute_value(attribute: AttributeProto) -> JsonValue:
    value = helper.get_attribute_value(attribute)
    if isinstance(value, TensorProto):
        return _tensor_data(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return _json_value(value)


def _json_value(value: Any) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise ModelValidationError(
        f"ONNX value of type {type(value).__name__} is not JSON serializable"
    )
