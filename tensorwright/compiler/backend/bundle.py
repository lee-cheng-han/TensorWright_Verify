"""Versioned, deterministic `.twmodel` bundle generation and validation."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tensorwright.compiler.errors import BundleValidationError
from tensorwright.compiler.ir import Graph, Operation, Tensor
from tensorwright.compiler.quantization import CompilationResult, execute_quantized

FORMAT_VERSION = 1
COMMAND_VERSION = 1
HARDWARE_INTERFACE_VERSION = "1.0"
ALIGNMENT = 64
COMMAND_STRUCT = struct.Struct("<8I")
REQUIRED_FILES = (
    "graph.json",
    "commands.bin",
    "weights.bin",
    "biases.bin",
    "quantization.bin",
    "constants.bin",
    "memory_plan.json",
    "schedule.json",
    "labels.txt",
    "reference_input.bin",
    "reference_output.bin",
    "compilation_report.json",
)


@dataclass(frozen=True)
class BundleContents:
    """Validated bundle metadata and location."""

    path: Path
    manifest: dict[str, Any]
    graph: dict[str, Any]
    memory_plan: dict[str, Any]
    schedule: dict[str, Any]


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _align(value: int) -> int:
    return math.ceil(value / ALIGNMENT) * ALIGNMENT


def _tensor_bytes(tensor: Tensor) -> bytes:
    if tensor.constant_data is None:
        return b""
    dtype = {
        "int8": "<i1",
        "int32": "<i4",
        "float": "<f4",
        "float32": "<f4",
        "float64": "<f8",
    }.get(tensor.compiled_dtype)
    if dtype is None:
        raise BundleValidationError(
            f'Cannot pack tensor "{tensor.name}" with dtype {tensor.compiled_dtype}'
        )
    return np.asarray(tensor.constant_data, dtype=dtype).tobytes(order="C")


def _memory_plan(graph: Graph) -> dict[str, Any]:
    offsets: dict[str, int] = {}
    cursor = 0
    for name in [
        *graph.inputs,
        *[name for op in graph.operations for name in op.outputs],
    ]:
        if name in offsets or graph.tensors[name].is_constant:
            continue
        tensor = graph.tensors[name]
        element_size = 1 if tensor.compiled_dtype == "int8" else 4
        size = math.prod(tensor.shape) * element_size
        cursor = _align(cursor)
        offsets[name] = cursor
        cursor += size
    allocations = [
        {
            "tensor": name,
            "offset": offset,
            "size": math.prod(graph.tensors[name].shape)
            * (1 if graph.tensors[name].compiled_dtype == "int8" else 4),
            "alignment": ALIGNMENT,
        }
        for name, offset in offsets.items()
    ]
    return {
        "format_version": FORMAT_VERSION,
        "alignment": ALIGNMENT,
        "scratch_size": _align(cursor),
        "allocations": allocations,
    }


def _pack_constants(graph: Graph) -> tuple[dict[str, bytes], dict[str, dict[str, int]]]:
    blobs = {"weights": bytearray(), "biases": bytearray(), "constants": bytearray()}
    locations: dict[str, dict[str, int]] = {}
    linear_inputs = {
        name: kind
        for operation in graph.operations
        if operation.operation_type in {"Conv", "Gemm"}
        for name, kind in zip(operation.inputs[1:3], ("weights", "biases"), strict=True)
    }
    for name, tensor in sorted(graph.tensors.items()):
        if not tensor.is_constant:
            continue
        kind = linear_inputs.get(name, "constants")
        blob = blobs[kind]
        padding = _align(len(blob)) - len(blob) if blob else 0
        blob.extend(b"\0" * padding)
        data = _tensor_bytes(tensor)
        locations[name] = {"file": kind, "offset": len(blob), "size": len(data)}
        blob.extend(data)
    return {name: bytes(value) for name, value in blobs.items()}, locations


def _schedule(graph: Graph, memory_plan: dict[str, Any]) -> dict[str, Any]:
    offsets = {item["tensor"]: item["offset"] for item in memory_plan["allocations"]}
    layers = []
    for index, operation in enumerate(graph.operations):
        output_elements = math.prod(graph.tensors[operation.outputs[0]].shape)
        input_elements = sum(
            math.prod(graph.tensors[name].shape) for name in operation.inputs
        )
        compute_cycles = operation.estimated_operation_count or output_elements
        if operation.operation_type == "Conv":
            weight_shape = graph.tensors[operation.inputs[1]].shape
            compute_cycles = output_elements * weight_shape[1]
        layers.append(
            {
                "index": index,
                "operation": operation.name,
                "type": operation.operation_type,
                "backend": operation.assigned_backend,
                "inputs": operation.inputs,
                "outputs": operation.outputs,
                "input_offset": offsets.get(operation.inputs[0]),
                "output_offset": offsets.get(operation.outputs[0]),
                "estimated_compute_cycles": compute_cycles,
                "estimated_transfer_bytes": input_elements + output_elements,
            }
        )
    return {"format_version": FORMAT_VERSION, "layers": layers}


def _pack_quantization(graph: Graph) -> tuple[bytes, dict[str, int]]:
    data = bytearray()
    offsets: dict[str, int] = {}
    for operation in graph.operations:
        if operation.operation_type not in {"Conv", "Gemm"}:
            continue
        multipliers = operation.attributes.get("requantization_multipliers")
        shifts = operation.attributes.get("requantization_shifts")
        if not isinstance(multipliers, list) or not isinstance(shifts, list):
            raise BundleValidationError(
                f'Operation "{operation.name}" lacks quantization'
            )
        offsets[operation.name] = len(data)
        for multiplier, shift in zip(multipliers, shifts, strict=True):
            data.extend(struct.pack("<IB3x", int(multiplier), int(shift)))
    return bytes(data), offsets


def _command(
    operation: Operation,
    index: int,
    memory_offsets: dict[str, int],
    constant_locations: dict[str, dict[str, int]],
    quant_offsets: dict[str, int],
) -> bytes:
    opcode = {"Conv": 1, "MaxPool": 2, "View": 3, "Gemm": 4, "Softmax": 5}.get(
        operation.operation_type, 255
    )
    backend = {"fpga": 1, "arm": 2, "metadata": 3}.get(operation.assigned_backend, 0)
    weight_offset = (
        constant_locations.get(operation.inputs[1], {}).get("offset", 0)
        if len(operation.inputs) > 1
        else 0
    )
    bias_offset = (
        constant_locations.get(operation.inputs[2], {}).get("offset", 0)
        if len(operation.inputs) > 2
        else 0
    )
    flags = int("Relu" in operation.fused_operations)
    return COMMAND_STRUCT.pack(
        opcode,
        backend,
        memory_offsets.get(operation.inputs[0], 0),
        memory_offsets.get(operation.outputs[0], 0),
        weight_offset,
        bias_offset,
        quant_offsets.get(operation.name, 0),
        flags | (index << 16),
    )


def build_bundle(
    result: CompilationResult,
    output_path: str | Path,
    reference_inputs: dict[str, np.ndarray],
    *,
    labels: list[str] | None = None,
) -> Path:
    """Create a complete deterministic deployment directory."""
    path = Path(output_path)
    if path.suffix != ".twmodel":
        raise BundleValidationError("Deployment bundle path must end in .twmodel")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise BundleValidationError(f"Bundle directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    graph = result.graph
    plan = _memory_plan(graph)
    schedule = _schedule(graph, plan)
    constants, locations = _pack_constants(graph)
    quantization, quant_offsets = _pack_quantization(graph)
    memory_offsets = {item["tensor"]: item["offset"] for item in plan["allocations"]}
    commands = b"".join(
        _command(operation, index, memory_offsets, locations, quant_offsets)
        for index, operation in enumerate(graph.operations)
    )
    input_blob = b"".join(
        np.asarray(reference_inputs[name], dtype="<f4").tobytes(order="C")
        for name in graph.inputs
    )
    outputs = execute_quantized(graph, reference_inputs)
    output_blob = b"".join(
        np.asarray(
            outputs[name], dtype="<f4" if outputs[name].dtype.kind == "f" else "<i1"
        ).tobytes(order="C")
        for name in graph.outputs
    )
    files = {
        "graph.json": graph.to_json().encode(),
        "commands.bin": commands,
        "weights.bin": constants["weights"],
        "biases.bin": constants["biases"],
        "quantization.bin": quantization,
        "constants.bin": constants["constants"],
        "memory_plan.json": _json_bytes(plan),
        "schedule.json": _json_bytes(schedule),
        "labels.txt": (("\n".join(labels) + "\n") if labels else "").encode(),
        "reference_input.bin": input_blob,
        "reference_output.bin": output_blob,
        "compilation_report.json": result.report_json().encode(),
    }
    for name, data in files.items():
        (path / name).write_bytes(data)
    manifest = {
        "format_version": FORMAT_VERSION,
        "command_version": COMMAND_VERSION,
        "hardware_interface_version": HARDWARE_INTERFACE_VERSION,
        "model": graph.name,
        "target": "tensorwright-simulation",
        "layer_count": len(graph.operations),
        "command_record_size": COMMAND_STRUCT.size,
        "scratch_memory_bytes": plan["scratch_size"],
        "constant_locations": locations,
        "files": {
            name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in sorted(files.items())
        },
    }
    (path / "manifest.json").write_bytes(_json_bytes(manifest))
    validate_bundle(path)
    return path


def validate_bundle(path: str | Path) -> None:
    """Validate versions, required files, checksums, sizes, and core schemas."""
    bundle = Path(path)
    if bundle.suffix != ".twmodel" or not bundle.is_dir():
        raise BundleValidationError("Bundle must be an existing .twmodel directory")
    try:
        manifest = json.loads((bundle / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise BundleValidationError("Missing or invalid manifest.json") from error
    if (
        manifest.get("format_version") != FORMAT_VERSION
        or manifest.get("command_version") != COMMAND_VERSION
    ):
        raise BundleValidationError("Unsupported bundle or command format version")
    if manifest.get("hardware_interface_version") != HARDWARE_INTERFACE_VERSION:
        raise BundleValidationError("Incompatible hardware interface version")
    for name in REQUIRED_FILES:
        metadata = manifest.get("files", {}).get(name)
        file_path = bundle / name
        if not isinstance(metadata, dict) or not file_path.is_file():
            raise BundleValidationError(f"Missing required bundle file: {name}")
        data = file_path.read_bytes()
        if (
            metadata.get("size") != len(data)
            or metadata.get("sha256") != hashlib.sha256(data).hexdigest()
        ):
            raise BundleValidationError(
                f"Bundle file checksum or size mismatch: {name}"
            )
    if (bundle / "commands.bin").stat().st_size % COMMAND_STRUCT.size:
        raise BundleValidationError("commands.bin has a partial command record")
    try:
        plan = json.loads((bundle / "memory_plan.json").read_text())
        schedule = json.loads((bundle / "schedule.json").read_text())
        graph = json.loads((bundle / "graph.json").read_text())
    except json.JSONDecodeError as error:
        raise BundleValidationError("Bundle JSON file is invalid") from error
    if any(item["offset"] % ALIGNMENT for item in plan.get("allocations", [])):
        raise BundleValidationError("Memory-plan allocation is not aligned")
    if len(schedule.get("layers", [])) != manifest.get("layer_count"):
        raise BundleValidationError("Schedule layer count does not match manifest")
    if len(graph.get("operations", [])) != manifest.get("layer_count"):
        raise BundleValidationError("Graph layer count does not match manifest")


def load_bundle(path: str | Path) -> BundleContents:
    """Validate and load runtime-neutral bundle metadata."""
    bundle = Path(path)
    validate_bundle(bundle)
    return BundleContents(
        bundle,
        json.loads((bundle / "manifest.json").read_text()),
        json.loads((bundle / "graph.json").read_text()),
        json.loads((bundle / "memory_plan.json").read_text()),
        json.loads((bundle / "schedule.json").read_text()),
    )
