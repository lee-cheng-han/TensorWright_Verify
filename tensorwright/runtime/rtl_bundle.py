"""Extract a supported convolution invocation from a deployment bundle."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tensorwright.compiler import BundleValidationError, load_bundle
from tensorwright.compiler.backend.bundle import COMMAND_STRUCT


@dataclass(frozen=True)
class FixedConvolutionInvocation:
    """Concrete values consumed by the current fixed-shape RTL engine."""

    model: str
    operation: str
    source_operation_id: str
    weights: list[int]
    activations: list[int]
    biases: list[int]
    multipliers: list[int]
    shifts: list[int]
    relu: list[bool]
    expected: list[int]


def extract_fixed_convolution(path: str | Path) -> FixedConvolutionInvocation:
    """Decode a 5x5x3-to-3x3x2 Conv bundle for real RTL execution."""
    bundle = load_bundle(path)
    operations = bundle.graph["operations"]
    if len(operations) != 1 or operations[0]["operation_type"] != "Conv":
        raise BundleValidationError("RTL bundle execution requires exactly one Conv")
    operation: dict[str, Any] = operations[0]
    if operation.get("assigned_backend") != "fpga":
        raise BundleValidationError("Convolution is not assigned to the FPGA backend")
    input_name, weight_name, bias_name = operation["inputs"]
    output_name = operation["outputs"][0]
    tensors = bundle.graph["tensors"]
    if (
        tensors[input_name]["shape"] != [1, 3, 5, 5]
        or tensors[weight_name]["shape"] != [2, 3, 3, 3]
        or tensors[bias_name]["shape"] != [2]
        or tensors[output_name]["shape"] != [1, 2, 3, 3]
    ):
        raise BundleValidationError(
            "Current RTL runner supports only 1x3x5x5 -> 1x2x3x3 convolution"
        )
    locations = bundle.manifest["constant_locations"]
    weights = _read_array(
        bundle.path / "weights.bin", locations[weight_name], np.dtype("<i1")
    )
    biases = _read_array(
        bundle.path / "biases.bin", locations[bias_name], np.dtype("<i4")
    )
    quantization_offset = _command_quantization_offset(bundle.path / "commands.bin")
    quantization = (bundle.path / "quantization.bin").read_bytes()
    multipliers: list[int] = []
    shifts: list[int] = []
    record = struct.Struct("<IB3x")
    for channel in range(2):
        multiplier, shift = record.unpack_from(
            quantization, quantization_offset + channel * record.size
        )
        multipliers.append(multiplier)
        shifts.append(shift)
    input_scale = tensors[input_name].get("quantization_scale")
    if not isinstance(input_scale, float) or input_scale <= 0:
        raise BundleValidationError("Bundle input lacks a positive scalar scale")
    float_input = np.frombuffer(
        (bundle.path / "reference_input.bin").read_bytes(), dtype="<f4"
    )
    if float_input.size != 75:
        raise BundleValidationError("Bundle reference input has the wrong size")
    activations = np.clip(
        np.copysign(np.floor(np.abs(float_input / input_scale) + 0.5), float_input),
        -128,
        127,
    ).astype(np.int8)
    expected = np.frombuffer(
        (bundle.path / "reference_output.bin").read_bytes(), dtype="<i1"
    )
    if expected.size != 18:
        raise BundleValidationError("Bundle reference output has the wrong size")
    flags = _command_flags(bundle.path / "commands.bin")
    return FixedConvolutionInvocation(
        model=str(bundle.manifest["model"]),
        operation=str(operation["name"]),
        source_operation_id=str(operation["source_operation_id"]),
        weights=[int(value) for value in weights],
        activations=[int(value) for value in activations],
        biases=[int(value) for value in biases],
        multipliers=multipliers,
        shifts=shifts,
        relu=[bool(flags & 1)] * 2,
        expected=[int(value) for value in expected],
    )


def write_convolution_vector(
    invocation: FixedConvolutionInvocation, path: str | Path
) -> Path:
    """Write the self-checking testbench format from decoded bundle values."""
    destination = Path(path)
    values = [
        *invocation.biases,
        *invocation.multipliers,
        *invocation.shifts,
        *[int(value) for value in invocation.relu],
        *invocation.weights,
        *invocation.activations,
        *invocation.expected,
    ]
    destination.write_text(
        "1\n" + " ".join(str(value) for value in values) + "\n",
        encoding="utf-8",
    )
    return destination


def _read_array(
    path: Path, location: dict[str, int | str], dtype: np.dtype[Any]
) -> np.ndarray[Any, Any]:
    offset = int(location["offset"])
    size = int(location["size"])
    data = path.read_bytes()[offset : offset + size]
    if len(data) != size:
        raise BundleValidationError(f"Constant data is truncated in {path.name}")
    return np.frombuffer(data, dtype=dtype)


def _command_values(path: Path) -> tuple[int, ...]:
    data = path.read_bytes()
    if len(data) != COMMAND_STRUCT.size:
        raise BundleValidationError("RTL bundle requires exactly one command")
    return COMMAND_STRUCT.unpack(data)


def _command_quantization_offset(path: Path) -> int:
    return _command_values(path)[6]


def _command_flags(path: Path) -> int:
    return _command_values(path)[7] & 0xFFFF
