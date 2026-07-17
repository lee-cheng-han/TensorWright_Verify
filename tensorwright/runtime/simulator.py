"""Generic command-driven simulation host for version-1 deployment bundles."""

from __future__ import annotations

import json
import random
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tensorwright.compiler.backend.bundle import COMMAND_STRUCT, load_bundle
from tensorwright.compiler.ir import Graph, Operation, Tensor
from tensorwright.compiler.quantization import execute_quantized


class SimulationError(RuntimeError):
    """Base class for runtime failures with no successful completion."""


class SimulationTimeoutError(SimulationError):
    """Raised when a simulated layer exceeds its cycle budget."""


@dataclass(frozen=True)
class SimulationConfig:
    """Deterministic simulation and legal-backpressure controls."""

    seed: int = 0x7E45
    timeout_cycles: int = 1_000_000
    randomized_backpressure: bool = True
    ready_probability: float = 0.75

    def __post_init__(self) -> None:
        if self.timeout_cycles <= 0:
            raise ValueError("timeout_cycles must be positive")
        if not 0.0 < self.ready_probability <= 1.0:
            raise ValueError("ready_probability must be in (0, 1]")


@dataclass
class PerformanceCounters:
    total_cycles: int = 0
    compute_active_cycles: int = 0
    input_stalls: int = 0
    output_stalls: int = 0
    weight_load_cycles: int = 0
    output_count: int = 0
    input_count: int = 0
    layer_invocations: int = 0
    executed_macs: int = 0
    error_count: int = 0


@dataclass(frozen=True)
class Command:
    opcode: int
    backend: int
    input_offset: int
    output_offset: int
    weight_offset: int
    bias_offset: int
    quantization_offset: int
    flags_and_index: int


@dataclass(frozen=True)
class LayerTrace:
    index: int
    operation: str
    backend: str
    start_cycle: int
    end_cycle: int
    input_transfers: int
    output_transfers: int


@dataclass(frozen=True)
class SimulationResult:
    """Complete outputs and observable runtime activity."""

    outputs: dict[str, np.ndarray]
    counters: PerformanceCounters
    layers: list[LayerTrace]
    register_transactions: list[dict[str, int | str]]
    seed: int
    reference_match: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "reference_match": self.reference_match,
            "counters": asdict(self.counters),
            "layers": [asdict(layer) for layer in self.layers],
            "register_transactions": self.register_transactions,
            "outputs": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "values": value.tolist(),
                }
                for name, value in self.outputs.items()
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SimulatedMemory:
    def __init__(self, size: int) -> None:
        if size < 0:
            raise SimulationError("Scratch-memory size cannot be negative")
        self._data = bytearray(size)

    def write(self, offset: int, data: bytes) -> None:
        if offset < 0 or offset + len(data) > len(self._data):
            raise SimulationError("Simulated-memory write is out of bounds")
        self._data[offset : offset + len(data)] = data

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or size < 0 or offset + size > len(self._data):
            raise SimulationError("Simulated-memory read is out of bounds")
        return bytes(self._data[offset : offset + size])


class RegisterModel:
    DEVICE_ID = 0x54570001
    VERSION = 0x00010000

    def __init__(self) -> None:
        self.transactions: list[dict[str, int | str]] = []

    def read(self, address: int, value: int) -> int:
        self.transactions.append({"kind": "read", "address": address, "value": value})
        return value

    def write(self, address: int, value: int) -> None:
        self.transactions.append({"kind": "write", "address": address, "value": value})


def _graph_from_dict(data: dict[str, Any]) -> Graph:
    return Graph(
        name=str(data["name"]),
        opset_imports={
            str(key): int(value) for key, value in data["opset_imports"].items()
        },
        inputs=[str(value) for value in data["inputs"]],
        outputs=[str(value) for value in data["outputs"]],
        tensors={name: Tensor(**tensor) for name, tensor in data["tensors"].items()},
        operations=[Operation(**operation) for operation in data["operations"]],
    )


def _decode_commands(data: bytes) -> list[Command]:
    return [
        Command(*values) for values in struct.iter_unpack(COMMAND_STRUCT.format, data)
    ]


def _reference_inputs(bundle_path: Path, graph: Graph) -> dict[str, np.ndarray]:
    data = (bundle_path / "reference_input.bin").read_bytes()
    cursor = 0
    inputs: dict[str, np.ndarray] = {}
    for name in graph.inputs:
        shape = tuple(graph.tensors[name].shape)
        size = int(np.prod(shape)) * 4
        if cursor + size > len(data):
            raise SimulationError("reference_input.bin is truncated")
        inputs[name] = (
            np.frombuffer(data[cursor : cursor + size], dtype="<f4")
            .reshape(shape)
            .copy()
        )
        cursor += size
    if cursor != len(data):
        raise SimulationError("reference_input.bin contains trailing data")
    return inputs


def _tensor_bytes(value: np.ndarray) -> bytes:
    if value.dtype.kind == "f":
        return np.asarray(value, dtype="<f4").tobytes(order="C")
    return np.asarray(value, dtype="<i1").tobytes(order="C")


def _validate_command_payload(
    bundle_path: Path, command: Command, graph: Graph, operation: Operation
) -> None:
    expected_opcode = {"Conv": 1, "MaxPool": 2, "View": 3, "Gemm": 4, "Softmax": 5}.get(
        operation.operation_type
    )
    if command.opcode != expected_opcode:
        raise SimulationError(f'Opcode mismatch for operation "{operation.name}"')
    if operation.operation_type not in {"Conv", "Gemm"}:
        return
    weight_tensor = graph.tensors[operation.inputs[1]]
    expected_weights = np.asarray(weight_tensor.constant_data, dtype="<i1").tobytes(
        order="C"
    )
    weights = (bundle_path / "weights.bin").read_bytes()
    if (
        weights[command.weight_offset : command.weight_offset + len(expected_weights)]
        != expected_weights
    ):
        raise SimulationError(
            f'Packed weights mismatch for operation "{operation.name}"'
        )
    bias_tensor = graph.tensors[operation.inputs[2]]
    expected_biases = np.asarray(bias_tensor.constant_data, dtype="<i4").tobytes(
        order="C"
    )
    biases = (bundle_path / "biases.bin").read_bytes()
    if (
        biases[command.bias_offset : command.bias_offset + len(expected_biases)]
        != expected_biases
    ):
        raise SimulationError(
            f'Packed biases mismatch for operation "{operation.name}"'
        )


def _stream(
    count: int,
    random_source: random.Random,
    config: SimulationConfig,
    *,
    stall_counter: str | None,
    counters: PerformanceCounters,
    layer_cycles: int,
) -> int:
    transferred = 0
    while transferred < count:
        if layer_cycles >= config.timeout_cycles:
            counters.error_count += 1
            raise SimulationTimeoutError(
                f"Layer timed out after {layer_cycles} cycles (seed={config.seed})"
            )
        ready = (
            not config.randomized_backpressure
            or random_source.random() < config.ready_probability
        )
        counters.total_cycles += 1
        layer_cycles += 1
        if ready:
            transferred += 1
        elif stall_counter is not None:
            setattr(counters, stall_counter, getattr(counters, stall_counter) + 1)
    return layer_cycles


def _program_conv(registers: RegisterModel, graph: Graph, operation: Operation) -> None:
    input_shape = graph.tensors[operation.inputs[0]].shape
    output_shape = graph.tensors[operation.outputs[0]].shape
    weight_shape = graph.tensors[operation.inputs[1]].shape
    values = (
        (0x020, input_shape[2]),
        (0x024, input_shape[3]),
        (0x028, input_shape[1]),
        (0x02C, output_shape[1]),
        (0x030, 0x1133),
        (0x034, output_shape[2]),
        (0x038, output_shape[3]),
        (0x03C, int("Relu" in operation.fused_operations)),
        (0x040, int(np.prod(input_shape))),
        (0x044, int(np.prod(weight_shape))),
        (0x048, int(np.prod(output_shape))),
    )
    for address, value in values:
        registers.write(address, value)
    registers.write(0x008, 1)


def simulate_bundle(
    path: str | Path,
    *,
    config: SimulationConfig | None = None,
    inputs: dict[str, np.ndarray] | None = None,
) -> SimulationResult:
    """Execute a validated bundle using only its public command/runtime contracts."""
    settings = config or SimulationConfig()
    bundle = load_bundle(path)
    graph = _graph_from_dict(bundle.graph)
    commands = _decode_commands((bundle.path / "commands.bin").read_bytes())
    if len(commands) != len(graph.operations):
        raise SimulationError("Command count does not match graph operation count")
    if [command.flags_and_index >> 16 for command in commands] != list(
        range(len(commands))
    ):
        raise SimulationError("Command layer indices are not sequential")
    runtime_inputs = (
        inputs if inputs is not None else _reference_inputs(bundle.path, graph)
    )
    values = execute_quantized(graph, runtime_inputs, capture_all=True)
    memory = SimulatedMemory(int(bundle.manifest["scratch_memory_bytes"]))
    allocations = {item["tensor"]: item for item in bundle.memory_plan["allocations"]}
    for name in graph.inputs:
        allocation = allocations[name]
        memory.write(allocation["offset"], _tensor_bytes(values[name]))

    registers = RegisterModel()
    if registers.read(0x000, registers.DEVICE_ID) != RegisterModel.DEVICE_ID:
        raise SimulationError("DEVICE_ID mismatch")
    if registers.read(0x004, registers.VERSION) != RegisterModel.VERSION:
        raise SimulationError("Interface VERSION mismatch")
    counters = PerformanceCounters()
    traces: list[LayerTrace] = []
    random_source = random.Random(settings.seed)
    for index, (command, operation) in enumerate(
        zip(commands, graph.operations, strict=True)
    ):
        _validate_command_payload(bundle.path, command, graph, operation)
        schedule = bundle.schedule["layers"][index]
        if schedule["operation"] != operation.name:
            raise SimulationError("Command and schedule operation mismatch")
        expected_backend = {1: "fpga", 2: "arm", 3: "metadata"}.get(command.backend)
        if expected_backend != operation.assigned_backend:
            raise SimulationError(f'Backend mismatch for operation "{operation.name}"')
        start_cycle = counters.total_cycles
        input_count = int(np.prod(graph.tensors[operation.inputs[0]].shape))
        output_count = int(np.prod(graph.tensors[operation.outputs[0]].shape))
        layer_cycles = 0
        if command.backend == 1:
            _program_conv(registers, graph, operation)
            weight_count = int(np.prod(graph.tensors[operation.inputs[1]].shape))
            layer_cycles = _stream(
                weight_count,
                random_source,
                settings,
                stall_counter=None,
                counters=counters,
                layer_cycles=layer_cycles,
            )
            counters.weight_load_cycles += weight_count
            layer_cycles = _stream(
                input_count,
                random_source,
                settings,
                stall_counter="input_stalls",
                counters=counters,
                layer_cycles=layer_cycles,
            )
            counters.input_count += input_count
            input_channels = graph.tensors[operation.inputs[0]].shape[1]
            compute_cycles = output_count * input_channels
            if layer_cycles + compute_cycles > settings.timeout_cycles:
                counters.error_count += 1
                raise SimulationTimeoutError(
                    f'Layer "{operation.name}" compute timed out (seed={settings.seed})'
                )
            counters.total_cycles += compute_cycles
            counters.compute_active_cycles += compute_cycles
            counters.executed_macs += compute_cycles * 9
            layer_cycles += compute_cycles
            layer_cycles = _stream(
                output_count,
                random_source,
                settings,
                stall_counter="output_stalls",
                counters=counters,
                layer_cycles=layer_cycles,
            )
            counters.output_count += output_count
            counters.layer_invocations += 1
            registers.read(0x00C, 2)
            registers.read(0x050, layer_cycles & 0xFFFFFFFF)
        else:
            fallback_cycles = max(1, int(schedule["estimated_compute_cycles"]))
            if fallback_cycles > settings.timeout_cycles:
                raise SimulationTimeoutError(
                    f'CPU fallback "{operation.name}" timed out (seed={settings.seed})'
                )
            counters.total_cycles += fallback_cycles
        output_value = values[operation.outputs[0]]
        allocation = allocations[operation.outputs[0]]
        memory.write(allocation["offset"], _tensor_bytes(output_value))
        traces.append(
            LayerTrace(
                index,
                operation.name,
                operation.assigned_backend,
                start_cycle,
                counters.total_cycles,
                input_count,
                output_count,
            )
        )

    outputs = {name: values[name].copy() for name in graph.outputs}
    actual = b"".join(_tensor_bytes(outputs[name]) for name in graph.outputs)
    expected = (bundle.path / "reference_output.bin").read_bytes()
    return SimulationResult(
        outputs,
        counters,
        traces,
        registers.transactions,
        settings.seed,
        actual == expected,
    )
