"""Deterministic failure-preserving tensor-input minimization."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


class MinimizationError(RuntimeError):
    """Raised when minimization cannot establish or preserve a failure."""


@dataclass(frozen=True)
class FailureSignature:
    """Stable identity of the failure an oracle must preserve."""

    kind: str
    source_operation_id: str
    tensor_name: str
    trace_point: str
    coordinate: tuple[int, ...]
    rule_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureSignature:
        required = {
            "kind",
            "source_operation_id",
            "tensor_name",
            "trace_point",
            "coordinate",
        }
        missing = sorted(required - data.keys())
        if missing:
            raise MinimizationError(
                f"Failure signature is missing fields: {', '.join(missing)}"
            )
        coordinate = data["coordinate"]
        if not isinstance(coordinate, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in coordinate
        ):
            raise MinimizationError(
                "Failure signature coordinate must be integer JSON array"
            )
        return cls(
            kind=str(data["kind"]),
            source_operation_id=str(data["source_operation_id"]),
            tensor_name=str(data["tensor_name"]),
            trace_point=str(data["trace_point"]),
            coordinate=tuple(coordinate),
            rule_id=str(data["rule_id"]) if data.get("rule_id") is not None else None,
        )


FailureOracle = Callable[[dict[str, np.ndarray]], FailureSignature | None]


@dataclass(frozen=True)
class MinimizationResult:
    """Reduced inputs and auditable minimizer statistics."""

    inputs: dict[str, np.ndarray]
    failure_signature: FailureSignature
    evaluations: int
    original_nonzero_values: int
    minimized_nonzero_values: int
    changed_values: int
    stopped_by_budget: bool

    def report_dict(self) -> dict[str, Any]:
        return {
            "failure_signature": asdict(self.failure_signature),
            "evaluations": self.evaluations,
            "original_nonzero_values": self.original_nonzero_values,
            "minimized_nonzero_values": self.minimized_nonzero_values,
            "changed_values": self.changed_values,
            "reduction_fraction": (
                0.0
                if self.original_nonzero_values == 0
                else 1.0 - self.minimized_nonzero_values / self.original_nonzero_values
            ),
            "stopped_by_budget": self.stopped_by_budget,
            "inputs": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in sorted(self.inputs.items())
            },
        }

    def report_json(self) -> str:
        return json.dumps(self.report_dict(), indent=2, sort_keys=True) + "\n"

    def write(self, output_path: str | Path, report_path: str | Path) -> None:
        output = Path(output_path)
        report = Path(report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, **self.inputs)  # type: ignore[arg-type]
        report.write_text(self.report_json(), encoding="utf-8")


def minimize_inputs(
    inputs: dict[str, np.ndarray],
    oracle: FailureOracle,
    *,
    expected_signature: FailureSignature | None = None,
    max_evaluations: int = 1000,
) -> MinimizationResult:
    """Minimize nonzero support and magnitude while preserving one exact failure."""
    if not inputs:
        raise MinimizationError("Minimization requires at least one input tensor")
    if max_evaluations <= 0:
        raise MinimizationError("max_evaluations must be positive")
    current = {name: np.asarray(value).copy() for name, value in sorted(inputs.items())}
    if any(value.dtype.kind not in "biuf" for value in current.values()):
        raise MinimizationError(
            "Minimization supports only numeric and boolean tensors"
        )
    if any(
        value.dtype.kind == "f" and not bool(np.all(np.isfinite(value)))
        for value in current.values()
    ):
        raise MinimizationError("Minimization inputs must contain only finite values")
    original = {name: value.copy() for name, value in current.items()}
    evaluations = 0

    def preserves(candidate: dict[str, np.ndarray]) -> bool:
        nonlocal evaluations
        if evaluations >= max_evaluations:
            return False
        evaluations += 1
        try:
            return (
                oracle({name: value.copy() for name, value in candidate.items()})
                == signature
            )
        except Exception as error:
            raise MinimizationError(
                f"Failure oracle raised an error: {error}"
            ) from error

    evaluations += 1
    try:
        initial_signature = oracle(
            {name: value.copy() for name, value in current.items()}
        )
    except Exception as error:
        raise MinimizationError(f"Failure oracle raised an error: {error}") from error
    if initial_signature is None:
        raise MinimizationError("Original input does not reproduce a failure")
    signature = expected_signature or initial_signature
    if initial_signature != signature:
        raise MinimizationError(
            "Original input does not reproduce the expected failure"
        )

    coordinates = _nonzero_coordinates(current)
    granularity = min(2, len(coordinates))
    while coordinates and evaluations < max_evaluations:
        removed = False
        for chunk in _partitions(coordinates, granularity):
            candidate = {name: value.copy() for name, value in current.items()}
            for name, flat_index in chunk:
                candidate[name].flat[flat_index] = 0
            if preserves(candidate):
                current = candidate
                removed_set = set(chunk)
                coordinates = [item for item in coordinates if item not in removed_set]
                granularity = min(max(2, granularity - 1), len(coordinates))
                removed = True
                break
        if removed:
            continue
        if granularity >= len(coordinates):
            break
        granularity = min(len(coordinates), granularity * 2)

    for name, flat_index in coordinates:
        if evaluations >= max_evaluations:
            break
        value = current[name].flat[flat_index].item()
        simplified = _simplified_value(value, current[name].dtype)
        if simplified == value:
            continue
        candidate = {key: tensor.copy() for key, tensor in current.items()}
        candidate[name].flat[flat_index] = simplified
        if preserves(candidate):
            current = candidate

    return MinimizationResult(
        current,
        signature,
        evaluations,
        len(_nonzero_coordinates(original)),
        len(_nonzero_coordinates(current)),
        sum(int(np.count_nonzero(original[name] != current[name])) for name in current),
        evaluations >= max_evaluations,
    )


def _nonzero_coordinates(inputs: dict[str, np.ndarray]) -> list[tuple[str, int]]:
    return [
        (name, int(index))
        for name, value in inputs.items()
        for index in np.flatnonzero(value)
    ]


def _partitions(
    values: list[tuple[str, int]], count: int
) -> list[list[tuple[str, int]]]:
    if not values:
        return []
    count = max(1, min(count, len(values)))
    quotient, remainder = divmod(len(values), count)
    result: list[list[tuple[str, int]]] = []
    start = 0
    for partition in range(count):
        size = quotient + (1 if partition < remainder else 0)
        result.append(values[start : start + size])
        start += size
    return result


def _simplified_value(value: int | float | bool, dtype: np.dtype[Any]) -> int | float:
    if value == 0 or dtype.kind == "b":
        return int(value)
    sign = -1 if value < 0 else 1
    if dtype.kind in "iu":
        return sign
    candidate = float(sign)
    return candidate if abs(float(value)) > 1.0 else float(value)
