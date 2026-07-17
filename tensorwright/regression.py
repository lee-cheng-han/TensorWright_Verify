"""Portable Cocotb regression-package generation."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tensorwright.minimizer import FailureSignature, MinimizationError
from tensorwright.trace.schema import TraceEvent, read_trace

REGRESSION_FORMAT_VERSION = 1
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class RegressionGenerationError(RuntimeError):
    """Raised when a portable regression package cannot be generated."""


@dataclass(frozen=True)
class RegressionPackage:
    """Generated package paths and manifest identity."""

    path: Path
    test_path: Path
    manifest_path: Path
    name: str


def generate_cocotb_regression(
    inputs_path: str | Path,
    minimization_report_path: str | Path,
    reference_trace_path: str | Path,
    output_directory: str | Path,
    *,
    name: str,
) -> RegressionPackage:
    """Generate a self-contained Cocotb regression around a canonical trace."""
    if NAME_PATTERN.fullmatch(name) is None:
        raise RegressionGenerationError("Regression name must match [a-z][a-z0-9_]*")
    inputs_source = Path(inputs_path)
    report_source = Path(minimization_report_path)
    reference_source = Path(reference_trace_path)
    destination = Path(output_directory)
    if destination.exists() and any(destination.iterdir()):
        raise RegressionGenerationError("Regression output directory is not empty")
    try:
        with np.load(inputs_source, allow_pickle=False) as archive:
            inputs = {key: archive[key].copy() for key in archive.files}
        if not inputs:
            raise RegressionGenerationError("Regression input archive is empty")
        minimization_data = json.loads(report_source.read_text(encoding="utf-8"))
        signature_data = minimization_data.get("failure_signature")
        if not isinstance(signature_data, dict):
            raise RegressionGenerationError(
                "Minimization report has no failure_signature object"
            )
        signature = FailureSignature.from_dict(signature_data)
        trace = read_trace(reference_source)
    except (OSError, ValueError, json.JSONDecodeError, MinimizationError) as error:
        raise RegressionGenerationError(
            f"Invalid regression source artifact: {error}"
        ) from error
    if not _signature_has_reference_value(signature, trace.events):
        raise RegressionGenerationError(
            "Failure signature does not identify a value in the reference trace"
        )

    destination.mkdir(parents=True, exist_ok=True)
    tensors_directory = destination / "tensors"
    shutil.copy2(inputs_source, destination / "inputs.npz")
    shutil.copy2(reference_source, destination / "reference.jsonl")
    copied_payloads: set[str] = set()
    for event in trace.events:
        if event.data_file is None or event.data_file in copied_payloads:
            continue
        copied_payloads.add(event.data_file)
        source_payload = reference_source.parent / event.data_file
        target_payload = destination / event.data_file
        target_payload.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_payload, target_payload)
    if not copied_payloads and tensors_directory.exists():
        tensors_directory.rmdir()

    test_name = f"test_{name}.py"
    (destination / test_name).write_text(_test_source(name), encoding="utf-8")
    (destination / "README.md").write_text(_readme(name, test_name), encoding="utf-8")
    manifest: dict[str, Any] = {
        "format_version": REGRESSION_FORMAT_VERSION,
        "name": name,
        "test_module": test_name.removesuffix(".py"),
        "model_id": trace.events[0].model_id,
        "failure_signature": asdict(signature),
        "inputs": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in sorted(inputs.items())
        },
        "files": {},
    }
    files = [
        path
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest["files"] = {
        str(path.relative_to(destination)): {
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in files
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return RegressionPackage(destination, destination / test_name, manifest_path, name)


def _signature_has_reference_value(
    signature: FailureSignature, events: list[TraceEvent]
) -> bool:
    for event in events:
        trace_point = (
            "operation_output"
            if event.trace_point == "stream_transfer"
            else event.trace_point
        )
        if (
            event.source_operation_id != signature.source_operation_id
            or event.tensor_name != signature.tensor_name
            or trace_point != signature.trace_point
        ):
            continue
        if event.coordinate is not None:
            if tuple(event.coordinate) == signature.coordinate:
                return True
            continue
        assert event.start_coordinate is not None and event.chunk_shape is not None
        if all(
            start <= coordinate < start + size
            for coordinate, start, size in zip(
                signature.coordinate,
                event.start_coordinate,
                event.chunk_shape,
                strict=True,
            )
        ):
            return True
    return False


def _test_source(name: str) -> str:
    return f'''"""Generated TensorWright regression: {name}."""

from __future__ import annotations

import importlib
import inspect
import json
import os
from pathlib import Path

import cocotb
import numpy as np

from tensorwright.trace import compare_trace_files

ROOT = Path(__file__).resolve().parent


@cocotb.test()  # type: ignore[untyped-decorator]
async def regression_{name}(dut) -> None:  # type: ignore[no-untyped-def]
    """Drive the minimized case and require the original mismatch to be fixed."""
    adapter_name = os.environ.get("TENSORWRIGHT_REGRESSION_ADAPTER")
    assert adapter_name, (
        "TENSORWRIGHT_REGRESSION_ADAPTER=module:function is required"
    )
    module_name, separator, function_name = adapter_name.partition(":")
    assert separator and module_name and function_name, (
        "adapter must be module:function"
    )
    adapter = getattr(importlib.import_module(module_name), function_name)
    with np.load(ROOT / "inputs.npz", allow_pickle=False) as archive:
        inputs = {{key: archive[key].copy() for key in archive.files}}
    candidate_path = Path(
        os.environ.get(
            "TENSORWRIGHT_REGRESSION_OUTPUT", str(Path.cwd() / "{name}_candidate.jsonl")
        )
    )
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    result = adapter(dut, inputs, candidate_path)
    if inspect.isawaitable(result):
        result = await result
    if result is not None:
        candidate_path = Path(result)
    comparison = compare_trace_files(ROOT / "reference.jsonl", candidate_path)
    if not comparison.matched:
        expected = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))[
            "failure_signature"
        ]
        divergence = comparison.first_divergence
        assert divergence is not None
        observed = {{
            "kind": divergence.kind,
            "source_operation_id": divergence.source_operation_id,
            "tensor_name": divergence.tensor_name,
            "trace_point": divergence.trace_point,
            "coordinate": divergence.coordinate,
        }}
        expected_identity = {{key: expected[key] for key in observed}}
        assert observed == expected_identity, (
            "Regression failure signature changed:\\n"
            + json.dumps(
                {{"expected": expected_identity, "observed": observed}}, indent=2
            )
        )
    assert comparison.matched, (
        "Original TensorWright divergence still reproduces:\\n" + comparison.to_json()
    )
'''


def _readme(name: str, test_name: str) -> str:
    return f"""# TensorWright regression: `{name}`

This package contains minimized named inputs, a canonical reference trace, the
preserved failure signature, and `{test_name}`. Set
`TENSORWRIGHT_REGRESSION_ADAPTER` to an importable `module:function`. The function
receives `(dut, inputs, candidate_path)`, may be synchronous or asynchronous, and
must write a canonical candidate trace or return its path.

Add this directory to the Cocotb test path and use `{Path(test_name).stem}` as the
test module. The test fails with the preserved signature while the bug remains,
reports signature drift as a distinct failure, and passes once the candidate trace
matches the bundled reference.
"""
