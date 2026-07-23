"""User-facing ONNX-to-deployment compilation workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tensorwright.compiler.backend import build_bundle, load_bundle
from tensorwright.compiler.errors import CompilerError
from tensorwright.compiler.frontend import load_onnx
from tensorwright.compiler.passes import optimize_graph
from tensorwright.compiler.quantization import compile_quantized


def load_calibration_npz(
    path: str | Path, input_shapes: dict[str, list[int]]
) -> list[dict[str, np.ndarray]]:
    """Load named calibration tensors, optionally with a leading sample axis."""
    source = Path(path)
    if source.suffix != ".npz":
        raise CompilerError("Calibration data must use the .npz format")
    try:
        with np.load(source, allow_pickle=False) as archive:
            missing = sorted(set(input_shapes) - set(archive.files))
            extra = sorted(set(archive.files) - set(input_shapes))
            if missing or extra:
                raise CompilerError(
                    f"Calibration tensors differ from model inputs; "
                    f"missing={missing}, extra={extra}"
                )
            arrays = {name: archive[name].copy() for name in input_shapes}
    except OSError as error:
        raise CompilerError(f"Could not load calibration data: {error}") from error
    sample_counts: set[int] = set()
    batched: dict[str, bool] = {}
    for name, shape in input_shapes.items():
        value = arrays[name]
        expected = tuple(shape)
        if value.shape == expected:
            sample_counts.add(1)
            batched[name] = False
        elif value.ndim == len(expected) + 1 and value.shape[1:] == expected:
            sample_counts.add(value.shape[0])
            batched[name] = True
        else:
            raise CompilerError(
                f'Calibration tensor "{name}" has shape {list(value.shape)}; '
                f"expected {list(expected)} or [samples, *shape]"
            )
    if len(sample_counts) != 1:
        raise CompilerError("Calibration tensors have inconsistent sample counts")
    count = sample_counts.pop()
    return [
        {
            name: arrays[name][index].copy() if batched[name] else arrays[name].copy()
            for name in input_shapes
        }
        for index in range(count)
    ]


def compile_onnx_bundle(
    model_path: str | Path,
    calibration_path: str | Path,
    output_path: str | Path,
    *,
    labels: list[str] | None = None,
) -> Path:
    """Import, optimize, quantize, validate, and package an ONNX model."""
    imported = load_onnx(model_path)
    optimized = optimize_graph(imported)
    input_shapes = {name: optimized.tensors[name].shape for name in optimized.inputs}
    samples = load_calibration_npz(calibration_path, input_shapes)
    compiled = compile_quantized(optimized, samples)
    return build_bundle(compiled, output_path, samples[0], labels=labels)


def inspect_bundle(path: str | Path) -> dict[str, Any]:
    """Return a concise, machine-readable deployment summary."""
    bundle = load_bundle(path)
    backends: dict[str, int] = {}
    estimated_cycles = 0
    transfer_bytes = 0
    for layer in bundle.schedule["layers"]:
        backend = str(layer["backend"])
        backends[backend] = backends.get(backend, 0) + 1
        estimated_cycles += int(layer["estimated_compute_cycles"])
        transfer_bytes += int(layer["estimated_transfer_bytes"])
    return {
        "model": bundle.manifest["model"],
        "format_version": bundle.manifest["format_version"],
        "hardware_interface_version": bundle.manifest["hardware_interface_version"],
        "layers": bundle.manifest["layer_count"],
        "backends": backends,
        "scratch_memory_bytes": bundle.manifest["scratch_memory_bytes"],
        "estimated_compute_cycles": estimated_cycles,
        "estimated_transfer_bytes": transfer_bytes,
        "bundle_bytes": sum(
            int(metadata["size"]) for metadata in bundle.manifest["files"].values()
        ),
    }


def inspect_bundle_json(path: str | Path) -> str:
    return json.dumps(inspect_bundle(path), indent=2, sort_keys=True) + "\n"
