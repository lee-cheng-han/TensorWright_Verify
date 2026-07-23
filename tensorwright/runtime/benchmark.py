"""Repeatable host-side performance benchmarking for deployment bundles."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from tensorwright.runtime.simulator import SimulationConfig, simulate_bundle


def benchmark_bundle(
    path: str | Path,
    *,
    runs: int = 10,
    seed: int = 0x7E45,
    randomized_backpressure: bool = True,
) -> dict[str, Any]:
    """Run a bundle repeatedly and summarize modeled accelerator performance."""
    if runs <= 0:
        raise ValueError("runs must be positive")
    results = [
        simulate_bundle(
            path,
            config=SimulationConfig(
                seed=seed + run,
                randomized_backpressure=randomized_backpressure,
            ),
        )
        for run in range(runs)
    ]
    cycles = [result.counters.total_cycles for result in results]
    outputs = [result.counters.output_count for result in results]
    return {
        "bundle": str(Path(path)),
        "runs": runs,
        "seed": seed,
        "randomized_backpressure": randomized_backpressure,
        "reference_match": all(result.reference_match for result in results),
        "total_cycles": {
            "minimum": min(cycles),
            "mean": mean(cycles),
            "maximum": max(cycles),
        },
        "inferences_per_cycle": {
            "minimum": 1 / max(cycles),
            "mean": mean(1 / cycle for cycle in cycles),
            "maximum": 1 / min(cycles),
        },
        "outputs_per_cycle": {
            "minimum": min(
                output_count / cycle
                for output_count, cycle in zip(outputs, cycles, strict=True)
            ),
            "mean": mean(
                output_count / cycle
                for output_count, cycle in zip(outputs, cycles, strict=True)
            ),
            "maximum": max(
                output_count / cycle
                for output_count, cycle in zip(outputs, cycles, strict=True)
            ),
        },
        "counters": {
            name: mean(getattr(result.counters, name) for result in results)
            for name in vars(results[0].counters)
        },
    }


def benchmark_bundle_json(*args: Any, **kwargs: Any) -> str:
    return (
        json.dumps(benchmark_bundle(*args, **kwargs), indent=2, sort_keys=True) + "\n"
    )
