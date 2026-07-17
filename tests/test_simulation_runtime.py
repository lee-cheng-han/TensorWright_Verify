from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from tensorwright.cli import main
from tensorwright.compiler import build_bundle, compile_quantized
from tensorwright.runtime import (
    SimulationConfig,
    SimulationTimeoutError,
    simulate_bundle,
)
from tests.test_deployment_bundle import _compilation_result
from tests.test_quantized_compilation import _small_classifier_graph


class SimulationRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.bundle = build_bundle(
            _compilation_result(),
            Path(self.temporary_directory.name) / "runtime.twmodel",
            {"input": np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)},
        )

    def test_executes_commands_memory_registers_and_streams(self) -> None:
        result = simulate_bundle(
            self.bundle,
            config=SimulationConfig(randomized_backpressure=False),
        )

        self.assertTrue(result.reference_match)
        self.assertEqual(result.outputs["output"].tolist(), [[[[-5]]]])
        self.assertEqual(result.counters.total_cycles, 20)
        self.assertEqual(result.counters.compute_active_cycles, 1)
        self.assertEqual(result.counters.weight_load_cycles, 9)
        self.assertEqual(result.counters.input_count, 9)
        self.assertEqual(result.counters.output_count, 1)
        self.assertEqual(result.counters.executed_macs, 9)
        self.assertEqual(result.counters.layer_invocations, 1)
        self.assertEqual(result.layers[0].backend, "fpga")
        self.assertIn(
            {"kind": "write", "address": 0x008, "value": 1},
            result.register_transactions,
        )

    def test_seeded_backpressure_is_reproducible(self) -> None:
        config = SimulationConfig(seed=91, ready_probability=0.4)
        first = simulate_bundle(self.bundle, config=config)
        second = simulate_bundle(self.bundle, config=config)

        self.assertEqual(first.to_json(), second.to_json())
        self.assertGreater(first.counters.total_cycles, 20)
        self.assertGreater(
            first.counters.input_stalls + first.counters.output_stalls, 0
        )

    def test_timeout_reports_seed(self) -> None:
        with self.assertRaisesRegex(SimulationTimeoutError, "seed=123"):
            simulate_bundle(
                self.bundle,
                config=SimulationConfig(
                    seed=123,
                    timeout_cycles=5,
                    randomized_backpressure=False,
                ),
            )

    def test_executes_declared_cpu_fallback_layers(self) -> None:
        sample = {"input": np.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4)}
        compiled = compile_quantized(_small_classifier_graph(), [sample])
        bundle = build_bundle(
            compiled,
            Path(self.temporary_directory.name) / "mixed.twmodel",
            sample,
        )

        result = simulate_bundle(
            bundle, config=SimulationConfig(randomized_backpressure=False)
        )

        self.assertTrue(result.reference_match)
        self.assertIn("fpga", [layer.backend for layer in result.layers])
        self.assertIn("arm", [layer.backend for layer in result.layers])
        self.assertIn("metadata", [layer.backend for layer in result.layers])
        self.assertEqual(result.outputs["output"].shape, (1, 2))

    def test_cli_simulate_outputs_machine_readable_report(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            status = main(["simulate", str(self.bundle), "--no-backpressure"])

        report = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertTrue(report["reference_match"])
        self.assertEqual(report["counters"]["executed_macs"], 9)

    def test_cli_returns_failure_for_invalid_bundle(self) -> None:
        error = StringIO()
        with redirect_stderr(error):
            status = main(["simulate", str(self.bundle / "missing.twmodel")])
        self.assertEqual(status, 1)
        self.assertIn("simulation failed", error.getvalue())


if __name__ == "__main__":
    unittest.main()
