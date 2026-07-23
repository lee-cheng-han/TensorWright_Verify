from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from scripts.run_bundle_rtl_demo import (
    _calibration_samples as convolution_calibration,
)
from scripts.run_bundle_rtl_demo import _write_model as write_convolution_model
from scripts.run_model_demo import DIGITS, _write_model
from tensorwright.cli import main
from tensorwright.compiler import (
    CompilerError,
    compile_onnx_bundle,
    inspect_bundle,
    load_calibration_npz,
)
from tensorwright.runtime import (
    benchmark_bundle,
    extract_fixed_convolution,
    simulate_bundle,
    write_convolution_vector,
)


class CompilerWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "digits.onnx"
        self.calibration = self.root / "calibration.npz"
        self.bundle = self.root / "digits.twmodel"
        _write_model(self.model)
        np.savez(self.calibration, segments=DIGITS[:, None, :])

    def test_compile_inspect_simulate_and_benchmark(self) -> None:
        compile_onnx_bundle(self.model, self.calibration, self.bundle)

        summary = inspect_bundle(self.bundle)
        result = simulate_bundle(
            self.bundle, inputs={"segments": DIGITS[3].reshape(1, 7)}
        )
        benchmark = benchmark_bundle(self.bundle, runs=2, randomized_backpressure=False)

        self.assertEqual(summary["model"], "seven_segment_digits")
        self.assertEqual(int(np.argmax(result.outputs["probabilities"])), 3)
        self.assertTrue(benchmark["reference_match"])
        self.assertEqual(benchmark["runs"], 2)
        self.assertGreater(benchmark["inferences_per_cycle"]["mean"], 0)

    def test_cli_compile_inspect_and_benchmark(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            status = main(
                [
                    "compile",
                    str(self.model),
                    str(self.calibration),
                    str(self.bundle),
                ]
            )
        self.assertEqual(status, 0)
        self.assertTrue(self.bundle.is_dir())

        output = StringIO()
        with redirect_stdout(output):
            status = main(["inspect-bundle", str(self.bundle)])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["layers"], 2)

        output = StringIO()
        with redirect_stdout(output):
            status = main(
                ["benchmark", str(self.bundle), "--runs", "2", "--no-backpressure"]
            )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["runs"], 2)

    def test_calibration_validation_rejects_wrong_shape(self) -> None:
        invalid = self.root / "invalid.npz"
        np.savez(invalid, segments=np.zeros((2, 8), dtype=np.float32))
        with self.assertRaisesRegex(CompilerError, "shape"):
            load_calibration_npz(invalid, {"segments": [1, 7]})

    def test_compiled_convolution_decodes_into_rtl_invocation(self) -> None:
        model = self.root / "convolution.onnx"
        calibration = self.root / "convolution.npz"
        bundle = self.root / "convolution.twmodel"
        write_convolution_model(model)
        np.savez(calibration, input=convolution_calibration())
        compile_onnx_bundle(model, calibration, bundle)

        invocation = extract_fixed_convolution(bundle)
        vector = write_convolution_vector(invocation, self.root / "vectors.txt")

        self.assertEqual(len(invocation.weights), 54)
        self.assertEqual(len(invocation.activations), 75)
        self.assertEqual(len(invocation.expected), 18)
        self.assertTrue(vector.read_text(encoding="utf-8").startswith("1\n"))


if __name__ == "__main__":
    unittest.main()
