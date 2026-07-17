from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from tensorwright.cli import main
from tensorwright.minimizer import (
    FailureSignature,
    MinimizationError,
    minimize_inputs,
)

SIGNATURE = FailureSignature(
    "value_mismatch",
    "onnx:Conv_0",
    "output",
    "operation_output",
    (0, 0, 0, 1),
    "requantization_rounding_mismatch",
)


class MinimizerTest(unittest.TestCase):
    def test_minimizes_support_and_value_magnitude_deterministically(self) -> None:
        inputs = {
            "image": np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int8),
            "aux": np.array([9, 10], dtype=np.int16),
        }

        def oracle(candidate: dict[str, np.ndarray]) -> FailureSignature | None:
            if candidate["image"][2] != 0 and candidate["aux"][1] != 0:
                return SIGNATURE
            return None

        first = minimize_inputs(inputs, oracle)
        second = minimize_inputs(inputs, oracle)
        self.assertEqual(first.failure_signature, SIGNATURE)
        self.assertEqual(first.original_nonzero_values, 10)
        self.assertEqual(first.minimized_nonzero_values, 2)
        self.assertEqual(first.inputs["image"].tolist(), [0, 0, 1, 0, 0, 0, 0, 0])
        self.assertEqual(first.inputs["aux"].tolist(), [0, 1])
        self.assertEqual(first.evaluations, second.evaluations)
        for name in first.inputs:
            np.testing.assert_array_equal(first.inputs[name], second.inputs[name])

    def test_requires_original_and_expected_failure(self) -> None:
        inputs = {"input": np.array([1], dtype=np.int8)}
        with self.assertRaisesRegex(MinimizationError, "does not reproduce"):
            minimize_inputs(inputs, lambda _: None)
        different = FailureSignature(
            "value_mismatch", "other", "output", "operation_output", (0,)
        )
        with self.assertRaisesRegex(MinimizationError, "expected failure"):
            minimize_inputs(inputs, lambda _: SIGNATURE, expected_signature=different)

    def test_budget_is_reported_and_inputs_are_not_mutated(self) -> None:
        original = np.array([5, 4, 3, 2], dtype=np.int8)
        result = minimize_inputs(
            {"input": original}, lambda _: SIGNATURE, max_evaluations=1
        )
        self.assertTrue(result.stopped_by_budget)
        self.assertEqual(result.evaluations, 1)
        self.assertEqual(original.tolist(), [5, 4, 3, 2])

    def test_result_writes_npz_and_json_report(self) -> None:
        result = minimize_inputs(
            {"input": np.array([3, 2], dtype=np.int8)}, lambda _: SIGNATURE
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "minimal.npz"
            report = Path(directory) / "report.json"
            result.write(output, report)
            with np.load(output, allow_pickle=False) as archive:
                values = archive["input"].tolist()
            data = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(values, [0, 0])
        self.assertEqual(data["failure_signature"]["kind"], "value_mismatch")

    def test_cli_uses_external_json_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "failing.npz"
            output = root / "minimal.npz"
            report = root / "minimal.json"
            np.savez(source, input=np.array([9, 8, 7, 6], dtype=np.int8))
            oracle = root / "oracle.py"
            oracle.write_text(
                """import json
import sys
import numpy as np
with np.load(sys.argv[-1], allow_pickle=False) as data:
    failed = int(data['input'][2]) != 0
print(json.dumps({
    'kind': 'value_mismatch',
    'source_operation_id': 'onnx:Conv_0',
    'tensor_name': 'output',
    'trace_point': 'operation_output',
    'coordinate': [0, 0, 0, 1],
    'rule_id': 'requantization_rounding_mismatch',
} if failed else None))
""",
                encoding="utf-8",
            )
            stdout = StringIO()
            with redirect_stdout(stdout):
                status = main(
                    [
                        "minimize",
                        str(source),
                        str(output),
                        "--report",
                        str(report),
                        "--oracle",
                        sys.executable,
                        str(oracle),
                    ]
                )
            with np.load(output, allow_pickle=False) as archive:
                minimized = archive["input"].tolist()
        self.assertEqual(status, 0)
        self.assertEqual(minimized, [0, 0, 1, 0])
        self.assertIn('"minimized_nonzero_values": 1', stdout.getvalue())

    def test_cli_reports_oracle_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.npz"
            np.savez(source, input=np.array([1], dtype=np.int8))
            error = StringIO()
            with redirect_stderr(error):
                status = main(
                    [
                        "minimize",
                        str(source),
                        str(Path(directory) / "output.npz"),
                        "--oracle",
                        sys.executable,
                        "-c",
                        "raise SystemExit(3)",
                    ]
                )
        self.assertEqual(status, 1)
        self.assertIn("minimization failed", error.getvalue())


if __name__ == "__main__":
    unittest.main()
