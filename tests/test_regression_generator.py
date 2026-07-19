from __future__ import annotations

import hashlib
import json
import py_compile
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from tensorwright.cli import main
from tensorwright.minimizer import FailureSignature
from tensorwright.regression import (
    RegressionGenerationError,
    generate_cocotb_regression,
    generate_rtl_arithmetic_regression,
)
from tensorwright.trace import compare_trace_files, write_trace
from tests.test_trace_comparison import _event


class RegressionGeneratorTest(unittest.TestCase):
    def _sources(self, root: Path) -> tuple[Path, Path, Path]:
        inputs = root / "minimal.npz"
        report = root / "minimal.json"
        reference = root / "reference.jsonl"
        np.savez(inputs, image=np.array([0, 0, 1, 0], dtype=np.int8))
        signature = FailureSignature(
            "value_mismatch",
            "onnx:Conv_0",
            "output",
            "operation_output",
            (0, 0, 0, 0),
            "requantization_rounding_mismatch",
        )
        report.write_text(
            json.dumps(
                {
                    "failure_signature": {
                        **signature.__dict__,
                        "coordinate": list(signature.coordinate),
                    }
                }
            ),
            encoding="utf-8",
        )
        write_trace(reference, [_event(10, [0, 0, 0, 0])])
        return inputs, report, reference

    def test_generates_portable_deterministic_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = self._sources(root)
            first = generate_cocotb_regression(
                *sources, root / "first", name="conv_rounding_regression"
            )
            second = generate_cocotb_regression(
                *sources, root / "second", name="conv_rounding_regression"
            )
            first_files = {
                str(path.relative_to(first.path)): path.read_bytes()
                for path in first.path.rglob("*")
                if path.is_file()
            }
            second_files = {
                str(path.relative_to(second.path)): path.read_bytes()
                for path in second.path.rglob("*")
                if path.is_file()
            }
            manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
            py_compile.compile(str(first.test_path), doraise=True)
        self.assertEqual(first_files, second_files)
        self.assertEqual(manifest["format_version"], 1)
        self.assertEqual(manifest["inputs"]["image"]["dtype"], "int8")
        self.assertEqual(
            manifest["failure_signature"]["rule_id"],
            "requantization_rounding_mismatch",
        )
        for relative, metadata in manifest["files"].items():
            self.assertEqual(
                hashlib.sha256(first_files[relative]).hexdigest(), metadata["sha256"]
            )
        source = first_files["test_conv_rounding_regression.py"].decode()
        self.assertIn("TENSORWRIGHT_REGRESSION_ADAPTER", source)
        self.assertIn("compare_trace_files", source)
        self.assertIn("failure signature changed", source)

    def test_copies_chunk_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, report, reference = self._sources(root)
            tensors = root / "tensors"
            tensors.mkdir()
            np.save(tensors / "output.npy", np.array([[[[10]]]], dtype=np.int8))
            event = _event(10, [0, 0, 0, 0])
            data = event.to_dict()
            data.update(
                event_type="tensor_chunk",
                value=None,
                coordinate=None,
                start_coordinate=[0, 0, 0, 0],
                chunk_shape=[1, 1, 1, 1],
                data_file="tensors/output.npy",
            )
            from tensorwright.trace import TraceEvent

            write_trace(reference, [TraceEvent.from_dict(data)])
            package = generate_cocotb_regression(
                inputs, report, reference, root / "generated", name="chunk_case"
            )
            copied = package.path / "tensors" / "output.npy"
            self.assertTrue(copied.is_file())

    def test_generates_rtl_arithmetic_case_from_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = write_trace(
                root / "reference.jsonl", [_event(-117, [0, 0, 0, 0])]
            )
            candidate = write_trace(
                root / "candidate.jsonl",
                [_event(-116, [0, 0, 0, 0], backend="custom.rtl")],
            )
            comparison = compare_trace_files(reference, candidate)
            package = generate_rtl_arithmetic_regression(
                comparison,
                {
                    "accumulator": 24,
                    "bias": -491,
                    "multiplier": 1,
                    "shift": 2,
                    "software_result": -117,
                },
                root / "generated",
            )
            py_compile.compile(str(package.test_path), doraise=True)
            manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
            source = package.test_path.read_text(encoding="utf-8")
        self.assertEqual(manifest["generated_from"]["reference_value"], -117)
        self.assertEqual(manifest["arithmetic"]["accumulator"], 24)
        self.assertEqual(manifest["arithmetic"]["software_result"], -117)
        self.assertIn("TENSORWRIGHT_REGRESSION_FAULTY", source)

    def test_rejects_bad_name_report_and_nonempty_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = self._sources(root)
            with self.assertRaisesRegex(RegressionGenerationError, "name"):
                generate_cocotb_regression(*sources, root / "bad", name="Bad-Name")
            nonempty = root / "nonempty"
            nonempty.mkdir()
            (nonempty / "owned.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(RegressionGenerationError, "not empty"):
                generate_cocotb_regression(*sources, nonempty, name="valid_name")
            sources[1].write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RegressionGenerationError, "failure_signature"):
                generate_cocotb_regression(
                    *sources, root / "missing", name="valid_name"
                )

    def test_cli_generates_regression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = self._sources(root)
            output = root / "regression"
            stdout = StringIO()
            with redirect_stdout(stdout):
                status = main(
                    [
                        "generate-regression",
                        *[str(path) for path in sources],
                        str(output),
                        "--name",
                        "cli_case",
                    ]
                )
        self.assertEqual(status, 0)
        self.assertIn("Generated Cocotb regression", stdout.getvalue())

    def test_cli_rejects_invalid_sources(self) -> None:
        error = StringIO()
        with redirect_stderr(error):
            status = main(
                [
                    "generate-regression",
                    "missing.npz",
                    "missing.json",
                    "missing.jsonl",
                    "output",
                    "--name",
                    "missing_case",
                ]
            )
        self.assertEqual(status, 1)
        self.assertIn("regression generation failed", error.getvalue())


if __name__ == "__main__":
    unittest.main()
