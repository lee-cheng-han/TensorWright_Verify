from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tensorwright.compiler import (
    BundleValidationError,
    CompilationResult,
    Graph,
    Operation,
    Tensor,
    build_bundle,
    load_bundle,
    validate_bundle,
)


def _compilation_result() -> CompilationResult:
    tensors = {
        "input": Tensor("input", [1, 1, 3, 3], "float", "int8", "NCHW", 1.0),
        "weights": Tensor(
            "weights",
            [1, 1, 3, 3],
            "float",
            "int8",
            "OIHW",
            [1.0],
            is_constant=True,
            constant_data=[[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]],
        ),
        "bias": Tensor(
            "bias",
            [1],
            "float",
            "int32",
            "UNSPECIFIED",
            [1.0],
            is_constant=True,
            constant_data=[3],
        ),
        "output": Tensor("output", [1, 1, 1, 1], "float", "int8", "NCHW", 1.0),
    }
    operation = Operation(
        "conv",
        "Conv",
        ["input", "weights", "bias"],
        ["output"],
        {
            "kernel_shape": [3, 3],
            "strides": [1, 1],
            "pads": [0, 0, 0, 0],
            "requantization_multipliers": [1],
            "requantization_shifts": [0],
        },
        True,
        "fpga",
        ["Relu"],
    )
    graph = Graph("bundle_test", {"": 13}, ["input"], ["output"], tensors, [operation])
    return CompilationResult(graph, {"format_version": 1, "comparison": {}})


class DeploymentBundleTest(unittest.TestCase):
    def test_bundle_is_complete_deterministic_and_loadable(self) -> None:
        result = _compilation_result()
        reference = {"input": np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)}
        with tempfile.TemporaryDirectory() as directory:
            first = build_bundle(
                result,
                Path(directory) / "first.twmodel",
                reference,
                labels=["edge"],
            )
            second = build_bundle(
                result, Path(directory) / "second.twmodel", reference, labels=["edge"]
            )
            self.assertEqual(
                {path.name: path.read_bytes() for path in first.iterdir()},
                {path.name: path.read_bytes() for path in second.iterdir()},
            )
            bundle = load_bundle(first)
            self.assertEqual(bundle.manifest["format_version"], 1)
            self.assertEqual(bundle.manifest["hardware_interface_version"], "1.0")
            self.assertEqual(bundle.manifest["layer_count"], 1)
            self.assertEqual(bundle.memory_plan["allocations"][0]["offset"], 0)
            command = struct.unpack("<8I", (first / "commands.bin").read_bytes())
            self.assertEqual(command[0:2], (1, 1))
            self.assertEqual(command[7] & 1, 1)
            self.assertEqual((first / "weights.bin").stat().st_size, 9)
            self.assertEqual((first / "biases.bin").stat().st_size, 4)
            self.assertEqual((first / "quantization.bin").stat().st_size, 8)
            self.assertEqual((first / "reference_input.bin").stat().st_size, 36)
            self.assertEqual((first / "reference_output.bin").stat().st_size, 1)
            self.assertEqual((first / "labels.txt").read_text(), "edge\n")

    def test_validator_rejects_corruption_and_version_mismatch(self) -> None:
        result = _compilation_result()
        reference = {"input": np.ones((1, 1, 3, 3), dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            path = build_bundle(result, Path(directory) / "model.twmodel", reference)
            (path / "weights.bin").write_bytes(b"corrupt")
            with self.assertRaisesRegex(BundleValidationError, "checksum"):
                validate_bundle(path)

            path = build_bundle(result, Path(directory) / "version.twmodel", reference)
            manifest = json.loads((path / "manifest.json").read_text())
            manifest["format_version"] = 99
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(BundleValidationError, "Unsupported"):
                validate_bundle(path)

    def test_output_path_contract_is_enforced(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(BundleValidationError, ".twmodel"),
        ):
            build_bundle(
                _compilation_result(),
                Path(directory) / "model.bundle",
                {"input": np.ones((1, 1, 3, 3), dtype=np.float32)},
            )


if __name__ == "__main__":
    unittest.main()
