"""Compile and simulate a recognizable seven-segment digit classifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tensorwright.compiler import compile_onnx_bundle, inspect_bundle
from tensorwright.runtime import SimulationConfig, simulate_bundle

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "model_demo"

# Segment order: top, upper-left, upper-right, middle, lower-left,
# lower-right, bottom.
DIGITS = np.asarray(
    [
        [1, 1, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 0, 1, 1],
        [0, 1, 1, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1],
    ],
    dtype=np.float32,
)


def main() -> int:
    BUILD.mkdir(parents=True, exist_ok=True)
    model_path = BUILD / "seven_segment_digits.onnx"
    calibration_path = BUILD / "calibration.npz"
    bundle_path = BUILD / "seven_segment_digits.twmodel"
    if bundle_path.exists():
        import shutil

        shutil.rmtree(bundle_path)
    _write_model(model_path)
    calibration = DIGITS[:, None, :]
    np.savez(calibration_path, segments=calibration)
    bundle = compile_onnx_bundle(
        model_path,
        calibration_path,
        bundle_path,
        labels=[str(value) for value in range(10)],
    )
    predictions: list[int] = []
    for digit, segments in enumerate(DIGITS):
        result = simulate_bundle(
            bundle,
            inputs={"segments": segments.reshape(1, 7)},
            config=SimulationConfig(randomized_backpressure=False),
        )
        probabilities = result.outputs["probabilities"]
        predictions.append(int(np.argmax(probabilities)))
        print(f"digit {digit} -> prediction {predictions[-1]}")
    accuracy = sum(
        prediction == expected for expected, prediction in enumerate(predictions)
    ) / len(predictions)
    report = {
        "model": "seven-segment digit classifier",
        "accuracy": accuracy,
        "predictions": predictions,
        "bundle": inspect_bundle(bundle),
    }
    report_path = BUILD / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"accuracy: {accuracy:.0%}")
    print(f"bundle: {bundle}")
    print(f"report: {report_path}")
    return 0 if accuracy == 1.0 else 2


def _write_model(path: Path) -> None:
    weights = (2.0 * DIGITS - 1.0).T
    biases = -np.sum(DIGITS, axis=1)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Gemm",
                ["segments", "weights", "biases"],
                ["logits"],
                name="digit_scores",
            ),
            helper.make_node(
                "Softmax", ["logits"], ["probabilities"], name="probabilities"
            ),
        ],
        "seven_segment_digits",
        [helper.make_tensor_value_info("segments", TensorProto.FLOAT, [1, 7])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [1, 10])],
        [
            numpy_helper.from_array(weights.astype(np.float32), "weights"),
            numpy_helper.from_array(biases.astype(np.float32), "biases"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="tensorwright",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


if __name__ == "__main__":
    raise SystemExit(main())
