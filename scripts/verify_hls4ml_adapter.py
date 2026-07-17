"""Run a genuine hls4ml C-simulation trace integration check."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import hls4ml  # type: ignore[import-not-found]

    from tensorwright.trace.plugins import AdapterRequest, default_adapter_registry
    from tensorwright.trace.schema import read_trace

    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    layers = [
        {"class_name": "Input", "name": "model_input", "input_shape": [2]},
        {
            "class_name": "Dense",
            "name": "dense",
            "n_in": 2,
            "n_out": 2,
            "weight_data": np.array([[1, 2], [3, 4]], dtype=float),
            "bias_data": np.array([1, -1], dtype=float),
        },
        {
            "class_name": "Activation",
            "name": "relu",
            "activation": "relu",
            "inputs": ["dense"],
        },
    ]
    config = {
        "OutputDir": str(root / "hls-project"),
        "ProjectName": "tw_hls4ml",
        "IOType": "io_parallel",
        "Backend": "Vivado",
        "HLSConfig": {
            "Model": {"Precision": "ap_fixed<16,6>", "ReuseFactor": 1},
            "LayerName": {
                "dense": {"Trace": True},
                "relu": {"Trace": True},
            },
        },
    }
    model = hls4ml.model.ModelGraph.from_layer_list(config, layers)
    prediction, trace = model.trace(np.array([[2, 3]], dtype=np.float32))
    source = root / "hls4ml_trace.npz"
    np.savez(source, **trace)

    options = {
        "run_id": "hls4ml-integration",
        "model_id": "hls4ml_adapter_integration",
        "tensors": [
            {
                "tensor_name": "dense",
                "source_operation_id": "source:Dense_0",
                "compiled_operation_id": "hls4ml:dense",
                "operation_name": "dense",
                "operation_type": "Dense",
                "layout": "NC",
            },
            {
                "tensor_name": "relu",
                "source_operation_id": "source:Relu_1",
                "compiled_operation_id": "hls4ml:relu",
                "operation_name": "relu",
                "operation_type": "Activation",
                "layout": "NC",
            },
        ],
    }
    output = default_adapter_registry().convert(
        "hls4ml.csim",
        AdapterRequest(source, root / "trace.jsonl", options),
    )
    values = [event.value for event in read_trace(output).events]
    expected = [12.0, 15.0, 12.0, 15.0]
    if values != expected or np.asarray(prediction).tolist() != [12.0, 15.0]:
        raise RuntimeError(
            f"hls4ml adapter result differs: prediction={prediction!r}, "
            f"values={values!r}"
        )
    print(f"hls4ml trace layers: {', '.join(sorted(trace))}")
    print(f"Canonical events: {len(values)}")
    print("hls4ml adapter integration: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
