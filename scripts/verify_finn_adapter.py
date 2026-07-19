"""Run a genuine FINN execution-context integration check."""

from __future__ import annotations

import argparse
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # The FINN-pinned QONNX revision uses ONNX's former public mapping alias.
    if not hasattr(onnx, "mapping"):
        onnx.mapping = types.SimpleNamespace(
            TENSOR_TYPE_TO_NP_TYPE={
                key: value.np_dtype
                for key, value in onnx._mapping.TENSOR_TYPE_MAP.items()
            }
        )

    from finn.core.onnx_exec import execute_onnx  # type: ignore[import-not-found]
    from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

    from tensorwright.trace.plugins import AdapterRequest, default_adapter_registry
    from tensorwright.trace.schema import read_trace

    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    context = execute_onnx(
        _make_model(ModelWrapper),
        {"input": np.array([[1, 2, -5, 8]], dtype=np.float32)},
        return_full_exec_context=True,
    )
    source = root / "verify_tensorwright_adapter_SUCCESS.npz"
    np.savez(source, **context)
    options = {
        "run_id": "finn-integration",
        "model_id": "finn_adapter_integration",
        "tensors": [
            {
                "tensor_name": "add_out",
                "source_operation_id": "onnx:Add_0",
                "compiled_operation_id": "finn:Add_0",
                "operation_name": "Add_0",
                "operation_type": "Add",
                "layout": "NC",
            },
            {
                "tensor_name": "output",
                "source_operation_id": "onnx:Relu_1",
                "compiled_operation_id": "finn:Relu_1",
                "operation_name": "Relu_1",
                "operation_type": "Relu",
                "layout": "NC",
            },
        ],
    }
    output = default_adapter_registry().convert(
        "finn.dataflow",
        AdapterRequest(source, root / "trace.jsonl", options),
    )
    values = [event.value for event in read_trace(output).events]
    expected = [-1.0, 3.0, -2.0, 4.0, 0.0, 3.0, 0.0, 4.0]
    if values != expected:
        raise RuntimeError(f"FINN adapter values differ: {values!r}")
    print(f"FINN context tensors: {', '.join(sorted(context))}")
    print(f"Canonical events: {len(values)}")
    print("FINN adapter integration: PASS")
    return 0


def _make_model(model_wrapper: Callable[[Any], Any]) -> Any:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    intermediate = helper.make_tensor_value_info("add_out", TensorProto.FLOAT, [1, 4])
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [1, 4], [-2, 1, 3, -4])
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["input", "bias"], ["add_out"], name="Add_0"),
            helper.make_node("Relu", ["add_out"], ["output"], name="Relu_1"),
        ],
        "finn_adapter_integration",
        [input_info],
        [output_info],
        [bias],
        value_info=[intermediate],
    )
    return model_wrapper(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    )


if __name__ == "__main__":
    raise SystemExit(main())
