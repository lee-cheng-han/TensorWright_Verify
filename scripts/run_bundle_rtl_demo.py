"""Compile an ONNX convolution bundle and execute that exact bundle on RTL."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from scripts.run_verilator_tests import (
    BUILD as RTL_BUILD,
)
from scripts.run_verilator_tests import (
    RTL,
    VERIFICATION,
    _build_and_run,
    _reference_convolution_trace,
)
from tensorwright.compiler import compile_onnx_bundle
from tensorwright.runtime import extract_fixed_convolution, write_convolution_vector
from tensorwright.trace import compare_trace_files
from tensorwright.trace.adapters.rtl import RtlTraceCapture, read_transfer_log

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "bundle_rtl_demo"


def main() -> int:
    BUILD.mkdir(parents=True, exist_ok=True)
    model_path = BUILD / "compiled_convolution.onnx"
    calibration_path = BUILD / "calibration.npz"
    bundle_path = BUILD / "compiled_convolution.twmodel"
    if bundle_path.exists():
        shutil.rmtree(bundle_path)
    _write_model(model_path)
    calibration = _calibration_samples()
    np.savez(calibration_path, input=calibration)
    compile_onnx_bundle(model_path, calibration_path, bundle_path)
    invocation = extract_fixed_convolution(bundle_path)
    vector_path = write_convolution_vector(invocation, BUILD / "bundle_vectors.txt")
    transfer_path = BUILD / "rtl_transfers.txt"
    _build_and_run(
        "tb_convolution_engine",
        [
            RTL / "compute" / "tensorwright_multiplier.sv",
            RTL / "compute" / "tensorwright_adder_tree.sv",
            RTL / "postprocess" / "tensorwright_postprocess.sv",
            RTL / "compute" / "tensorwright_arithmetic_core.sv",
            RTL / "engine" / "tensorwright_convolution_engine.sv",
            VERIFICATION / "tb_convolution_engine.sv",
        ],
        vector_path,
        [f"+TRACE_FILE={transfer_path}"],
        build_name="bundle_rtl_convolution",
        build_root=RTL_BUILD,
    )
    capture = RtlTraceCapture(
        enabled=True,
        run_id="compiled_bundle_rtl",
        model_id=invocation.model,
        source_operation_id=invocation.source_operation_id,
        compiled_operation_id="compiled:op_0000",
        operation_name=invocation.operation,
        tensor_name="output",
        shape=[1, 2, 3, 3],
        source_backend="tensorwright.verilator_rtl",
    )
    transfers = read_transfer_log(transfer_path)
    for transfer in transfers:
        capture.record(transfer)
    rtl_trace = capture.write(BUILD / "rtl_trace.jsonl")
    assert rtl_trace is not None
    reference_trace = _reference_convolution_trace(
        BUILD / "reference_trace.jsonl",
        invocation.expected,
        source_operation_id=invocation.source_operation_id,
        model_id=invocation.model,
        operation_name=invocation.operation,
        tensor_name="output",
    )
    comparison = compare_trace_files(reference_trace, rtl_trace)
    if not comparison.matched:
        raise RuntimeError(
            f"Compiled bundle disagrees with RTL: {comparison.to_json()}"
        )
    first_cycle = transfers[0].cycle
    last_cycle = transfers[-1].cycle
    report = {
        "status": "pass",
        "model": invocation.model,
        "bundle": str(bundle_path.relative_to(ROOT)),
        "operation": invocation.operation,
        "source_operation_id": invocation.source_operation_id,
        "matched_values": comparison.matched_values,
        "accepted_output_cycles": {
            "first": first_cycle,
            "last": last_cycle,
            "span": last_cycle - first_cycle + 1,
        },
        "observed_outputs_per_cycle": len(transfers) / (last_cycle - first_cycle + 1),
        "output_phase_latency_ns_at_100mhz": (last_cycle - first_cycle + 1) * 10,
        "observed_outputs_per_second_at_100mhz": (
            len(transfers) / (last_cycle - first_cycle + 1)
        )
        * 100_000_000,
        "clock_hz": 100_000_000,
        "rtl_execution": "verilator",
    }
    report_path = BUILD / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Compiler-generated .twmodel -> Verilator RTL: PASS")
    print(f"Matched outputs: {comparison.matched_values}/18")
    print(f"Bundle: {bundle_path}")
    print(f"Report: {report_path}")
    return 0


def _calibration_samples() -> np.ndarray:
    values = np.arange(4 * 75, dtype=np.float32).reshape(4, 1, 3, 5, 5)
    return ((values * 13 + 7) % 31 - 15).astype(np.float32)


def _write_model(path: Path) -> None:
    weights = (
        (np.arange(54, dtype=np.float32).reshape(2, 3, 3, 3) * 7 + 3) % 15 - 7
    ) / 8.0
    bias = np.asarray([-0.375, 0.625], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["input", "weights", "bias"],
                ["output"],
                name="compiled_conv",
            )
        ],
        "compiled_bundle_convolution",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 5, 5])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 3, 3])],
        [
            numpy_helper.from_array(weights.astype(np.float32), "weights"),
            numpy_helper.from_array(bias, "bias"),
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
