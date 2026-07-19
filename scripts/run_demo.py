"""Run the video-friendly TensorWright reference-versus-RTL demo."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from scripts.run_verilator_tests import (
    ROOT,
    RTL,
    VERIFICATION,
    RtlArithmeticSample,
    _build_and_run,
    _convolution_vectors,
    _reference_convolution_trace,
    read_arithmetic_log,
)
from tensorwright.compiler import load_onnx
from tensorwright.dashboard import generate_dashboard, generate_presentation_dashboard
from tensorwright.regression import generate_rtl_arithmetic_regression
from tensorwright.trace import (
    TRACE_VERSION,
    TraceEvent,
    analyze_protocol_files,
    compare_trace_files,
    diagnose_comparison,
    write_trace,
)
from tensorwright.trace.adapters.rtl import (
    RtlTraceCapture,
    RtlTransfer,
    read_transfer_log,
)

BUILD = ROOT / "build" / "demo"
DEMO_MODEL_ID = "tensorwright_onnx_conv_demo"
DEMO_SOURCE_ID = "onnx:demo_conv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a real Verilator differential-debugging demonstration."
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=0.45 if sys.stdout.isatty() else 0.0,
        help="seconds between presentation steps (default: 0.45 on a terminal)",
    )
    parser.add_argument(
        "--focus",
        choices=("all", "clean", "numerical", "protocol"),
        default="all",
        help="dashboard to highlight at completion (all scenarios are verified)",
    )
    args = parser.parse_args(argv)
    if args.pace < 0:
        parser.error("--pace must be non-negative")

    BUILD.mkdir(parents=True, exist_ok=True)

    _banner()
    _step("1 / 8", "Generating deterministic INT8 convolution workload", args.pace)
    vectors = BUILD / "convolution_vectors.txt"
    case_count, expected = _convolution_vectors(vectors)
    model_path = _create_onnx_workload(vectors, BUILD / "tiny_conv.onnx")
    graph = load_onnx(model_path)
    reference = _reference_convolution_trace(
        BUILD / "reference.jsonl",
        expected,
        source_operation_id=DEMO_SOURCE_ID,
        model_id=DEMO_MODEL_ID,
    )
    print(
        f"      Imported {model_path.name} · {len(graph.operations)} ONNX operation · "
        f"{case_count} deterministic vector cases · 18 traced outputs"
    )

    sources = [
        RTL / "compute" / "tensorwright_multiplier.sv",
        RTL / "compute" / "tensorwright_adder_tree.sv",
        RTL / "postprocess" / "tensorwright_postprocess.sv",
        RTL / "compute" / "tensorwright_arithmetic_core.sv",
        RTL / "engine" / "tensorwright_convolution_engine.sv",
        VERIFICATION / "tb_convolution_engine.sv",
    ]

    _step("2 / 8", "Compiling and running the known-good RTL", args.pace)
    clean_log = BUILD / "clean_transfers.txt"
    _build_and_run(
        "tb_convolution_engine",
        sources,
        vectors,
        [f"+TRACE_FILE={clean_log}"],
        build_name="clean_rtl",
        build_root=BUILD,
        quiet=True,
    )
    clean_trace = _convert_log(clean_log, BUILD / "clean_rtl.jsonl", "clean")
    clean = compare_trace_files(reference, clean_trace)
    if not clean.matched:
        raise RuntimeError("Known-good RTL unexpectedly diverged")
    print(f"      PASS · {clean.matched_values}/{clean.reference_values} values match")

    _step("3 / 8", "Injecting a real requantization rounding defect", args.pace)
    print("      Defect: skip round-to-nearest bias and truncate the product")
    rounding_log = BUILD / "rounding_fault_transfers.txt"
    arithmetic_log = BUILD / "rounding_fault_arithmetic.txt"
    _build_and_run(
        "tb_convolution_engine",
        sources,
        vectors,
        [
            f"+TRACE_FILE={rounding_log}",
            f"+ARITH_TRACE_FILE={arithmetic_log}",
            "+ALLOW_MISMATCH",
        ],
        build_name="rounding_fault_rtl",
        build_root=BUILD,
        quiet=True,
        verilator_args=["-DTENSORWRIGHT_DEMO_FAULT_REQUANT_ROUND"],
    )
    rounding_trace = _convert_log(
        rounding_log, BUILD / "rounding_fault_rtl.jsonl", "rounding_fault"
    )
    arithmetic_samples = read_arithmetic_log(arithmetic_log)
    reference_pipeline = _write_pipeline_trace(
        BUILD / "reference_pipeline.jsonl",
        _reference_arithmetic_samples(vectors),
        read_transfer_log(rounding_log),
        backend="tensorwright.python_reference",
    )
    candidate_pipeline = _write_pipeline_trace(
        BUILD / "rounding_fault_pipeline.jsonl",
        arithmetic_samples,
        read_transfer_log(rounding_log),
        backend="tensorwright.verilator_rtl",
    )
    pipeline_comparison = compare_trace_files(reference_pipeline, candidate_pipeline)
    if (
        pipeline_comparison.first_divergence is None
        or pipeline_comparison.first_divergence.trace_point != "post_requantization"
    ):
        raise RuntimeError("Canonical pipeline trace did not localize requantization")
    pipeline_divergence = pipeline_comparison.first_divergence

    _step("4 / 8", "Diagnosing the numerical divergence", args.pace)
    comparison = compare_trace_files(reference, rounding_trace)
    diagnosis = diagnose_comparison(comparison)
    if (
        comparison.matched
        or comparison.first_divergence is None
        or diagnosis.diagnosis is None
        or diagnosis.diagnosis.rule_id != "requantization_rounding_mismatch"
    ):
        raise RuntimeError("Requantization fault was not diagnosed correctly")
    divergence = comparison.first_divergence
    sample = next(
        item
        for item in arithmetic_samples
        if item.sequence == comparison.matched_values
    )
    print(f"      First divergence: {divergence.compiled_operation_id}")
    print(f"      Coordinate:       {divergence.coordinate}")
    print(f"      Software:         {divergence.reference_value}")
    print(f"      RTL:              {divergence.candidate_value}")
    print(f"      Accepted cycle:   {divergence.candidate_cycle}")
    print("      Cause:            Confirmed requantization rounding mismatch")
    print("      Confidence:       confirmed (sampled internal RTL evidence)")

    _step("5 / 8", "Injecting a dropped RTL output transfer", args.pace)
    print("      Defect: logical output #5 is consumed without valid/ready")
    protocol_log = BUILD / "protocol_fault_transfers.txt"
    _build_and_run(
        "tb_convolution_engine",
        sources,
        vectors,
        [
            f"+TRACE_FILE={protocol_log}",
            "+ALLOW_MISMATCH",
            "+EXPECT_DROPPED_TRANSFER",
        ],
        build_name="protocol_fault_rtl",
        build_root=BUILD,
        quiet=True,
        verilator_args=["-DTENSORWRIGHT_DEMO_FAULT_DROPPED_TRANSFER"],
    )
    protocol_trace = _convert_log(
        protocol_log,
        BUILD / "protocol_fault_rtl.jsonl",
        "protocol_fault",
        preserve_sequence=True,
    )

    _step("6 / 8", "Diagnosing the streaming failure", args.pace)
    protocol = analyze_protocol_files(reference, protocol_trace)
    protocol_rules = {finding.rule_id for finding in protocol.findings}
    if protocol.protocol_ok or "missing_output_transfer" not in protocol_rules:
        raise RuntimeError("Dropped transfer was not diagnosed correctly")
    print("      Protocol:       FAIL")
    print("      Missing output: coordinate [0, 0, 1, 1]")
    print(f"      Findings:       {', '.join(sorted(protocol_rules))}")

    _step("7 / 8", "Generating a regression from the first divergence", args.pace)
    arithmetic_evidence = {
        "source": "sampled directly from Verilator RTL internals",
        "localized_trace_point": pipeline_divergence.trace_point,
        "cycle": sample.cycle,
        "accepted_cycle": divergence.candidate_cycle,
        "accumulator": sample.accumulator,
        "bias": sample.bias,
        "biased": sample.post_bias,
        "multiplier": sample.multiplier,
        "shift": sample.shift,
        "product": sample.product,
        "rtl_rounded": sample.rounded,
        "rounding_offset": 1 << (sample.shift - 1) if sample.shift else 0,
        "software_result": divergence.reference_value,
        "rtl_result": sample.result,
    }
    regression_package = generate_rtl_arithmetic_regression(
        pipeline_comparison,
        arithmetic_evidence,
        BUILD / "generated_regression",
        name="requantization_first_divergence",
    )
    regression_relative_path = regression_package.test_path.relative_to(ROOT)
    faulty_regression = _run_generated_regression(
        regression_package.test_path, faulty=True
    )
    corrected_regression = _run_generated_regression(regression_package.test_path)
    if faulty_regression or not corrected_regression:
        raise RuntimeError("Generated requantization regression behaved unexpectedly")
    print("      Faulty RTL:    FAIL")
    print("      Corrected RTL: PASS")
    print(f"      Generated:     {regression_package.test_path}")

    _step("8 / 8", "Generating video dashboards", args.pace)

    comparison_path = BUILD / "comparison.json"
    diagnosis_path = BUILD / "diagnosis.json"
    pipeline_comparison_path = BUILD / "pipeline_comparison.json"
    numerical_performance = _performance(read_transfer_log(rounding_log))
    dashboard = generate_dashboard(
        reference,
        rounding_trace,
        BUILD / "index.html",
        baseline_candidate_trace=clean_trace,
        scenario_note=(
            "TensorWright found the first incorrect hardware result and traced it to "
            "missing rounding logic in the RTL requantizer. One bad stage produces "
            "four wrong values in this 18-value tensor; real models can contain "
            "millions."
        ),
        arithmetic_evidence=arithmetic_evidence,
        generated_regression=regression_relative_path,
        performance=numerical_performance,
    )
    presentation_dashboard = generate_presentation_dashboard(
        BUILD / "presentation.html",
        dashboard,
        arithmetic_evidence=arithmetic_evidence,
        performance=numerical_performance,
        regression_path=regression_relative_path,
    )
    comparison_path.write_text(dashboard.comparison.to_json(), encoding="utf-8")
    diagnosis_path.write_text(dashboard.diagnosis.to_json(), encoding="utf-8")
    pipeline_comparison_path.write_text(pipeline_comparison.to_json(), encoding="utf-8")
    protocol_dashboard = generate_dashboard(
        reference,
        protocol_trace,
        BUILD / "protocol.html",
        baseline_candidate_trace=clean_trace,
        scenario_note=(
            "The RTL consumes logical output #5 internally without presenting a "
            "valid/ready transfer. Normal builds preserve every output."
        ),
        performance=_performance(read_transfer_log(protocol_log)),
    )
    summary = {
        "clean_match": clean.to_dict(),
        "fault_comparison": dashboard.comparison.to_dict(),
        "diagnosis": dashboard.diagnosis.to_dict(),
        "pipeline_comparison": pipeline_comparison.to_dict(),
        "numerical_dashboard": str(dashboard.path.relative_to(ROOT)),
        "presentation_dashboard": str(presentation_dashboard.relative_to(ROOT)),
        "protocol_report": protocol.to_dict(),
        "protocol_dashboard": str(protocol_dashboard.path.relative_to(ROOT)),
    }
    (BUILD / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_archive = _package_report(
        BUILD / "tensorwright-demo-report.zip",
        [
            BUILD / "index.html",
            presentation_dashboard,
            BUILD / "protocol.html",
            BUILD / "summary.json",
            comparison_path,
            diagnosis_path,
            pipeline_comparison_path,
            reference,
            clean_trace,
            rounding_trace,
            protocol_trace,
            arithmetic_log,
            reference_pipeline,
            candidate_pipeline,
            regression_package.test_path,
            regression_package.manifest_path,
            model_path,
            BUILD / "reference_input.npy",
        ],
    )

    print()
    print("  DEMO COMPLETE")
    print(f"  Numerical dashboard: {dashboard.path}")
    print(f"  Presentation view:   {presentation_dashboard}")
    print(f"  Protocol dashboard:  {protocol_dashboard.path}")
    print(f"  Portable report:     {report_archive}")
    if args.focus == "clean":
        print("  Clean baseline verified: 18/18 values match.")
    elif args.focus == "numerical":
        print("  Open index.html for the guided numerical-fault story.")
    elif args.focus == "protocol":
        print("  Open protocol.html for the guided protocol-fault story.")
    else:
        print("  Open index.html, then protocol.html for the video reveal.")
    return 0


def _convert_log(
    source: Path,
    destination: Path,
    run: str,
    *,
    preserve_sequence: bool = False,
) -> Path:
    transfers = read_transfer_log(source)
    if preserve_sequence:
        events = [
            TraceEvent(
                trace_version=TRACE_VERSION,
                event_type="scalar",
                run_id=f"demo_{run}",
                source_backend="tensorwright.verilator_rtl",
                model_id=DEMO_MODEL_ID,
                source_operation_id=DEMO_SOURCE_ID,
                compiled_operation_id="compiled:op_0000",
                fused_source_operation_ids=[],
                graph_stage="rtl_execution",
                operation_name="conv_0",
                operation_type="Conv",
                hardware_stage="convolution_output_stream",
                trace_point="stream_transfer",
                tensor_name="conv_0_output",
                shape=[1, 2, 3, 3],
                layout="NCHW",
                dtype="int8",
                coordinate=_unravel(transfer.sequence, [1, 2, 3, 3]),
                value=transfer.value,
                cycle=transfer.cycle,
                metadata={
                    "valid": transfer.valid,
                    "ready": transfer.ready,
                    "tlast": transfer.last,
                    "sequence": transfer.sequence,
                },
            )
            for transfer in transfers
        ]
        return write_trace(destination, events)
    capture = RtlTraceCapture(
        enabled=True,
        run_id=f"demo_{run}",
        model_id=DEMO_MODEL_ID,
        source_operation_id=DEMO_SOURCE_ID,
        compiled_operation_id="compiled:op_0000",
        operation_name="conv_0",
        tensor_name="conv_0_output",
        shape=[1, 2, 3, 3],
        source_backend="tensorwright.verilator_rtl",
    )
    for transfer in transfers:
        capture.record(transfer)
    output = capture.write(destination)
    assert output is not None
    return output


def _unravel(index: int, shape: list[int]) -> list[int]:
    coordinate = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        coordinate[axis] = index % shape[axis]
        index //= shape[axis]
    return coordinate


def _reference_arithmetic_samples(vectors: Path) -> list[RtlArithmeticSample]:
    """Recompute case zero independently for canonical software stage traces."""
    values = [
        int(value)
        for value in vectors.read_text(encoding="utf-8").splitlines()[1].split()
    ]
    biases = values[0:2]
    multipliers = values[2:4]
    shifts = values[4:6]
    relu = [bool(value) for value in values[6:8]]
    weights = values[8:62]
    activations = values[62:137]
    expected = values[137:155]
    samples: list[RtlArithmeticSample] = []
    for sequence in range(18):
        output_channel, remainder = divmod(sequence, 9)
        output_y, output_x = divmod(remainder, 3)
        accumulator = 0
        for input_channel in range(3):
            for kernel_y in range(3):
                for kernel_x in range(3):
                    activation_index = (
                        input_channel * 25
                        + (output_y + kernel_y) * 5
                        + output_x
                        + kernel_x
                    )
                    weight_index = (
                        output_channel * 27
                        + input_channel * 9
                        + kernel_y * 3
                        + kernel_x
                    )
                    accumulator += activations[activation_index] * weights[weight_index]
        post_bias = accumulator + biases[output_channel]
        product = post_bias * multipliers[output_channel]
        magnitude = abs(product)
        shift = shifts[output_channel]
        rounded_magnitude = (
            magnitude if shift == 0 else (magnitude + (1 << (shift - 1))) >> shift
        )
        rounded = -rounded_magnitude if product < 0 else rounded_magnitude
        if relu[output_channel] and rounded < 0:
            result = 0
        else:
            result = max(-128, min(127, rounded))
        if result != expected[sequence]:
            raise RuntimeError(
                "Independent reference arithmetic disagrees with vectors"
            )
        samples.append(
            RtlArithmeticSample(
                sequence,
                0,
                accumulator,
                biases[output_channel],
                post_bias,
                multipliers[output_channel],
                shift,
                product,
                rounded,
                result,
            )
        )
    return samples


def _write_pipeline_trace(
    destination: Path,
    samples: list[RtlArithmeticSample],
    transfers: list[RtlTransfer],
    *,
    backend: str,
) -> Path:
    """Write canonical accumulator-to-output events for one convolution."""
    transfer_by_sequence = {transfer.sequence: transfer for transfer in transfers}
    events: list[TraceEvent] = []
    is_reference = backend == "tensorwright.python_reference"
    for sample in samples:
        coordinate = _unravel(sample.sequence, [1, 2, 3, 3])
        transfer = transfer_by_sequence[sample.sequence]
        for trace_point, value, dtype in (
            ("accumulator", sample.accumulator, "int32"),
            ("post_bias", sample.post_bias, "int32"),
            ("post_requantization", sample.rounded, "int64"),
            ("operation_output", sample.result, "int8"),
        ):
            cycle = None
            if not is_reference:
                cycle = (
                    transfer.cycle
                    if trace_point == "operation_output"
                    else sample.cycle
                )
            events.append(
                TraceEvent(
                    trace_version=TRACE_VERSION,
                    event_type="scalar",
                    run_id="demo_reference_pipeline"
                    if is_reference
                    else "demo_rtl_pipeline",
                    source_backend=backend,
                    model_id=DEMO_MODEL_ID,
                    source_operation_id=DEMO_SOURCE_ID,
                    compiled_operation_id="compiled:op_0000",
                    fused_source_operation_ids=[],
                    graph_stage="post_quantization"
                    if is_reference
                    else "rtl_execution",
                    operation_name="conv_0",
                    operation_type="Conv",
                    hardware_stage="software_reference"
                    if is_reference
                    else "rtl_postprocess",
                    trace_point=trace_point,
                    tensor_name="conv_0_output",
                    shape=[1, 2, 3, 3],
                    layout="NCHW",
                    dtype=dtype,
                    coordinate=coordinate,
                    value=value,
                    cycle=cycle,
                    metadata={"sequence": sample.sequence},
                )
            )
    return write_trace(destination, events)


def _run_generated_regression(path: Path, *, faulty: bool = False) -> bool:
    environment = os.environ.copy()
    if faulty:
        environment["TENSORWRIGHT_REGRESSION_FAULTY"] = "1"
    return (
        subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            env=environment,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _performance(transfers: list[RtlTransfer]) -> dict[str, object]:
    if not transfers:
        return {"Accepted outputs": 0}
    first_cycle = transfers[0].cycle
    last_cycle = transfers[-1].cycle
    span = max(1, last_cycle - first_cycle + 1)
    throughput = len(transfers) / span
    return {
        "Accepted outputs": len(transfers),
        "Output cycle span": span,
        "Observed stream throughput": f"{throughput:.3f} outputs/cycle",
        "Traced inference MACs": 18 * 3 * 3 * 3,
        "FPGA utilization": "not measured; run synthesis",
    }


def _package_report(destination: Path, sources: list[Path]) -> Path:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            if source.is_file():
                archive.write(source, source.relative_to(ROOT))
    return destination


def _create_onnx_workload(vectors: Path, destination: Path) -> Path:
    values = [
        int(value)
        for value in vectors.read_text(encoding="utf-8").splitlines()[1].split()
    ]
    biases = np.asarray(values[0:2], dtype=np.float32)
    weights = np.asarray(values[8:62], dtype=np.float32).reshape(2, 3, 3, 3)
    activations = np.asarray(values[62:137], dtype=np.float32).reshape(1, 3, 5, 5)
    np.save(destination.parent / "reference_input.npy", activations, allow_pickle=False)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["input", "weights", "bias"],
                ["output"],
                name="demo_conv",
            )
        ],
        "tensorwright_demo",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 5, 5])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 3, 3])],
        [
            numpy_helper.from_array(weights, "weights"),
            numpy_helper.from_array(biases, "bias"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="tensorwright-demo",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, destination)
    return destination


def _banner() -> None:
    print()
    print("  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print("  ┃  TensorWright Verify                                ┃")
    print("  ┃  Find the first hardware/software mismatch.         ┃")
    print("  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    print()


def _step(number: str, title: str, pace: float) -> None:
    if pace:
        time.sleep(pace)
    print(f"  [{number}] {title}...")


if __name__ == "__main__":
    raise SystemExit(main())
