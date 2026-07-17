"""Run the video-friendly TensorWright reference-versus-RTL demo."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from scripts.run_verilator_tests import (
    ROOT,
    RTL,
    VERIFICATION,
    _build_and_run,
    _convolution_vectors,
    _reference_convolution_trace,
)
from tensorwright.dashboard import generate_dashboard
from tensorwright.trace import (
    TRACE_VERSION,
    TraceEvent,
    analyze_protocol_files,
    compare_trace_files,
    diagnose_comparison,
    write_trace,
)
from tensorwright.trace.adapters.rtl import RtlTraceCapture, read_transfer_log
from verification.generated.test_requant_rounding_case_001 import run_case

BUILD = ROOT / "build" / "demo"


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
    args = parser.parse_args(argv)
    if args.pace < 0:
        parser.error("--pace must be non-negative")

    BUILD.mkdir(parents=True, exist_ok=True)

    _banner()
    _step("1 / 8", "Generating deterministic INT8 convolution workload", args.pace)
    vectors = BUILD / "convolution_vectors.txt"
    case_count, expected = _convolution_vectors(vectors)
    reference = _reference_convolution_trace(BUILD / "reference.jsonl", expected)
    print(f"      {case_count} convolution layers · 18 traced outputs in case 0")

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
    _build_and_run(
        "tb_convolution_engine",
        sources,
        vectors,
        [f"+TRACE_FILE={rounding_log}", "+ALLOW_MISMATCH"],
        build_name="rounding_fault_rtl",
        build_root=BUILD,
        quiet=True,
        verilator_args=["-DTENSORWRIGHT_DEMO_FAULT_REQUANT_ROUND"],
    )
    rounding_trace = _convert_log(
        rounding_log, BUILD / "rounding_fault_rtl.jsonl", "rounding_fault"
    )

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
    print(f"      First divergence: {divergence.compiled_operation_id}")
    print(f"      Coordinate:       {divergence.coordinate}")
    print(f"      Software:         {divergence.reference_value}")
    print(f"      RTL:              {divergence.candidate_value}")
    print(f"      Accepted cycle:   {divergence.candidate_cycle}")
    print(f"      Likely cause: {diagnosis.diagnosis.title}")
    print(f"      Confidence:   {diagnosis.diagnosis.confidence}")

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

    _step("7 / 8", "Running the generated regression before and after fix", args.pace)
    faulty_regression = run_case(faulty=True)
    corrected_regression = run_case()
    if faulty_regression or not corrected_regression:
        raise RuntimeError("Generated requantization regression behaved unexpectedly")
    print("      Faulty RTL:    FAIL")
    print("      Corrected RTL: PASS")
    print("      Regression: verification/generated/test_requant_rounding_case_001.py")

    _step("8 / 8", "Generating video dashboards", args.pace)

    comparison_path = BUILD / "comparison.json"
    diagnosis_path = BUILD / "diagnosis.json"
    comparison_path.write_text(comparison.to_json(), encoding="utf-8")
    diagnosis_path.write_text(diagnosis.to_json(), encoding="utf-8")
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
        arithmetic_evidence={
            "accumulator": 24,
            "bias": -491,
            "biased": -467,
            "multiplier": 1,
            "shift": 2,
            "product": -467,
            "rounding_offset": 2,
            "software_result": -117,
            "rtl_result": -116,
        },
        generated_regression=(
            "verification/generated/test_requant_rounding_case_001.py"
        ),
    )
    protocol_dashboard = generate_dashboard(
        reference,
        protocol_trace,
        BUILD / "protocol.html",
        baseline_candidate_trace=clean_trace,
        scenario_note=(
            "The RTL consumes logical output #5 internally without presenting a "
            "valid/ready transfer. Normal builds preserve every output."
        ),
    )
    summary = {
        "clean_match": clean.to_dict(),
        "fault_comparison": comparison.to_dict(),
        "diagnosis": diagnosis.to_dict(),
        "numerical_dashboard": str(dashboard.path),
        "protocol_report": protocol.to_dict(),
        "protocol_dashboard": str(protocol_dashboard.path),
    }
    (BUILD / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print()
    print("  DEMO COMPLETE")
    print(f"  Numerical dashboard: {dashboard.path}")
    print(f"  Protocol dashboard:  {protocol_dashboard.path}")
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
                model_id="rtl_convolution_regression",
                source_operation_id="synthetic:conv_0",
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
        model_id="rtl_convolution_regression",
        source_operation_id="synthetic:conv_0",
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
