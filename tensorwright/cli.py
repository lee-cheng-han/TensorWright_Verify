"""Command-line entry point for TensorWright."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tensorwright import __version__
from tensorwright.compiler import CompilerError
from tensorwright.runtime import SimulationConfig, SimulationError, simulate_bundle
from tensorwright.trace import (
    ComparisonReport,
    DiagnosisReport,
    TraceError,
    compare_trace_files,
    diagnose_comparison,
    read_trace,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the TensorWright command-line parser."""
    parser = argparse.ArgumentParser(
        prog="tensorwright",
        description=(
            "Cross-layer debugging and differential verification for "
            "quantized AI accelerators."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")
    simulate = subparsers.add_parser(
        "simulate", help="execute a validated .twmodel bundle in the simulation host"
    )
    simulate.add_argument("bundle", help="path to a .twmodel deployment directory")
    simulate.add_argument("--seed", type=int, default=0x7E45)
    simulate.add_argument("--timeout-cycles", type=int, default=1_000_000)
    simulate.add_argument(
        "--no-backpressure", action="store_true", help="keep modeled stream ready high"
    )
    trace = subparsers.add_parser("trace", help="inspect canonical verification traces")
    trace_commands = trace.add_subparsers(dest="trace_command", required=True)
    inspect = trace_commands.add_parser(
        "inspect", help="summarize a canonical JSONL trace"
    )
    inspect.add_argument("path", help="path to a canonical .jsonl trace")
    compare = trace_commands.add_parser(
        "compare", help="align two traces and report their first divergence"
    )
    compare.add_argument("reference", help="canonical software-reference trace")
    compare.add_argument("candidate", help="canonical RTL or HLS candidate trace")
    compare.add_argument(
        "--json", action="store_true", help="print the machine-readable report"
    )
    compare.add_argument("--report", help="also write the JSON report to this path")
    diagnose = trace_commands.add_parser(
        "diagnose", help="classify the first numerical divergence"
    )
    diagnose.add_argument("reference", help="canonical software-reference trace")
    diagnose.add_argument("candidate", help="canonical RTL or HLS candidate trace")
    diagnose.add_argument(
        "--json", action="store_true", help="print the machine-readable diagnosis"
    )
    diagnose.add_argument("--report", help="also write the JSON report to this path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TensorWright command-line interface."""
    arguments = build_parser().parse_args(argv)
    if arguments.command == "simulate":
        try:
            result = simulate_bundle(
                arguments.bundle,
                config=SimulationConfig(
                    seed=arguments.seed,
                    timeout_cycles=arguments.timeout_cycles,
                    randomized_backpressure=not arguments.no_backpressure,
                ),
            )
        except (CompilerError, SimulationError, ValueError) as error:
            print(f"tensorwright: simulation failed: {error}", file=sys.stderr)
            return 1
        print(result.to_json(), end="")
        return 0 if result.reference_match else 2
    if arguments.command == "trace" and arguments.trace_command == "inspect":
        try:
            trace_set = read_trace(arguments.path)
        except TraceError as error:
            print(f"tensorwright: trace inspection failed: {error}", file=sys.stderr)
            return 1
        events = trace_set.events
        operations = sorted({event.compiled_operation_id for event in events})
        tensors = sorted({event.tensor_name for event in events})
        has_cycles = any(event.cycle is not None for event in events)
        has_quantization = any(event.quantization is not None for event in events)
        print("TensorWright Verify trace")
        print(f"Trace version: {events[0].trace_version}")
        print(f"Backend: {events[0].source_backend}")
        print(f"Model: {events[0].model_id}")
        print(f"Events: {len(events)}")
        print(f"Operations ({len(operations)}): {', '.join(operations)}")
        print(f"Tensors ({len(tensors)}): {', '.join(tensors)}")
        print(f"Cycle information: {'yes' if has_cycles else 'no'}")
        print(f"Quantization metadata: {'yes' if has_quantization else 'no'}")
        return 0
    if arguments.command == "trace" and arguments.trace_command == "compare":
        try:
            report = compare_trace_files(arguments.reference, arguments.candidate)
            if arguments.report:
                report_path = Path(arguments.report)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(report.to_json(), encoding="utf-8")
        except (TraceError, OSError, ValueError) as error:
            print(f"tensorwright: trace comparison failed: {error}", file=sys.stderr)
            return 1
        if arguments.json:
            print(report.to_json(), end="")
        else:
            _print_comparison(report)
        return 0 if report.matched else 2
    if arguments.command == "trace" and arguments.trace_command == "diagnose":
        try:
            diagnosis = diagnose_comparison(
                compare_trace_files(arguments.reference, arguments.candidate)
            )
            if arguments.report:
                report_path = Path(arguments.report)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(diagnosis.to_json(), encoding="utf-8")
        except (TraceError, OSError, ValueError) as error:
            print(f"tensorwright: trace diagnosis failed: {error}", file=sys.stderr)
            return 1
        if arguments.json:
            print(diagnosis.to_json(), end="")
        else:
            _print_diagnosis(diagnosis)
        return 0 if diagnosis.matched else 2
    return 0


def _print_comparison(report: ComparisonReport) -> None:
    print("TensorWright Verify comparison")
    print(f"Reference backend: {report.reference_backend}")
    print(f"Candidate backend: {report.candidate_backend}")
    print(f"Model: {report.model_id}")
    print(f"Matched values: {report.matched_values}")
    if report.matched:
        print("Result: MATCH")
        return
    divergence = report.first_divergence
    assert divergence is not None
    print("Result: DIVERGENCE")
    print(f"Kind: {divergence.kind}")
    print(f"Operation: {divergence.compiled_operation_id}")
    print(f"Tensor: {divergence.tensor_name}")
    print(f"Trace point: {divergence.trace_point}")
    print(f"Coordinate: {divergence.coordinate}")
    print(f"Reference: {divergence.reference_value}")
    print(f"Candidate: {divergence.candidate_value}")
    if divergence.candidate_cycle is not None:
        print(f"Candidate cycle: {divergence.candidate_cycle}")


def _print_diagnosis(report: DiagnosisReport) -> None:
    print("TensorWright Verify diagnosis")
    print(f"Ruleset version: {report.ruleset_version}")
    if report.matched:
        print("Result: MATCH — no diagnosis required")
        return
    diagnosis = report.diagnosis
    assert diagnosis is not None
    divergence = report.comparison.first_divergence
    assert divergence is not None
    print(f"First divergence: {divergence.compiled_operation_id}")
    print(f"Tensor coordinate: {divergence.tensor_name} {divergence.coordinate}")
    print(f"Likely cause: {diagnosis.title}")
    print(f"Rule: {diagnosis.rule_id}")
    print(f"Confidence: {diagnosis.confidence}")
    print("Evidence:")
    for item in diagnosis.evidence:
        print(f"  - {item}")
    print("Recommended checks:")
    for item in diagnosis.recommended_checks:
        print(f"  - {item}")
