"""Command-line entry point for TensorWright."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from tensorwright import __version__
from tensorwright.compiler import (
    CompilerError,
    compile_onnx_bundle,
    inspect_bundle_json,
)
from tensorwright.dashboard import DashboardError, generate_dashboard
from tensorwright.minimizer import FailureSignature, MinimizationError, minimize_inputs
from tensorwright.regression import (
    RegressionGenerationError,
    generate_cocotb_regression,
)
from tensorwright.runtime import (
    SimulationConfig,
    SimulationError,
    benchmark_bundle_json,
    simulate_bundle,
)
from tensorwright.trace import (
    AdapterError,
    AdapterRequest,
    ComparisonReport,
    DiagnosisReport,
    ProtocolReport,
    TraceError,
    analyze_protocol_files,
    compare_trace_files,
    default_adapter_registry,
    diagnose_comparison,
    read_trace,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the TensorWright command-line parser."""
    parser = argparse.ArgumentParser(
        prog="tensorwright",
        description=(
            "Hardware-aware compilation and cross-layer verification for quantized "
            "AI accelerators."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")
    compile_command = subparsers.add_parser(
        "compile", help="compile an ONNX model into a validated .twmodel bundle"
    )
    compile_command.add_argument("model", help="input ONNX model")
    compile_command.add_argument("calibration", help="named calibration tensors in NPZ")
    compile_command.add_argument("output", help="destination .twmodel directory")
    compile_command.add_argument(
        "--labels", help="optional UTF-8 file containing one class label per line"
    )
    inspect_bundle_command = subparsers.add_parser(
        "inspect-bundle", help="validate and summarize a .twmodel deployment bundle"
    )
    inspect_bundle_command.add_argument("bundle")
    simulate = subparsers.add_parser(
        "simulate", help="execute a validated .twmodel bundle in the simulation host"
    )
    simulate.add_argument("bundle", help="path to a .twmodel deployment directory")
    simulate.add_argument("--seed", type=int, default=0x7E45)
    simulate.add_argument("--timeout-cycles", type=int, default=1_000_000)
    simulate.add_argument(
        "--no-backpressure", action="store_true", help="keep modeled stream ready high"
    )
    benchmark = subparsers.add_parser(
        "benchmark", help="measure repeatable host-modeled bundle performance"
    )
    benchmark.add_argument("bundle", help="path to a .twmodel deployment directory")
    benchmark.add_argument("--runs", type=int, default=10)
    benchmark.add_argument("--seed", type=int, default=0x7E45)
    benchmark.add_argument(
        "--no-backpressure", action="store_true", help="keep modeled stream ready high"
    )
    minimize = subparsers.add_parser(
        "minimize", help="reduce a failing NPZ input with an external failure oracle"
    )
    minimize.add_argument("input", help="input .npz containing named tensors")
    minimize.add_argument("output", help="destination .npz for minimized tensors")
    minimize.add_argument("--report", help="destination JSON report path")
    minimize.add_argument("--max-evaluations", type=int, default=1000)
    minimize.add_argument("--oracle-timeout", type=float, default=300.0)
    minimize.add_argument(
        "--oracle",
        nargs=argparse.REMAINDER,
        required=True,
        help="oracle command; TensorWright appends the candidate .npz path",
    )
    generate = subparsers.add_parser(
        "generate-regression", help="package a minimized failure as a Cocotb test"
    )
    generate.add_argument("inputs", help="minimized named-tensor .npz")
    generate.add_argument("minimization_report", help="minimizer JSON report")
    generate.add_argument("reference_trace", help="canonical minimized reference trace")
    generate.add_argument("output", help="new or empty output directory")
    generate.add_argument("--name", required=True, help="portable test identifier")
    dashboard = subparsers.add_parser(
        "dashboard", help="generate a self-contained HTML debugging report"
    )
    dashboard.add_argument("reference", help="canonical software-reference trace")
    dashboard.add_argument("candidate", help="canonical RTL or HLS candidate trace")
    dashboard.add_argument("output", help="destination .html report")
    dashboard.add_argument("--minimization-report", help="optional minimizer JSON")
    dashboard.add_argument("--regression-manifest", help="optional regression manifest")
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
    protocol = trace_commands.add_parser(
        "diagnose-protocol", help="analyze streaming and packetization behavior"
    )
    protocol.add_argument("reference", help="canonical software-reference trace")
    protocol.add_argument("candidate", help="canonical RTL or HLS candidate trace")
    protocol.add_argument(
        "--json", action="store_true", help="print the machine-readable protocol report"
    )
    protocol.add_argument("--report", help="also write the JSON report to this path")
    adapters = trace_commands.add_parser(
        "adapters", help="list registered canonical trace adapters"
    )
    adapters.add_argument(
        "--discover",
        action="store_true",
        help="load installed third-party entry points",
    )
    convert = trace_commands.add_parser(
        "convert", help="convert a backend artifact to canonical JSONL"
    )
    convert.add_argument("source", help="adapter-specific source artifact")
    convert.add_argument("output", help="destination canonical .jsonl trace")
    convert.add_argument("--adapter", required=True, help="registered adapter name")
    convert.add_argument(
        "--options",
        default="{}",
        help="JSON object or @path containing adapter-specific options",
    )
    convert.add_argument(
        "--discover",
        action="store_true",
        help="load installed third-party entry points",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TensorWright command-line interface."""
    arguments = build_parser().parse_args(argv)
    if arguments.command == "compile":
        try:
            labels = (
                Path(arguments.labels).read_text(encoding="utf-8").splitlines()
                if arguments.labels
                else None
            )
            bundle_path = compile_onnx_bundle(
                arguments.model,
                arguments.calibration,
                arguments.output,
                labels=labels,
            )
        except (CompilerError, OSError, ValueError) as error:
            print(f"tensorwright: compilation failed: {error}", file=sys.stderr)
            return 1
        print(f"Compiled TensorWright bundle: {bundle_path}")
        return 0
    if arguments.command == "inspect-bundle":
        try:
            print(inspect_bundle_json(arguments.bundle), end="")
        except (CompilerError, OSError, ValueError) as error:
            print(f"tensorwright: bundle inspection failed: {error}", file=sys.stderr)
            return 1
    if arguments.command == "simulate":
        try:
            simulation_result = simulate_bundle(
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
        print(simulation_result.to_json(), end="")
        return 0 if simulation_result.reference_match else 2
    if arguments.command == "benchmark":
        try:
            print(
                benchmark_bundle_json(
                    arguments.bundle,
                    runs=arguments.runs,
                    seed=arguments.seed,
                    randomized_backpressure=not arguments.no_backpressure,
                ),
                end="",
            )
        except (CompilerError, SimulationError, OSError, ValueError) as error:
            print(f"tensorwright: benchmark failed: {error}", file=sys.stderr)
            return 1
        return 0
    if arguments.command == "minimize":
        try:
            input_path = Path(arguments.input)
            output_path = Path(arguments.output)
            if input_path.suffix != ".npz" or output_path.suffix != ".npz":
                raise MinimizationError("Minimizer input and output must use .npz")
            if not arguments.oracle:
                raise MinimizationError("An oracle command is required")
            if arguments.oracle_timeout <= 0:
                raise MinimizationError("Oracle timeout must be positive")
            with np.load(input_path, allow_pickle=False) as archive:
                inputs = {name: archive[name].copy() for name in archive.files}
            with tempfile.TemporaryDirectory(
                prefix="tensorwright-minimize-"
            ) as directory:
                candidate_path = Path(directory) / "candidate.npz"

                def oracle(candidate: dict[str, np.ndarray]) -> FailureSignature | None:
                    np.savez(candidate_path, **candidate)  # type: ignore[arg-type]
                    try:
                        completed = subprocess.run(
                            [*arguments.oracle, str(candidate_path)],
                            check=False,
                            capture_output=True,
                            text=True,
                            timeout=arguments.oracle_timeout,
                        )
                    except subprocess.TimeoutExpired as error:
                        raise MinimizationError(
                            f"Oracle exceeded {arguments.oracle_timeout:g} seconds"
                        ) from error
                    if completed.returncode != 0:
                        detail = completed.stderr.strip() or "no stderr"
                        raise MinimizationError(
                            f"Oracle exited with {completed.returncode}: {detail}"
                        )
                    payload = json.loads(completed.stdout)
                    if payload is None:
                        return None
                    if not isinstance(payload, dict):
                        raise MinimizationError(
                            "Oracle output must be a failure-signature object or null"
                        )
                    return FailureSignature.from_dict(payload)

                minimization_result = minimize_inputs(
                    inputs,
                    oracle,
                    max_evaluations=arguments.max_evaluations,
                )
            report_path = (
                Path(arguments.report)
                if arguments.report
                else output_path.with_suffix(".report.json")
            )
            minimization_result.write(output_path, report_path)
        except (
            MinimizationError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            print(f"tensorwright: minimization failed: {error}", file=sys.stderr)
            return 1
        print(minimization_result.report_json(), end="")
        return 0
    if arguments.command == "generate-regression":
        try:
            package = generate_cocotb_regression(
                arguments.inputs,
                arguments.minimization_report,
                arguments.reference_trace,
                arguments.output,
                name=arguments.name,
            )
        except RegressionGenerationError as error:
            print(
                f"tensorwright: regression generation failed: {error}", file=sys.stderr
            )
            return 1
        print(f"Generated Cocotb regression: {package.path}")
        print(f"Test module: {package.test_path.name}")
        return 0
    if arguments.command == "dashboard":
        try:
            dashboard_result = generate_dashboard(
                arguments.reference,
                arguments.candidate,
                arguments.output,
                minimization_report=arguments.minimization_report,
                regression_manifest=arguments.regression_manifest,
            )
        except (DashboardError, TraceError, OSError, ValueError) as error:
            print(
                f"tensorwright: dashboard generation failed: {error}", file=sys.stderr
            )
            return 1
        print(f"Generated TensorWright dashboard: {dashboard_result.path}")
        return 0 if dashboard_result.comparison.matched else 2
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
    if arguments.command == "trace" and arguments.trace_command == "diagnose-protocol":
        try:
            protocol_report = analyze_protocol_files(
                arguments.reference, arguments.candidate
            )
            if arguments.report:
                report_path = Path(arguments.report)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(protocol_report.to_json(), encoding="utf-8")
        except (TraceError, OSError, ValueError) as error:
            print(f"tensorwright: protocol diagnosis failed: {error}", file=sys.stderr)
            return 1
        if arguments.json:
            print(protocol_report.to_json(), end="")
        else:
            _print_protocol_report(protocol_report)
        return 0 if protocol_report.protocol_ok else 2
    if arguments.command == "trace" and arguments.trace_command == "adapters":
        try:
            registry = default_adapter_registry(discover=arguments.discover)
        except AdapterError as error:
            print(f"tensorwright: adapter discovery failed: {error}", file=sys.stderr)
            return 1
        print("TensorWright trace adapters")
        for descriptor in registry.descriptors():
            print(
                f"{descriptor.name} {descriptor.version} (API {descriptor.api_version})"
            )
            print(f"  Inputs: {', '.join(descriptor.input_formats)}")
            print(f"  Trace points: {', '.join(descriptor.trace_points)}")
            print(f"  {descriptor.description}")
        return 0
    if arguments.command == "trace" and arguments.trace_command == "convert":
        try:
            options_text = arguments.options
            if options_text.startswith("@"):
                options_text = Path(options_text[1:]).read_text(encoding="utf-8")
            options = json.loads(options_text)
            if not isinstance(options, dict):
                raise AdapterError("Adapter options must be a JSON object")
            registry = default_adapter_registry(discover=arguments.discover)
            converted = registry.convert(
                arguments.adapter,
                AdapterRequest(Path(arguments.source), Path(arguments.output), options),
            )
        except (AdapterError, OSError, ValueError, json.JSONDecodeError) as error:
            print(f"tensorwright: trace conversion failed: {error}", file=sys.stderr)
            return 1
        print(f"Generated canonical trace: {converted}")
        return 0
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


def _print_protocol_report(report: ProtocolReport) -> None:
    print("TensorWright Verify protocol diagnosis")
    print(f"Ruleset version: {report.ruleset_version}")
    print(f"Candidate backend: {report.candidate_backend}")
    print(f"Stream events: {report.stream_events}")
    print(f"Protocol result: {'PASS' if report.protocol_ok else 'FAIL'}")
    if not report.findings:
        print("Findings: none")
        return
    print(f"Findings ({len(report.findings)}):")
    for finding in report.findings:
        location = ""
        if finding.event_index is not None:
            location = f" at event {finding.event_index}"
        if finding.cycle is not None:
            location += f", cycle {finding.cycle}"
        print(
            f"  - [{finding.severity}] {finding.rule_id}{location}: {finding.evidence}"
        )
