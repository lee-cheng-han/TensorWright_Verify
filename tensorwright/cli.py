"""Command-line entry point for TensorWright."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from tensorwright import __version__
from tensorwright.compiler import CompilerError
from tensorwright.runtime import SimulationConfig, SimulationError, simulate_bundle
from tensorwright.trace import TraceError, read_trace


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
        operations = sorted({event.operation_id for event in events})
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
    return 0
