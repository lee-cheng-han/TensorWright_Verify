"""Generate Python-reference vectors and execute self-checking Verilator tests."""

from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tensorwright.reference import requantize_int32
from tensorwright.trace.adapters.rtl import RtlTraceCapture, read_transfer_log
from tensorwright.trace.compare import compare_trace_files
from tensorwright.trace.schema import TRACE_VERSION, TraceEvent, write_trace

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "rtl_vectors"
RTL = ROOT / "rtl"
VERIFICATION = ROOT / "verification" / "systemverilog"


@dataclass(frozen=True)
class RtlArithmeticSample:
    """Postprocess values sampled from the real RTL on a valid input cycle."""

    sequence: int
    cycle: int
    accumulator: int
    bias: int
    post_bias: int
    multiplier: int
    shift: int
    product: int
    rounded: int
    result: int


def read_arithmetic_log(path: Path) -> list[RtlArithmeticSample]:
    samples: list[RtlArithmeticSample] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 10:
            raise ValueError(f"Malformed RTL arithmetic log line {line_number}")
        samples.append(RtlArithmeticSample(*(int(field) for field in fields)))
    return samples


def _run(command: list[str], *, quiet: bool = False) -> None:
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
    )


def _postprocess_vectors(path: Path) -> int:
    cases = [
        (0, 0, 1, 0, False),
        (3, 0, 1, 1, False),
        (-3, 0, 1, 1, False),
        (1000, 0, 1, 0, False),
        (-1000, 0, 1, 0, False),
        (-10, 0, 1, 0, True),
        ((1 << 31) - 1, 0, 1 << 30, 62, False),
        (-(1 << 31), 0, 1 << 30, 62, False),
    ]
    random_source = random.Random(0xA511)
    cases.extend(
        (
            random_source.randint(-1_000_000, 1_000_000),
            random_source.randint(-100_000, 100_000),
            random_source.randint(0, (1 << 31) - 1),
            random_source.randint(0, 70),
            bool(random_source.getrandbits(1)),
        )
        for _ in range(500)
    )
    lines = [
        f"{acc} {bias} {multiplier} {shift} {int(relu)} "
        f"{requantize_int32(acc, bias, multiplier, shift, relu=relu)}"
        for acc, bias, multiplier, shift, relu in cases
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(cases)


def _core_vectors(path: Path) -> int:
    random_source = random.Random(0xC0DE5)
    lines: list[str] = []
    count = 150
    for _ in range(count):
        cycle_count = random_source.randint(1, 8)
        accumulator = 0
        bias = random_source.randint(-10_000, 10_000)
        multiplier = random_source.randint(0, (1 << 31) - 1)
        shift = random_source.randint(0, 45)
        relu = bool(random_source.getrandbits(1))
        cycles: list[tuple[list[int], list[int]]] = []
        for _ in range(cycle_count):
            activations = [random_source.randint(-128, 127) for _ in range(9)]
            weights = [random_source.randint(-128, 127) for _ in range(9)]
            accumulator += sum(
                activation * weight
                for activation, weight in zip(activations, weights, strict=True)
            )
            cycles.append((activations, weights))
        expected = requantize_int32(accumulator, bias, multiplier, shift, relu=relu)
        lines.append(
            f"{cycle_count} {bias} {multiplier} {shift} {int(relu)} {expected}"
        )
        for activations, weights in cycles:
            lines.append(
                " ".join(
                    f"{activation} {weight}"
                    for activation, weight in zip(activations, weights, strict=True)
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return count


def _convolution_vectors(path: Path) -> tuple[int, list[int]]:
    random_source = random.Random(0xC07E)
    case_count = 20
    lines = [str(case_count)]
    first_expected: list[int] = []
    for case_index in range(case_count):
        biases = [random_source.randint(-500, 500) for _ in range(2)]
        multipliers = [random_source.randint(1, 4) for _ in range(2)]
        shifts = [random_source.randint(1, 4) for _ in range(2)]
        relu = [bool(random_source.getrandbits(1)) for _ in range(2)]
        weights = [random_source.randint(-8, 7) for _ in range(2 * 3 * 9)]
        activations = [random_source.randint(-16, 15) for _ in range(3 * 5 * 5)]
        expected: list[int] = []
        for output_channel in range(2):
            for output_y in range(3):
                for output_x in range(3):
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
                                accumulator += (
                                    activations[activation_index]
                                    * weights[weight_index]
                                )
                    expected.append(
                        requantize_int32(
                            accumulator,
                            biases[output_channel],
                            multipliers[output_channel],
                            shifts[output_channel],
                            relu=relu[output_channel],
                        )
                    )
        values = [
            *biases,
            *multipliers,
            *shifts,
            *[int(value) for value in relu],
            *weights,
            *activations,
            *expected,
        ]
        lines.append(" ".join(str(value) for value in values))
        if case_index == 0:
            first_expected = expected
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return case_count, first_expected


def _reference_convolution_trace(
    path: Path,
    values: list[int],
    *,
    source_operation_id: str = "synthetic:conv_0",
    model_id: str = "rtl_convolution_regression",
) -> Path:
    events = []
    for sequence, value in enumerate(values):
        output_channel, remainder = divmod(sequence, 9)
        output_y, output_x = divmod(remainder, 3)
        events.append(
            TraceEvent(
                trace_version=TRACE_VERSION,
                event_type="scalar",
                run_id="python_convolution_case_0000",
                source_backend="tensorwright.python_reference",
                model_id=model_id,
                source_operation_id=source_operation_id,
                compiled_operation_id="compiled:op_0000",
                fused_source_operation_ids=[],
                graph_stage="post_quantization",
                operation_name="conv_0",
                operation_type="Conv",
                hardware_stage="software_operation_output",
                trace_point="operation_output",
                tensor_name="conv_0_output",
                shape=[1, 2, 3, 3],
                layout="NCHW",
                dtype="int8",
                value=value,
                coordinate=[0, output_channel, output_y, output_x],
            )
        )
    return write_trace(path, events)


def _build_and_run(
    top: str,
    sources: list[Path],
    vector_file: Path | None = None,
    plusargs: list[str] | None = None,
    *,
    build_name: str | None = None,
    build_root: Path = BUILD,
    quiet: bool = False,
    verilator_args: list[str] | None = None,
) -> None:
    build_dir = build_root / (build_name or top)
    build_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "verilator",
            "--binary",
            "--timing",
            "--assert",
            "-Wall",
            "-Wno-fatal",
            "-Wno-BLKSEQ",
            "-Wno-UNUSEDSIGNAL",
            "-Wno-WIDTHEXPAND",
            "-Wno-WIDTHTRUNC",
            "--top-module",
            top,
            *(verilator_args or []),
            "--Mdir",
            str(build_dir),
            *[str(source) for source in sources],
        ],
        quiet=quiet,
    )
    run_command = [str(build_dir / f"V{top}")]
    if vector_file is not None:
        run_command.append(f"+VECTOR_FILE={vector_file}")
    run_command.extend(plusargs or [])
    _run(run_command, quiet=quiet)


def main() -> int:
    BUILD.mkdir(parents=True, exist_ok=True)
    postprocess_file = BUILD / "postprocess_vectors.txt"
    core_file = BUILD / "core_vectors.txt"
    convolution_file = BUILD / "convolution_vectors.txt"
    postprocess_count = _postprocess_vectors(postprocess_file)
    core_count = _core_vectors(core_file)
    convolution_count, first_convolution_expected = _convolution_vectors(
        convolution_file
    )
    rtl_transfer_file = BUILD / "convolution_rtl_transfers.txt"
    _build_and_run(
        "tb_primitives",
        [
            RTL / "compute" / "tensorwright_multiplier.sv",
            RTL / "compute" / "tensorwright_mac.sv",
            RTL / "compute" / "tensorwright_adder_tree.sv",
            RTL / "compute" / "tensorwright_channel_accumulator.sv",
            VERIFICATION / "tb_primitives.sv",
        ],
    )
    _build_and_run(
        "tb_postprocess",
        [
            RTL / "postprocess" / "tensorwright_postprocess.sv",
            VERIFICATION / "tb_postprocess.sv",
        ],
        postprocess_file,
    )
    _build_and_run(
        "tb_arithmetic_core",
        [
            RTL / "compute" / "tensorwright_multiplier.sv",
            RTL / "compute" / "tensorwright_adder_tree.sv",
            RTL / "postprocess" / "tensorwright_postprocess.sv",
            RTL / "compute" / "tensorwright_arithmetic_core.sv",
            VERIFICATION / "tb_arithmetic_core.sv",
        ],
        core_file,
    )
    _build_and_run(
        "tb_streaming",
        [
            RTL / "memory" / "tensorwright_stream_fifo.sv",
            RTL / "memory" / "tensorwright_activation_buffer.sv",
            RTL / "memory" / "tensorwright_weight_buffer.sv",
            RTL / "memory" / "tensorwright_window_generator_3x3.sv",
            VERIFICATION / "tb_streaming.sv",
        ],
    )
    _build_and_run(
        "tb_control",
        [
            RTL / "control" / "tensorwright_registers_pkg.sv",
            RTL / "control" / "tensorwright_control.sv",
            VERIFICATION / "tb_control.sv",
        ],
    )
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
        convolution_file,
        [f"+TRACE_FILE={rtl_transfer_file}"],
    )
    capture = RtlTraceCapture(
        enabled=True,
        run_id="verilator_convolution_case_0000",
        model_id="rtl_convolution_regression",
        source_operation_id="synthetic:conv_0",
        compiled_operation_id="compiled:op_0000",
        operation_name="conv_0",
        tensor_name="conv_0_output",
        shape=[1, 2, 3, 3],
        source_backend="tensorwright.verilator_rtl",
    )
    for transfer in read_transfer_log(rtl_transfer_file):
        capture.record(transfer)
    trace_path = capture.write(BUILD / "convolution_rtl_trace.jsonl")
    assert trace_path is not None
    reference_trace_path = _reference_convolution_trace(
        BUILD / "convolution_reference_trace.jsonl", first_convolution_expected
    )
    comparison = compare_trace_files(reference_trace_path, trace_path)
    if not comparison.matched:
        raise RuntimeError(f"RTL trace comparison failed: {comparison.to_json()}")
    comparison_path = BUILD / "convolution_comparison_report.json"
    comparison_path.write_text(comparison.to_json(), encoding="utf-8")
    print(
        f"RTL differential tests passed: postprocess={postprocess_count}, "
        f"arithmetic_core={core_count}, convolution_layers={convolution_count}, "
        f"aligned_trace_values={comparison.matched_values}, "
        f"comparison_report={comparison_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
