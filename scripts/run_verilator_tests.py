"""Generate Python-reference vectors and execute self-checking Verilator tests."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

from tensorwright.reference import requantize_int32

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "rtl_vectors"
RTL = ROOT / "rtl"
VERIFICATION = ROOT / "verification" / "systemverilog"


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


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


def _convolution_vectors(path: Path) -> int:
    random_source = random.Random(0xC07E)
    case_count = 20
    lines = [str(case_count)]
    for _ in range(case_count):
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return case_count


def _build_and_run(
    top: str, sources: list[Path], vector_file: Path | None = None
) -> None:
    build_dir = BUILD / top
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
            "--Mdir",
            str(build_dir),
            *[str(source) for source in sources],
        ]
    )
    run_command = [str(build_dir / f"V{top}")]
    if vector_file is not None:
        run_command.append(f"+VECTOR_FILE={vector_file}")
    _run(run_command)


def main() -> int:
    BUILD.mkdir(parents=True, exist_ok=True)
    postprocess_file = BUILD / "postprocess_vectors.txt"
    core_file = BUILD / "core_vectors.txt"
    convolution_file = BUILD / "convolution_vectors.txt"
    postprocess_count = _postprocess_vectors(postprocess_file)
    core_count = _core_vectors(core_file)
    convolution_count = _convolution_vectors(convolution_file)
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
    )
    print(
        f"RTL differential tests passed: postprocess={postprocess_count}, "
        f"arithmetic_core={core_count}, convolution_layers={convolution_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
