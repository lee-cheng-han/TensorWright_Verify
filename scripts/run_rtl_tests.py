"""Build and run Milestone 5 cocotb tests with Verilator."""

from __future__ import annotations

import os
from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl"
COCOTB = ROOT / "verification" / "cocotb"


def run_test(
    top: str,
    sources: list[Path],
    module: str,
    *,
    waves: bool = False,
) -> None:
    """Build one top and execute its cocotb module."""
    runner = get_runner("verilator")
    runner.build(
        sources=sources,
        hdl_toplevel=top,
        build_dir=ROOT / "build" / "rtl" / top,
        always=True,
        waves=waves,
        build_args=["--assert", "-Wall", "-Wno-fatal"],
    )
    runner.test(
        hdl_toplevel=top,
        test_module=module,
        test_dir=COCOTB,
        waves=waves,
    )


def main() -> int:
    """Run every arithmetic-core RTL test."""
    os.environ.setdefault("COCOTB_RANDOM_SEED", "51966")
    run_test(
        "tensorwright_multiplier",
        [RTL / "compute" / "tensorwright_multiplier.sv"],
        "test_multiplier",
    )
    run_test(
        "tensorwright_postprocess",
        [RTL / "postprocess" / "tensorwright_postprocess.sv"],
        "test_postprocess",
    )
    run_test(
        "tensorwright_arithmetic_core",
        [
            RTL / "compute" / "tensorwright_multiplier.sv",
            RTL / "compute" / "tensorwright_adder_tree.sv",
            RTL / "postprocess" / "tensorwright_postprocess.sv",
            RTL / "compute" / "tensorwright_arithmetic_core.sv",
        ],
        "test_arithmetic_core",
        waves=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
