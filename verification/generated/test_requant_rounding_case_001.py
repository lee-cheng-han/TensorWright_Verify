"""Regression for negative round-to-nearest requantization behavior."""

from __future__ import annotations

import subprocess
import unittest

from scripts.run_verilator_tests import ROOT, RTL, VERIFICATION, _build_and_run

BUILD = ROOT / "build" / "generated_regressions" / "requant_rounding_case_001"


def run_case(*, faulty: bool = False, quiet: bool = True) -> bool:
    """Return whether the RTL produces the bit-accurate expected value."""
    BUILD.mkdir(parents=True, exist_ok=True)
    vector = BUILD / "vector.txt"
    vector.write_text("24 -491 1 2 0 -117\n", encoding="utf-8")
    defines = ["-DTENSORWRIGHT_DEMO_FAULT_REQUANT_ROUND"] if faulty else []
    try:
        _build_and_run(
            "tb_postprocess",
            [
                RTL / "postprocess" / "tensorwright_postprocess.sv",
                VERIFICATION / "tb_postprocess.sv",
            ],
            vector,
            build_name="faulty" if faulty else "corrected",
            build_root=BUILD,
            quiet=quiet,
            verilator_args=defines,
        )
    except subprocess.CalledProcessError:
        return False
    return True


class RequantRoundingRegression(unittest.TestCase):
    def test_negative_halfway_rounds_away_from_zero(self) -> None:
        self.assertTrue(run_case(), "corrected RTL must produce -117")


if __name__ == "__main__":
    unittest.main()
