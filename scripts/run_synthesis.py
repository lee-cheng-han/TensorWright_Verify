"""Run reproducible board-independent Vivado synthesis for the Zybo Z7-20."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "synthesis"


def main() -> int:
    vivado = shutil.which("vivado")
    if vivado is None:
        print("Vivado is not available on PATH; synthesis was not run.")
        return 1
    BUILD.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed = subprocess.run(
        [
            vivado,
            "-mode",
            "batch",
            "-nojournal",
            "-nolog",
            "-source",
            str(ROOT / "scripts" / "synthesize_z7.tcl"),
        ],
        cwd=ROOT,
        check=False,
    )
    elapsed = time.monotonic() - started
    status_file = BUILD / "synthesis_status.txt"
    status = _read_status(status_file) if status_file.is_file() else {}
    summary = {
        "status": "complete" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "tool": vivado,
        "part": status.get("part", "xc7z020clg400-1"),
        "top": status.get("top", "tensorwright_top"),
        "source_count": int(status.get("source_count", 0)),
        "clock_period_ns": float(status.get("clock_period_ns", 10.0)),
        "artifacts": {
            name: str((BUILD / name).relative_to(ROOT))
            for name in (
                "tensorwright_synth.dcp",
                "utilization.rpt",
                "timing_summary.rpt",
                "clock_utilization.rpt",
            )
            if (BUILD / name).is_file()
        },
    }
    utilization = _read_utilization(BUILD / "utilization.rpt")
    if utilization:
        summary["utilization"] = utilization
    timing = _read_timing(BUILD / "timing_summary.rpt")
    if timing:
        summary["timing"] = timing
    (BUILD / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return completed.returncode


def _read_status(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )


def _read_utilization(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    match = re.search(
        r"^\|\s*tensorwright_top\s*\|\s*\(top\)\s*\|\s*(\d+)\s*\|"
        r"\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
        r"\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|",
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match is None:
        return {}
    names = (
        "luts",
        "logic_luts",
        "lutram",
        "srl",
        "flip_flops",
        "ramb36",
        "ramb18",
        "dsp",
    )
    return dict(zip(names, (int(value) for value in match.groups()), strict=True))


def _read_timing(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    document = path.read_text(encoding="utf-8")
    section = document.split("| Design Timing Summary", 1)
    if len(section) != 2:
        return {}
    match = re.search(
        r"^\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\d+\s+\d+",
        section[1],
        re.MULTILINE,
    )
    if match is None:
        return {}
    return {"wns_ns": float(match.group(1)), "tns_ns": float(match.group(2))}


if __name__ == "__main__":
    raise SystemExit(main())
