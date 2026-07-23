"""Run board-independent out-of-context placement and routing."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from scripts.run_synthesis import _read_status, _read_timing, _read_utilization

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "implementation"


def main() -> int:
    vivado = shutil.which("vivado")
    if vivado is None:
        print("Vivado is not available on PATH; implementation was not run.")
        return 1
    BUILD.mkdir(parents=True, exist_ok=True)
    reports_only = "--reports-only" in sys.argv[1:]
    if reports_only:
        return_code = 0
        elapsed = 0.0
    else:
        started = time.monotonic()
        completed = subprocess.run(
            [
                vivado,
                "-mode",
                "batch",
                "-nojournal",
                "-nolog",
                "-source",
                str(ROOT / "scripts" / "implement_z7.tcl"),
            ],
            cwd=BUILD,
            check=False,
        )
        return_code = completed.returncode
        elapsed = time.monotonic() - started
    status_path = BUILD / "implementation_status.txt"
    status = _read_status(status_path) if status_path.is_file() else {}
    summary: dict[str, object] = {
        "status": "complete" if return_code == 0 else "failed",
        "return_code": return_code,
        "elapsed_seconds": round(elapsed, 3),
        "tool": vivado,
        "part": status.get("part", "xc7z020clg400-1"),
        "top": status.get("top", "tensorwright_top"),
        "source_count": int(status.get("source_count", 0)),
        "clock_period_ns": float(status.get("clock_period_ns", 10.0)),
        "implementation_mode": "out_of_context",
        "route_status": _route_status(BUILD / "route_status.rpt"),
        "artifacts": {
            name: str((BUILD / name).relative_to(ROOT))
            for name in (
                "tensorwright_routed.dcp",
                "utilization.rpt",
                "timing_summary.rpt",
                "route_status.rpt",
                "drc.rpt",
                "power.rpt",
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
    power = _power(BUILD / "power.rpt")
    if power:
        summary["estimated_power_w"] = power
    summary["drc_warnings"] = _drc_warnings(BUILD / "drc.rpt")
    (BUILD / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return return_code


def _route_status(path: Path) -> dict[str, int | bool]:
    if not path.is_file():
        return {"fully_routed": False}
    document = path.read_text(encoding="utf-8")

    def count(label: str) -> int:
        match = re.search(rf"{re.escape(label)}\.*\s*:\s*(\d+)", document)
        return int(match.group(1)) if match else 0

    routable = count("# of routable nets")
    routed = count("# of fully routed nets")
    errors = count("# of nets with routing errors")
    return {
        "fully_routed": routable > 0 and routed == routable and errors == 0,
        "routable_nets": routable,
        "fully_routed_nets": routed,
        "nets_with_routing_errors": errors,
    }


def _power(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    document = path.read_text(encoding="utf-8")
    values: dict[str, float] = {}
    for key, label in (
        ("total", "Total On-Chip Power (W)"),
        ("dynamic", "Dynamic (W)"),
        ("static", "Device Static (W)"),
    ):
        match = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*(\d+\.\d+)", document)
        if match:
            values[key] = float(match.group(1))
    return values


def _drc_warnings(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return sorted(
        set(
            re.findall(
                r"^\|\s*([A-Z][A-Z0-9-]+)\s*\|\s*Warning\s*\|",
                path.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
