"""Bootstrap an isolated environment and launch the TensorWright demo."""

from __future__ import annotations

import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / ".venv"


def main() -> int:
    if shutil.which("verilator") is None:
        print(
            "TensorWright demo requires Verilator. On Ubuntu/WSL, run:\n"
            "  sudo apt update && sudo apt install verilator",
            file=sys.stderr,
        )
        return 1

    python = ENVIRONMENT / "bin" / "python"
    if not python.is_file():
        print("Creating isolated demo environment in .venv...", flush=True)
        try:
            venv.EnvBuilder(with_pip=True).create(ENVIRONMENT)
        except Exception as error:
            print(
                "Could not create .venv. Install python3-venv and retry:\n"
                "  sudo apt install python3-venv\n"
                f"Details: {error}",
                file=sys.stderr,
            )
            return 1

    check = subprocess.run(
        [
            str(python),
            "-c",
            "import numpy, onnx, tensorwright",
        ],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if check.returncode != 0:
        print("Installing TensorWright demo dependencies...", flush=True)
        install = subprocess.run(
            [str(python), "-m", "pip", "install", "-e", str(ROOT)],
            cwd=ROOT,
            check=False,
        )
        if install.returncode != 0:
            print(
                "Dependency installation failed. Check the pip output above and retry.",
                file=sys.stderr,
            )
            return install.returncode

    return subprocess.run(
        [str(python), "-m", "scripts.run_demo", *sys.argv[1:]],
        cwd=ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
