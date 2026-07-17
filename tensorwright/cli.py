"""Command-line entry point for TensorWright."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from tensorwright import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the TensorWright command-line parser."""
    parser = argparse.ArgumentParser(
        prog="tensorwright",
        description=(
            "Hardware-aware machine-learning compiler and FPGA inference platform."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TensorWright command-line interface."""
    build_parser().parse_args(argv)
    return 0
