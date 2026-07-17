"""Compiler-pass protocol."""

from __future__ import annotations

from typing import Protocol

from tensorwright.compiler.ir import Graph


class CompilerPass(Protocol):
    """One side-effect-free graph transformation."""

    name: str

    def run(self, graph: Graph) -> Graph:
        """Return a transformed copy of the input graph."""
        ...
