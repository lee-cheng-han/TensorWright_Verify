"""Deterministic graph-optimization passes."""

from tensorwright.compiler.passes.base import CompilerPass
from tensorwright.compiler.passes.batch_normalization import FoldBatchNormalization
from tensorwright.compiler.passes.canonicalize import CanonicalizeShapeOperations
from tensorwright.compiler.passes.constant_folding import FoldConstants
from tensorwright.compiler.passes.dead_code import EliminateDeadCode
from tensorwright.compiler.passes.fusion import FuseConvBiasRelu
from tensorwright.compiler.passes.partition import AssignBackends
from tensorwright.compiler.passes.pipeline import DEFAULT_PIPELINE, optimize_graph

__all__ = [
    "AssignBackends",
    "CanonicalizeShapeOperations",
    "CompilerPass",
    "DEFAULT_PIPELINE",
    "EliminateDeadCode",
    "FoldBatchNormalization",
    "FoldConstants",
    "FuseConvBiasRelu",
    "optimize_graph",
]
