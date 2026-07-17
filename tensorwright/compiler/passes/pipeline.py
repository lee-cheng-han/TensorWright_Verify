"""Default graph-optimization pipeline."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from tensorwright.compiler.ir import Graph
from tensorwright.compiler.passes.base import CompilerPass
from tensorwright.compiler.passes.batch_normalization import FoldBatchNormalization
from tensorwright.compiler.passes.canonicalize import CanonicalizeShapeOperations
from tensorwright.compiler.passes.constant_folding import FoldConstants
from tensorwright.compiler.passes.dead_code import EliminateDeadCode
from tensorwright.compiler.passes.fusion import FuseConvBiasRelu
from tensorwright.compiler.passes.partition import AssignBackends

logger = logging.getLogger(__name__)

DEFAULT_PIPELINE: tuple[CompilerPass, ...] = (
    FoldConstants(),
    FoldBatchNormalization(),
    FuseConvBiasRelu(),
    CanonicalizeShapeOperations(),
    EliminateDeadCode(),
    AssignBackends(),
)


def optimize_graph(
    graph: Graph, passes: Iterable[CompilerPass] = DEFAULT_PIPELINE
) -> Graph:
    """Run compiler passes in order and return the optimized graph."""
    result = graph
    for compiler_pass in passes:
        before_operations = len(result.operations)
        before_tensors = len(result.tensors)
        result = compiler_pass.run(result)
        logger.debug(
            "pass=%s operations=%d->%d tensors=%d->%d",
            compiler_pass.name,
            before_operations,
            len(result.operations),
            before_tensors,
            len(result.tensors),
        )
    return result
