# TensorWright Verify roadmap

Original Milestones 0–10 remain the completed technical foundation: repository and CLI,
integer reference, ONNX/IR, graph optimization, quantized software, RTL arithmetic,
streaming/buffering, register control, integrated convolution, `.twmodel` bundles, and
the command-driven simulation runtime.

| Milestone | Scope | Status |
| --- | --- | --- |
| 0–10 | Original compiler, numerical, RTL, bundle, and runtime foundation | Complete |
| 11 | Product migration and canonical trace specification | Complete |
| 12 | Python-reference and RTL trace capture | Complete |
| 13 | Semantic alignment and first-divergence detection | Complete |
| 14 | Deterministic numerical diagnosis rules | Complete |
| 15 | Streaming and protocol diagnosis | Next |
| 16 | Deterministic failing-input minimization | Planned |
| 17 | Cocotb regression-test generation | Planned |
| 18 | Debugging dashboard | Planned |
| 19 | Trace-adapter plugin API | Planned |
| 20 | Real tested FINN adapter | Future |
| 21 | Real tested hls4ml adapter | Future |

Milestone 12 adds optional operation-output traces from the Python reference and accepted
output-stream transfers from the custom RTL. The simulator-independent adapter is usable
from Cocotb; the Verilator regression emits the same compact samples and converts them to
canonical JSONL. Capture is intentionally limited to the stable convolution output boundary.
It does not implement alignment or diagnosis.

Milestone 13 aligns operation outputs with RTL stream transfers by stable primary source ID,
tensor, semantic trace point, and coordinate. It expands chunk payloads, rejects ambiguity,
and reports the first value or structural divergence. It does not assign a likely cause.

Milestone 14 adds a versioned, deterministic numerical ruleset for accumulator, bias,
activation, saturation, requantization, and output-only mismatches. Every classification
includes confidence, evidence, and recommended checks. Structural transfer failures are
explicitly left unclassified for Milestone 15.
