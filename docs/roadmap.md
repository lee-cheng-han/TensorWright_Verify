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
| 13 | Semantic alignment and first-divergence detection | Next |
| 14 | Deterministic numerical diagnosis rules | Planned |
| 15 | Streaming and protocol diagnosis | Planned |
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
