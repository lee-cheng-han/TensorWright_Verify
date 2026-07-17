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
| 15 | Streaming and protocol diagnosis | Complete |
| 16 | Deterministic failing-input minimization | Complete |
| 17 | Cocotb regression-test generation | Complete |
| 18 | Debugging dashboard | Complete |
| 19 | Trace-adapter plugin API | Complete |
| 20 | Real tested FINN adapter | Complete |
| 21 | Real tested hls4ml adapter | Complete |

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

Milestone 15 adds a separate versioned protocol ruleset for accepted-handshake metadata,
transfer sequences, simulator-cycle ordering, TLAST placement, and missing or unexpected
outputs. Numerical mismatches do not become protocol failures. Minimization begins in
Milestone 16.

Milestone 16 adds deterministic delta debugging for named tensor inputs. Reductions are
accepted only when an oracle returns the exact original failure signature; the minimizer
then simplifies retained magnitudes, honors an evaluation budget, and writes replayable NPZ
inputs plus an auditable JSON report. Regression generation begins in Milestone 17.

Milestone 17 packages minimized NPZ inputs, canonical reference traces and payloads, the
preserved failure identity, a checksummed manifest, and a portable Cocotb test. A small
`module:function` adapter hook drives project-specific DUT interfaces while the generated
test reuses TensorWright semantic comparison. Dashboard work begins in Milestone 18.

Milestone 18 adds deterministic, self-contained HTML debugging dashboards combining trace
counts, the first divergence, numerical diagnosis, protocol findings, complete embedded JSON,
and optional minimization and regression metadata. Reports require no server or JavaScript and
are suitable for local inspection or CI artifacts. The adapter plugin API begins in Milestone 19.

Milestone 19 adds trace-adapter API version 1, validated descriptors, isolated registries,
opt-in Python entry-point discovery, canonical-output enforcement, and CLI listing/conversion.
The maintained Verilator transfer-log converter is the first built-in implementation. External
backends remain unsupported until their separately tested adapter milestones are completed.

Milestone 20 adds the maintained `finn.dataflow` converter for FINN full execution-context
NPZ artifacts. Explicit tensor mappings preserve source and compiled operation provenance;
small tensors use scalar events and larger tensors use external NumPy payloads. A repository
integration script invokes FINN's real full-context executor and verifies converted values.
The adapter does not infer cycles or streaming handshakes absent from FINN's context. The
real tested hls4ml adapter begins in Milestone 21.

Milestone 21 adds the maintained `hls4ml.csim` converter for layer arrays returned by
hls4ml's real C-simulation trace API. Explicit mappings retain source and optimized-graph
operation identity, and the shared NPZ conversion path provides scalar and chunked events.
The integration script compiles and traces a real hls4ml Dense/Activation graph and verifies
the resulting canonical values. It accurately identifies C simulation and does not claim
RTL timing evidence. This completes the currently defined TensorWright Verify roadmap.
