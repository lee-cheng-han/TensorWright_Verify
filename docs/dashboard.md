# Debugging dashboard

Dashboard format version 2 renders one self-contained HTML investigation report directly
from canonical traces:

```bash
tensorwright dashboard reference.jsonl rtl.jsonl report.html
tensorwright dashboard reference.jsonl rtl.jsonl report.html \
  --minimization-report minimal.report.json \
  --regression-manifest regressions/conv_rounding/manifest.json
```

The dashboard contains:

- match/divergence status and reference, candidate, and matched value counts;
- first-divergence operation, tensor, trace point, coordinate, values, and candidate cycle;
- numerical rule, confidence, evidence, and recommended checks;
- protocol pass/fail state and every finding with event and cycle location;
- optional minimization statistics and generated-regression manifest;
- a complete machine-readable JSON report embedded for auditing.

Large tensors are never expanded into the HTML document. Chunk-to-chunk comparisons use
memory-mapped `.npy` payloads and bounded 65,536-value comparison blocks. The report shows
at most a 5×5 window around the first divergence, regardless of whether the source tensor
contains hundreds or millions of values. Source previews are capped at 12,000 characters.

The desktop layout uses sticky section navigation and progressive disclosure. Summary,
diagnostic lanes, arithmetic evidence, the bounded tensor window, and regression status stay
prominent; raw identifiers and machine-readable details remain collapsed until requested.

Reports are deterministic for identical artifacts, responsive, printable, and usable in light or
dark color schemes. They contain no external assets, network requests, or server-side components.
An optional inline clipboard action copies generated-regression commands; all diagnostic content
works without it. Trace, metadata, and source previews are HTML-escaped.

The command exits with status 0 for a matching comparison, 2 for a divergence, and 1 for invalid
inputs. A dashboard can still show a clean protocol result alongside a numerical divergence,
clearly separating arithmetic failures from transport and handshake failures.
