# Debugging dashboard

Milestone 18 renders one self-contained HTML investigation report directly from canonical traces:

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

Reports are deterministic for identical artifacts, responsive, printable, and usable in light or
dark color schemes. They contain no external assets, network requests, JavaScript, or server-side
components, making them straightforward CI artifacts. Trace and JSON content is HTML-escaped.

The command exits with status 0 for a matching comparison, 2 for a divergence, and 1 for invalid
inputs. A dashboard can still show a clean protocol result alongside a numerical divergence,
preserving the separation established by Milestones 14 and 15.
