# Deterministic numerical diagnosis

Milestone 14 adds a versioned numerical ruleset on top of first-divergence reports:

```bash
tensorwright trace diagnose reference.jsonl rtl.jsonl
tensorwright trace diagnose reference.jsonl rtl.jsonl --json
tensorwright trace diagnose reference.jsonl rtl.jsonl --report diagnosis.json
```

The command uses the same exit contract as trace comparison: 0 for a match, 2 for a diagnosed
divergence, and 1 for invalid or unalignable traces. Reports contain the original comparison,
ruleset version, stable rule ID, title, confidence, direct evidence, and recommended checks.

Rules are evaluated in a fixed order:

| Evidence at first divergence | Classification |
| --- | --- |
| `accumulator` unequal | accumulator arithmetic mismatch |
| `post_bias` unequal | bias application mismatch |
| `post_activation` unequal | activation mismatch |
| exactly one value at -128 or 127 | saturation boundary mismatch |
| requantized/output values differ by one | likely rounding mismatch |
| other `post_requantization` mismatch | likely parameter mismatch |
| output-only mismatch | insufficient internal localization |

These are deterministic hypotheses, not statistical predictions. Confidence describes how
directly the trace point supports the classification. Output-only mismatches intentionally get
low confidence and a request for earlier trace points. Missing, extra, or structurally
incompatible values produce `insufficient_numerical_evidence`; streaming and protocol causes
belong to Milestone 15.
