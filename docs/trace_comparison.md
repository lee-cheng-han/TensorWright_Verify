# Semantic trace comparison

Milestone 13 compares a canonical software-reference trace with an RTL or HLS candidate:

```bash
tensorwright trace compare reference.jsonl rtl.jsonl
tensorwright trace compare reference.jsonl rtl.jsonl --json
tensorwright trace compare reference.jsonl rtl.jsonl --report report.json
```

A match exits with status 0, an observed divergence exits with status 2, and an invalid or
unalignable input exits with status 1. JSON reports include backend and model identity,
value counts, the number of values matched before failure, and the first divergence.

Alignment is semantic rather than line-based. Values are keyed by the preserved primary
source-operation ID, tensor name, trace point, and tensor coordinate. `stream_transfer` is normalized to
`operation_output`, allowing the current RTL output stream to align with its Python reference.
Scalar events and `.npy` tensor chunks share the same comparison path. Shape, layout, dtype,
and operation type must agree.

The engine rejects different model IDs, malformed payload shapes, and duplicate semantic
keys instead of guessing. It distinguishes value mismatches, missing candidate values,
unexpected candidate values, and metadata mismatches. Candidate cycle information is retained
in the divergence report.

The Verilator regression writes a matching reference/RTL trace pair and
`build/rtl_vectors/convolution_comparison_report.json`, proving the maintained end-to-end path.
Milestone 13 deliberately reports facts only; deterministic likely-cause rules belong to
Milestone 14.
