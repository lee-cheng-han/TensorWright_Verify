# Failure-preserving input minimization

Milestone 16 reduces named tensor inputs while preserving the exact original failure identity.
Inputs and outputs use non-pickled NPZ files:

```bash
tensorwright minimize failing.npz minimal.npz \
  --report minimal.report.json \
  --max-evaluations 1000 \
  --oracle-timeout 300 \
  --oracle python verify_failure.py
```

TensorWright appends the candidate NPZ path to the oracle command. The oracle exits successfully
and prints either `null` when the candidate passes or one JSON failure signature:

```json
{
  "kind": "value_mismatch",
  "source_operation_id": "onnx:Conv_0",
  "tensor_name": "conv_output",
  "trace_point": "operation_output",
  "coordinate": [0, 5, 7, 11],
  "rule_id": "requantization_rounding_mismatch"
}
```

The original input must reproduce a failure. Every accepted reduction must return the exact same
signature, preventing the minimizer from silently switching to an easier unrelated bug. Named
tensors are traversed in lexical order and flat coordinates in increasing order. Delta debugging
first removes chunks of nonzero values, then attempts to reduce retained magnitudes to `-1` or
`1`. The source arrays are never mutated.

The JSON report records tensor shapes and dtypes, evaluation count, original and minimized
nonzero counts, changed values, reduction fraction, signature, and whether the evaluation budget
stopped the search. A budget-limited result remains a valid reproducer but is not claimed to be
globally minimal. The algorithm seeks a locally irreducible zero-support result; it does not
claim mathematical minimality for arbitrary stateful or nondeterministic oracles.
