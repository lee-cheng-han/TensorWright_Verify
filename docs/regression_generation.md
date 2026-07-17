# Cocotb regression generation

Milestone 17 turns a minimized failure into a portable regression directory:

```bash
tensorwright generate-regression \
  minimal.npz \
  minimal.report.json \
  reference.jsonl \
  regressions/conv_rounding \
  --name conv_rounding
```

The destination must be new or empty. A package contains:

```text
conv_rounding/
├── README.md
├── manifest.json
├── inputs.npz
├── reference.jsonl
├── tensors/                 # present when the trace has chunk payloads
└── test_conv_rounding.py
```

`manifest.json` records format version 1, model identity, tensor shapes and dtypes, the preserved
failure signature, and SHA-256 plus byte size for every package file. Generation is deterministic
for identical source artifacts. Names are restricted to portable lowercase Python identifiers.

The generated test requires an adapter selected at runtime:

```bash
export TENSORWRIGHT_REGRESSION_ADAPTER=my_project.cocotb_adapter:run_case
```

The adapter receives `(dut, inputs, candidate_path)`. It may be synchronous or asynchronous and
must write a canonical candidate trace to `candidate_path`, or return another trace path. This
small hook isolates DUT-specific reset, clock, register, and stream driving without embedding
project assumptions in generated code.

The test compares the candidate against the packaged reference using TensorWright's semantic
comparison engine. It passes after the mismatch is fixed. While the original bug remains it fails
with the preserved signature and full comparison report. If the first divergence changes, it
fails separately with expected and observed identities, preventing silent regression drift.

TensorWright validates generated Python syntax during its own tests. Cocotb execution remains
dependent on a project adapter, simulator, DUT sources, and a Cocotb-supported Python version;
the package does not claim those external components were run during generation.
