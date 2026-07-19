# Hackathon video demo

The TensorWright demo runs a bit-accurate Python reference against the real
SystemVerilog convolution engine compiled by Verilator. The exact weights, biases, and input
are also written as `tiny_conv.onnx` plus `reference_input.npy` and imported through
TensorWright's ONNX frontend before the RTL run. It first proves that the normal
RTL matches, then runs two explicitly demo-only defective builds. The numerical build
removes the requantizer's round-to-nearest bias, while the protocol build consumes one
logical output without presenting a valid transfer. TensorWright diagnoses both failures
and creates offline dashboards.

## Run it

Install Verilator, then run:

```bash
make demo
```

Focused launch targets keep the same automatic verification but end on the report needed for
that part of a presentation:

```bash
make demo-clean
make demo-numerical-fault
make demo-protocol-fault
```

`make demo` creates an ignored local `.venv` and installs TensorWright's Python
dependencies automatically on the first run. Subsequent runs reuse that environment. To
control the presentation pace, pass arguments through Make:

```bash
make demo DEMO_ARGS="--pace 0"
```

Run the command once before recording so dependency installation is complete. The recorded
run will then begin immediately with the TensorWright title card.

For an immediate CI-style run without presentation pauses:

```bash
python -m scripts.run_demo --pace 0
```

Artifacts are written under `build/demo/`:

- `reference.jsonl`: bit-accurate software values
- `tiny_conv.onnx`: the imported ONNX workload executed by the fixed convolution mapping
- `reference_input.npy`: the exact input tensor supplied to the deterministic vector run
- `clean_rtl.jsonl`: known-good accepted RTL transfers
- `rounding_fault_rtl.jsonl`: transfers from the truncating requantizer
- `protocol_fault_rtl.jsonl`: transfers with logical output #5 missing
- `comparison.json` and `diagnosis.json`: machine-readable results
- `summary.json`: complete demo summary
- `index.html`: numerical-failure visual report
- `protocol.html`: dropped-transfer visual report
- `rounding_fault_arithmetic.txt`: real accumulator-to-output values sampled inside RTL
- `reference_pipeline.jsonl` and `rounding_fault_pipeline.jsonl`: canonical internal-stage traces
- `pipeline_comparison.json`: standard semantic comparison localizing the first unequal
  `post_requantization` value
- `generated_regression/`: executable Verilator regression generated from that divergence
- `presentation.html`: single-screen video view of the failure, proof, fix, and regression
- `tensorwright-demo-report.zip`: portable dashboards, traces, evidence, model, and regression
- `verification/generated/test_requant_rounding_case_001.py`: permanent RTL regression

Open the report with your normal browser. For example, on Linux:

```bash
xdg-open build/demo/index.html
```

## Suggested 75-second recording

1. **0–10 seconds — problem.** Show the title: “One arithmetic bug can corrupt thousands
   of downstream values. Which value failed first?”
2. **10–25 seconds — real system.** Briefly show the Python reference and SystemVerilog
   convolution engine side by side. State that this is real Verilator execution, not a
   prerecorded trace.
3. **25–45 seconds — terminal.** Run `make demo`. Let the clean 18/18 match establish trust,
   then show the real truncation defect followed by the dropped-transfer defect.
4. **45–60 seconds — result.** Pause on the first coordinate, software/RTL values, accepted
   cycle, exact accumulator-to-shift arithmetic sampled from RTL, and confirmed rounding
   mechanism. Show the
   generated regression failing before the fix and passing afterward.
5. **60–75 seconds — dashboards.** Reveal `build/demo/presentation.html`, then switch briefly to
   the detailed `build/demo/index.html` and
   `build/demo/protocol.html` to contrast numerical and protocol diagnosis. End with:
   “TensorWright finds the first hardware/software mismatch—not the ten thousand errors
   that follow it.”

The injected macros are `TENSORWRIGHT_DEMO_FAULT_REQUANT_ROUND` and
`TENSORWRIGHT_DEMO_FAULT_DROPPED_TRANSFER`. Both are absent from production and normal
regression builds and are visibly documented in the terminal. The demo therefore remains
honest while exercising genuine RTL arithmetic, handshakes, tracing, diagnosis, and reports.
