# RTL trace capture

Milestone 12 establishes the first executable hardware trace boundary: accepted output
transfers from the TensorWright convolution engine. A sample becomes an event only when
both `valid` and `ready` are asserted. Each canonical event records the simulator cycle,
flat transfer sequence, reconstructed NCHW coordinate, and TLAST state.

Capture is disabled unless explicitly requested. The self-checking SystemVerilog regression
accepts `+TRACE_FILE=<path>` and writes a compact transfer log for its first deterministic
convolution case. `scripts/run_verilator_tests.py` converts that log to
`build/rtl_vectors/convolution_rtl_trace.jsonl` with source backend
`tensorwright.verilator_rtl`. The adapter itself has no Cocotb import and can therefore be used
by a Cocotb monitor or by the maintained Verilator fallback.

Run the complete RTL regression with:

```bash
make test-rtl
tensorwright trace inspect build/rtl_vectors/convolution_rtl_trace.jsonl
```

The stable Milestone 12 trace point is `stream_transfer` at the final convolution output.
Accumulator and post-processing trace points remain in the schema for later instrumentation.
This milestone deliberately does not align reference and RTL events or infer a cause; those
features begin in Milestone 13.
