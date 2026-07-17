# Simulation runtime contract

Milestone 10 implements a generic device runtime over the same contracts intended
for a future ARM runtime. It will load and validate `.twmodel` bundles, allocate modeled
memory regions, interpret compiler-generated commands, program registers, stream packed
weights and activations, enforce timeouts, read status/errors/counters, collect outputs,
and execute declared CPU fallback operations.

The M10 contract-model transport executes compiler commands against the bit-accurate
integer backend while modeling register transactions, scratch memory, stream transfers,
timeouts, and seeded backpressure. Its counters are modeled observations and are never
reported as RTL-simulator or hardware measurements. Revised Milestone 12 replaces that transport
with the Cocotb/RTL adapter without changing bundle or runtime orchestration.

Cocotb is the preferred RTL integration for revised Milestone 12. Drivers must obey legal AXI ready/valid
behavior and support disabled, deterministic, and seeded-random backpressure. A failed
run reports the seed. Completion is accepted only after the final output transfer, and
timeouts and device error status return failure to the CLI.

The runtime may use simulator APIs for clocking and signal access, but it may not inspect
internal datapath state to control normal execution, bypass register programming, or
encode a particular model's layer sequence. Optional debug capture may read internal
arithmetic context solely to improve a mismatch report.

`tensorwright simulate MODEL.twmodel` is functional in M10 and emits a machine-readable
JSON report. `--seed`, `--timeout-cycles`, and `--no-backpressure` control deterministic
execution. Compile, inspect, verify, and synthesize remain future CLI interfaces and are
not registered as placeholders.
