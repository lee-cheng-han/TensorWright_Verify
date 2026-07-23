# Simulation runtime contract

Milestone 10 implements a generic device runtime over the same contracts intended
for a future ARM runtime. It loads and validates `.twmodel` bundles, allocates modeled
memory regions, interpret compiler-generated commands, program registers, stream packed
weights and activations, enforce timeouts, read status/errors/counters, collect outputs,
and execute declared CPU fallback operations.

The M10 contract-model transport executes compiler commands against the bit-accurate
integer backend while modeling register transactions, scratch memory, stream transfers,
timeouts, and seeded backpressure. Its counters are modeled observations and are never
reported as RTL-simulator or hardware measurements. Revised Milestone 12 replaces that transport
with the Cocotb/RTL adapter without changing bundle or runtime orchestration. The native
Verilator runner now complements that model by executing supported compiler-emitted
convolution data directly on real RTL.

Cocotb is the preferred RTL integration for revised Milestone 12. Drivers must obey legal AXI ready/valid
behavior and support disabled, deterministic, and seeded-random backpressure. A failed
run reports the seed. Completion is accepted only after the final output transfer, and
timeouts and device error status return failure to the CLI.

The runtime may use simulator APIs for clocking and signal access, but it may not inspect
internal datapath state to control normal execution, bypass register programming, or
encode a particular model's layer sequence. Optional debug capture may read internal
arithmetic context solely to improve a mismatch report.

`tensorwright simulate MODEL.twmodel` emits a machine-readable
JSON report. `--seed`, `--timeout-cycles`, and `--no-backpressure` control deterministic
execution. `tensorwright compile`, `inspect-bundle`, and `benchmark` are also implemented.
Vivado synthesis and implementation remain repository workflows because they require
vendor tools rather than normal package dependencies.
