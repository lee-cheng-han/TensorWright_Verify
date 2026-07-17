# Simulation runtime contract

Milestone 10 will implement a generic device runtime over the same contracts intended
for a future ARM runtime. It will load and validate `.twmodel` bundles, allocate modeled
memory regions, interpret compiler-generated commands, program registers, stream packed
weights and activations, enforce timeouts, read status/errors/counters, collect outputs,
and execute declared CPU fallback operations.

Cocotb is the preferred simulator integration. Drivers must obey legal AXI ready/valid
behavior and support disabled, deterministic, and seeded-random backpressure. A failed
run reports the seed. Completion is accepted only after the final output transfer, and
timeouts and device error status return failure to the CLI.

The runtime may use simulator APIs for clocking and signal access, but it may not inspect
internal datapath state to control normal execution, bypass register programming, or
encode a particular model's layer sequence. Optional debug capture may read internal
arithmetic context solely to improve a mismatch report.

Planned commands are `tensorwright compile`, `inspect`, `simulate`, `verify`, and
`synthesize`. They are roadmap interfaces, not registered CLI commands today; adding
nonfunctional placeholders would misrepresent current capability.
