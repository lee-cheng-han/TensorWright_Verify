# Canonical trace format version 2

Canonical traces use UTF-8 JSON Lines for metadata and events. Small tensors and focused
debugging regions may use scalar events. Large tensors use `tensor_chunk` events whose
payload is a relative, non-pickled `.npy` file. This avoids hundreds of megabytes of
repeated JSON keys while retaining coordinate-level events for minimized failures and
values near a divergence. The reference writer switches to chunks above 4,096 values by
default.

All events in one file share `run_id`, `source_backend`, and `model_id`. Required
provenance separates `source_operation_id`, `compiled_operation_id`,
`fused_source_operation_ids`, and `graph_stage`. ONNX source IDs are assigned during
import and preserved by compiler passes. Compiled IDs describe one particular optimized
graph and are not presented as stable across transformations or re-export.

Every event has an explicit `trace_point`: `operation_input`, `accumulator`, `post_bias`,
`post_requantization`, `post_activation`, `operation_output`, or `stream_transfer`.
Hardware stage remains a separate adapter-specific field.

Backend identifiers are extensible dotted names such as
`tensorwright.python_reference`, `tensorwright.cocotb_rtl`,
`tensorwright.verilator_rtl`, or `custom.my_adapter`.
Schema validation rejects malformed names, not unknown well-formed adapters. A later
adapter registry will determine execution and interpretation capabilities. FINN and
hls4ml names do not imply adapters exist.

Unsupported versions, non-finite values/scales, rank mismatches, out-of-range
coordinates/chunks, unsafe payload paths, negative cycles, empty traces, mixed run
identities, and missing payload files are rejected. Trace schema v2 replaces the
short-lived v1 development schema before any external compatibility guarantee.
