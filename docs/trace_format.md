# Canonical trace format version 1

Canonical traces are UTF-8 JSON Lines: one independently valid event object per line.
All events in one file share `run_id`, `source_backend`, and `model_id`.

Required event fields are trace version, run/model/backend identity, stable operation ID
and name/type, hardware stage, tensor name, coordinate and shape, layout, dtype, and
numeric value. Cycle and quantization metadata are optional. Arbitrary adapter metadata
is preserved in the `metadata` object.

Version 1 source identifiers implemented by the schema are `python_reference`,
`cocotb_rtl`, and `custom_rtl`; only `python_reference` has a writer in Milestone 11.
Unsupported versions and backends, non-finite values/scales, rank mismatches,
out-of-range coordinates, negative cycles, empty traces, and mixed run identities are
rejected.

Reference operation IDs use `op_NNNN:<operation-name>`, combining deterministic graph
position with the existing operation name. This avoids model-specific names while
remaining stable for identical optimized graphs. Milestone 13 may consume explicit
bundle alignment metadata where available; it will not silently guess ambiguous maps.
