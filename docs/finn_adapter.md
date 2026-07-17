# FINN trace adapter

Milestone 20 adds the maintained `finn.dataflow` adapter. It consumes the full
execution-context NPZ written by FINN verification when
`verify_save_full_context` is enabled. The repository integration script calls FINN's
own `execute_onnx(..., return_full_exec_context=True)`, saves that returned context,
converts it, and checks every resulting value.

## Conversion

FINN context keys identify tensors but do not contain durable source-operation
provenance. The adapter therefore requires an explicit `tensors` mapping. Only listed
tensors are traced; model inputs, parameters, and unrelated values are ignored.

```json
{
  "run_id": "verify-folded-model",
  "model_id": "finn-model",
  "graph_stage": "post_folding",
  "scalar_event_limit": 4096,
  "tensors": [
    {
      "tensor_name": "global_out",
      "source_operation_id": "onnx:Conv_3",
      "compiled_operation_id": "finn:MVAU_0",
      "fused_source_operation_ids": ["onnx:Relu_4"],
      "operation_name": "MVAU_0",
      "operation_type": "MVAU",
      "hardware_stage": "finn_operation_output",
      "layout": "NHWC"
    }
  ]
}
```

```bash
tensorwright trace convert verification_output/verify_step_SUCCESS.npz \
  finn-trace.jsonl --adapter finn.dataflow --options @finn-options.json
```

Small tensors become scalar events. Tensors exceeding `scalar_event_limit` become
`.npy` payloads referenced by a `tensor_chunk` event. FINN contexts contain values,
not simulator cycles or stream handshakes, so the adapter invents no timing metadata.

## Tested compatibility

The integration was run against FINN revision
`39f0c9a6b7675f62d47390fbf9a591707bcbac9b` and the QONNX revision
`fd61cfeebbdaba351abf7e9d54cd785d7776fa4f` pinned by that FINN checkout. With both
source trees and their dependencies on `PYTHONPATH`, run:

```bash
python scripts/verify_finn_adapter.py --output-dir build/finn-adapter-check
```

FINN returns the input, initializer, intermediate Add output, and final Relu output.
TensorWright selects both operation outputs and checks eight canonical events.
