# hls4ml trace adapter

Milestone 21 adds the maintained `hls4ml.csim` adapter. hls4ml's
`ModelGraph.trace(x)` method compiles tracing support and returns a dictionary mapping
configured layer names to NumPy arrays. Save that dictionary without object arrays:

```python
prediction, trace = hls_model.trace(inputs)
np.savez("hls4ml_trace.npz", **trace)
```

Layers must have `Trace: true` in the hls4ml configuration. The returned names identify
hls4ml graph layers, but they do not provide durable source-model provenance. TensorWright
therefore requires an explicit mapping for every selected array:

```json
{
  "run_id": "hls4ml-csim-1",
  "model_id": "quantized-network",
  "graph_stage": "hls4ml_optimized_graph",
  "scalar_event_limit": 4096,
  "tensors": [
    {
      "tensor_name": "dense",
      "source_operation_id": "keras:dense_1",
      "compiled_operation_id": "hls4ml:dense",
      "operation_name": "dense",
      "operation_type": "Dense",
      "hardware_stage": "hls4ml_layer_output",
      "layout": "NC"
    }
  ]
}
```

```bash
tensorwright trace convert hls4ml_trace.npz hls4ml-trace.jsonl \
  --adapter hls4ml.csim --options @hls4ml-options.json
```

Small arrays become scalar events; large arrays use external `.npy` payloads. C
simulation layer traces contain neither RTL cycles nor stream handshakes, so the adapter
does not manufacture timing information. This boundary prevents C simulation from being
misrepresented as an RTL trace.

## Tested compatibility

The integration was run against hls4ml revision
`046e3c4697bf41c03f81468e8352867e28eb2faf`. It constructs a real hls4ml
`ModelGraph`, compiles a Vivado-backend C simulation with tracing enabled, calls
`ModelGraph.trace`, exports its returned arrays, converts them, and checks all values:

```bash
python scripts/verify_hls4ml_adapter.py \
  --output-dir build/hls4ml-adapter-check
```

The test requires hls4ml, a C++ compiler, and the normal hls4ml C-simulation headers. It
does not require Vivado synthesis.
