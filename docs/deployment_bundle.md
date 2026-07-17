# Deployment bundle format

TensorWright compilation produces a directory named `model_name.twmodel` with this layout:

```text
model_name.twmodel/
├── manifest.json
├── graph.json
├── commands.bin
├── weights.bin
├── biases.bin
├── quantization.bin
├── constants.bin
├── memory_plan.json
├── schedule.json
├── labels.txt
├── reference_input.bin
├── reference_output.bin
└── compilation_report.json
```

The bundle is a directory, not a single archive. `manifest.json` will identify the
format version, model, target, required hardware version, tensor interfaces, layer
count, and scratch-memory requirement. Schemas, alignment, byte ordering, checksums,
and binary encodings will be versioned before backend implementation; example values
must never be hardcoded into general compilation logic.
