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

The bundle is a directory, not a single archive. It is the runtime-neutral boundary
between compilation and both simulated and future board execution. `manifest.json`
will identify the format version, model, target, required hardware-interface version,
command-format version, tensor interfaces, layer count, and scratch-memory requirement.

The simulation runtime must consume `commands.bin` and packed tensor files rather than
reconstructing work from model-specific testbench knowledge. A future ARM runtime must
accept the same validated content. Simulator seeds, logs, waveforms, and synthesis
reports are run artifacts and do not alter bundle semantics.

Schemas, command encoding, alignment, byte ordering, checksums, error behavior, and
version compatibility remain drafts until Milestone 9. Example values must never be
hardcoded into general compilation logic.
