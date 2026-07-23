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
between compilation, RTL simulation, and future board execution. `manifest.json`
identifies the format version, model, target, required hardware-interface version,
command-format version, tensor interfaces, layer count, and scratch-memory requirement.

The simulation runtime consumes `commands.bin` and packed tensor files rather than
reconstructing work from model-specific testbench knowledge. The fixed-shape Verilator
runner also decodes FPGA convolution data directly from the bundle. A future ARM runtime must
accept the same validated content. Simulator seeds, logs, waveforms, and synthesis
reports are run artifacts and do not alter bundle semantics.

Milestone 9 freezes bundle format version 1, command version 1, and required hardware
interface version 1.0. Every manifest-listed file has an exact byte size and SHA-256
digest. Loading fails before execution on a missing file, checksum mismatch, unsupported
version, malformed JSON, partial command record, misaligned allocation, or inconsistent
layer count.

All binary integers and floats are little-endian. Scratch allocations begin at 64-byte
boundaries. Tensor elements are contiguous in the stream orders documented by
`tensor_layout.md`. `weights.bin` stores signed INT8 constants, `biases.bin` stores
signed INT32 values, and `constants.bin` stores remaining constants. Each per-channel
quantization record is eight bytes: unsigned 31-bit multiplier in a 32-bit word, shift
in one byte, and three reserved zero bytes.

## Commands and schedules

`commands.bin` contains fixed 32-byte records of eight unsigned 32-bit words:

```text
opcode, backend, input_offset, output_offset,
weight_offset, bias_offset, quantization_offset, flags_and_layer_index
```

Opcodes 1–5 are Conv, MaxPool, View, Gemm, and Softmax. Backends 1–3 are FPGA, ARM,
and metadata. Bit zero of the flags word denotes fused ReLU and bits 31:16 contain the
schedule index. File-relative constant offsets and scratch-memory offsets are never
host pointers.

`memory_plan.json` records every nonconstant tensor allocation, size, alignment, and
total scratch requirement. `schedule.json` retains graph order and records backend,
tensor names, offsets, and deterministic compute-cycle and transfer-byte estimates.
These are compiler estimates, not simulator measurements.

Reference inputs are stored as little-endian FP32 values in graph-input order so the
runtime can reproduce input quantization. Reference outputs use FP32 for floating CPU
outputs and signed INT8 for quantized outputs. `labels.txt` is UTF-8 with one label per
line and may be empty.

`make demo-bundle-rtl` compiles a fresh ONNX convolution, validates its bundle, decodes
the binary records, runs the native RTL, and requires all 18 output values to match.
