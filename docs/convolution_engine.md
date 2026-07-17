# Convolution engine

Milestone 8 integrates the nine-lane arithmetic core into a parameterized, reusable
INT8 convolution engine. Image dimensions and channel counts are elaboration parameters;
each invocation performs valid, unpadded, stride-one 3x3 convolution for batch size one.
One input channel is accumulated per compute cycle, so every output performs exactly
`9 * INPUT_CHANNELS` MACs before bias, fixed-point requantization, optional ReLU, and
INT8 saturation.

After start, the engine accepts the complete weight packet followed by the complete
activation packet. It then emits outputs under ready/valid backpressure. Weight and
activation sources may insert arbitrary gaps, and the output remains stable during
arbitrary legal stalls. `tlast` is required only on the final beat of each packet.
Soft reset aborts any phase and returns the engine to idle.

`tensorwright_top.sv` connects the engine transfers, compute activity, MAC count, and
completion to the Milestone 7 AXI4-Lite control plane. Its elaboration parameters must
match the dimensions programmed into the register file. Quantization metadata remains
an explicit port contract until Milestone 9 freezes its deployment-bundle encoding.
