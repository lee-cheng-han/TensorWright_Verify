# Convolution engine

Milestone 8 integrates the nine-lane arithmetic core into a parameterized, reusable
INT8 convolution engine. Image dimensions and channel counts are elaboration parameters;
each invocation performs valid, unpadded, stride-one 3x3 convolution for batch size one.
One input channel is accumulated per compute cycle, so every output performs exactly
`9 * INPUT_CHANNELS` MACs before bias, fixed-point requantization, optional ReLU, and
INT8 saturation.

The implementation is pipelined across operand capture, multiplication, balanced
nine-lane reduction, accumulation, bias, DSP multiplication, magnitude, rounding, and
saturation. `valid_o` carries the resulting latency. The current configuration meets a
100 MHz out-of-context routed constraint for `xc7z020clg400-1`.

After start, the engine accepts the complete weight packet followed by the complete
activation packet. It then emits outputs under ready/valid backpressure. Weight and
activation sources may insert arbitrary gaps, and the output remains stable during
arbitrary legal stalls. `tlast` is required only on the final beat of each packet.
Soft reset aborts any phase and returns the engine to idle.

`tensorwright_top.sv` connects engine transfers, compute activity, MAC count, completion,
errors, interrupts, and counters to the AXI4-Lite control plane. Its elaboration parameters
must match the programmed dimensions. The board-independent `.twmodel` runner currently
supports the synthesized `1x3x5x5 -> 1x2x3x3` configuration and reads invocation data
from bundle binaries.
