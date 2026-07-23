# RTL arithmetic core

Milestone 5 implements synthesizable arithmetic units under `rtl/compute` and
`rtl/postprocess`. These are layer-engine building blocks, not an AXI-streaming
accelerator or `tensorwright_top.sv` integration.

## Numerical interface

| Value | RTL format |
| --- | --- |
| Activation and weight | Signed INT8 |
| Product | Signed INT16 |
| Adder-tree and channel accumulator | Signed INT32 |
| Bias | Signed INT32 |
| Requantization multiplier | Unsigned 31-bit integer |
| Requantization shift | Unsigned 7-bit integer, 0–127 |
| Requantization product | Signed 64-bit |
| Output | Signed INT8 |

Post-processing adds bias, multiplies, performs a rounded right shift with ties away
from zero, applies optional ReLU, and saturates to INT8. Bias and accumulation overflow
are contract violations: RTL exposes `overflow_o`, simulation assertions fail, and no
wrapped value is accepted as correct. The compiler rejects fixed-point shifts that do
not fit the 7-bit interface.

Sequential units use active-low synchronous reset and explicit `valid_i`/`valid_o`.
The integrated nine-lane core accepts one group of nine activation/weight pairs per
valid cycle. `clear_i` starts a new channel accumulation and `last_i` sends the complete
accumulator through registered operand, product, balanced-reduction, and post-processing
stages. Requantization separately registers bias addition, the 64-bit product, magnitude,
rounding, and activation/saturation. This pipeline meets the repository's 100 MHz
post-synthesis constraint for the Zynq-7020. Its additional latency is carried by `valid_o`;
callers must not assume a single-cycle response.

## Verification

`make lint-rtl` performs Verilator lint. `make test-rtl` generates expected values from
the Python integer reference and runs self-checking SystemVerilog simulations. It covers
all 65,536 signed INT8 products, directed MAC/adder/accumulator checks, 508 directed and
seeded-random post-processing cases, and 150 seeded-random multi-cycle nine-lane dot
products.

Cocotb equivalents are provided under `verification/cocotb` and run with
`make test-cocotb` in a supported Python environment. Cocotb 2.0.1 cannot run on the
current Python 3.14 environment, so those tests were not executed here; the same RTL was
executed through Verilator's self-checking SystemVerilog flow instead.
