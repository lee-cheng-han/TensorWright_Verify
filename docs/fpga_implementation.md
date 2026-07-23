# FPGA implementation

TensorWright targets the Zynq-7020 device used by the Zybo Z7-20. Two vendor-tool flows
are available without a physical board:

```bash
make synth
make implement
```

`make synth` builds the normal accelerator top and reports post-synthesis timing and
utilization. `make implement` treats `tensorwright_top` as an out-of-context accelerator
IP, then optimizes, places, physically optimizes, and routes it at 100 MHz. Out-of-context
mode is intentional: the accelerator ports will ultimately connect to the Zynq processing
system, DMA, reset, and interrupt fabric rather than directly to package pins.

The current routed result for `xc7z020clg400-1` is fully routed with zero routing errors:

| Metric | Result |
| --- | ---: |
| Clock constraint | 10.000 ns / 100 MHz |
| Routed WNS | +1.302 ns |
| Routed TNS | 0.000 ns |
| LUTs | 2,555 |
| Flip-flops | 1,623 |
| DSP blocks | 4 |
| LUTRAMs | 324 |
| Vectorless estimated power | 0.137 W |

The power value is an implementation estimate with medium confidence and no simulation
activity file. It is not a board power measurement. DRC warnings `DPOP-1`, `DPOP-2`,
`RTSTAT-10`, and `ZPS7-1` describe optional DSP pipelining or expected out-of-context
top-level conditions; the implementation has no DRC errors.

The routed checkpoint is not a deployable bitstream. Bitstream generation requires the
board-level processing-system block design, DMA/interconnect choices, clocks, resets,
interrupt wiring, address map, and board constraints.
