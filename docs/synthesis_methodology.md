# Synthesis and implementation methodology

TensorWright provides reproducible Vivado flows for `xc7z020clg400-1`:

```bash
make synth
make implement
```

Both flows declare the source list, `tensorwright_top` module, target part, and 10 ns
clock constraint. Synthesis writes a netlist checkpoint and parsed utilization/timing
summary. Implementation synthesizes the accelerator out of context, then runs logical
optimization, placement, physical optimization, routing, DRC, timing, utilization, and
vectorless power analysis.

Machine-readable summaries are written to `build/synthesis/summary.json` and
`build/implementation/summary.json`. The routed design is fully routed and meets 100 MHz.
See `fpga_implementation.md` for current results and interpretation.

Out-of-context implementation demonstrates physical feasibility of the accelerator IP.
It does not include the processing system, DMA, board clocks/resets, software, package
constraints, or a deployable bitstream. Vectorless power is an estimate, not a measurement.
