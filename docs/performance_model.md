# Metric classification and performance model

TensorWright separates observations, derived estimates, synthesis results, and metrics
that require physical hardware.

| Metric | Classification |
| --- | --- |
| RTL output correctness | Verified in simulation |
| Simulated cycle count | Measured from RTL simulation |
| Estimated latency | Derived from cycle count and stated clock |
| LUT/DSP/BRAM use | Parsed from Vivado synthesis and implementation |
| Timing slack | Parsed from Vivado implementation |
| Real DMA overhead | Requires board |
| Real ARM runtime overhead | Requires board |
| Actual FPGA speedup | Requires board |
| Physical power | Requires board |

Estimated accelerator latency is `simulated_cycle_count / stated_clock_frequency`. It
must always report both inputs and use the word “estimated.” Simulated utilization is
`active_compute_cycles / total_simulated_cycles`; it describes the simulated design,
not physical-board utilization or throughput.

The RTL counters distinguish total, compute-active, input-stall, output-stall,
weight-loading, produced-output, consumed-input, layer-invocation, executed-MAC, and
error counts. Their RTL behavior and register reads are implemented and tested.
End-to-end physical latency, DMA behavior,
power, and speedup remain unavailable without a board.
