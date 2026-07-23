# Performance evidence

TensorWright distinguishes three kinds of performance data:

- Runtime-model counters from `tensorwright simulate` and `tensorwright benchmark`.
- Cycle observations from real Verilator RTL.
- Timing, utilization, and vectorless power estimates from Vivado.

`make demo-bundle-rtl` compiles a fresh convolution to `.twmodel`, executes the bundle
data on Verilator, and writes `build/bundle_rtl_demo/report.json`. The report records the
accepted-output cycle span, outputs per cycle, and the corresponding 100 MHz output-phase
latency. This is real RTL simulation timing but excludes host file handling and board DMA.

`make synth` and `make implement` write machine-readable summaries under
`build/synthesis/` and `build/implementation/`. Routed timing at 100 MHz is the strongest
pre-board frequency evidence. Actual end-to-end latency and throughput remain unknown
until DMA, the processing system, memory contention, and software overhead are measured
on the board.

The current engine prioritizes correctness and diagnosability. It processes one output at
a time and therefore does not yet achieve one output per clock even though its internal
arithmetic is pipelined.
