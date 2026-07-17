# Synthesis methodology

Milestone 13 will run reproducible Vivado synthesis or implementation for the
Zynq-7020 part `xc7z020clg400-1`; no connected board is required. Scripts will declare
the target part, source list, top module, configuration parameters, clocks, and timing
constraints. Tool versions and warnings will be retained with each run.

Parsers will extract LUT, flip-flop, DSP, BRAM, worst setup slack, worst hold slack,
critical-path summary, clock target, warnings, and unconstrained paths into JSON. Values
remain `null` with `report_status: "not_run"` until actual reports are parsed. Checked-in
documentation must not contain manually transcribed utilization or timing claims.

Synthesis demonstrates mapping and timing feasibility for a target device. It does not
demonstrate physical execution, real board throughput, DMA latency, power, thermal
behavior, or timing under operating conditions.
