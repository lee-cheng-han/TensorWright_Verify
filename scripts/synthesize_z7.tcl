set root [file normalize [file join [file dirname [info script]] ..]]
set build [file join $root build synthesis]
file mkdir $build

set sources [concat \
    [glob -nocomplain [file join $root rtl compute *.sv]] \
    [glob -nocomplain [file join $root rtl postprocess *.sv]] \
    [glob -nocomplain [file join $root rtl interfaces *.sv]] \
    [glob -nocomplain [file join $root rtl memory *.sv]] \
    [glob -nocomplain [file join $root rtl control *.sv]] \
    [glob -nocomplain [file join $root rtl engine *.sv]] \
    [list [file join $root rtl tensorwright_top.sv]]]

read_verilog -sv $sources
synth_design -top tensorwright_top -part xc7z020clg400-1 -flatten_hierarchy rebuilt
# The registered memory-read, MAC, and requantization pipeline targets 100 MHz.
set clock_period_ns 10.000
create_clock -name clk_i -period $clock_period_ns [get_ports clk_i]
write_checkpoint -force [file join $build tensorwright_synth.dcp]
report_utilization -hierarchical -file [file join $build utilization.rpt]
report_timing_summary -delay_type max -max_paths 10 \
    -file [file join $build timing_summary.rpt]
report_clock_utilization -file [file join $build clock_utilization.rpt]

set summary [open [file join $build synthesis_status.txt] w]
puts $summary "status=complete"
puts $summary "part=xc7z020clg400-1"
puts $summary "top=tensorwright_top"
puts $summary "source_count=[llength $sources]"
puts $summary "clock_period_ns=$clock_period_ns"
close $summary
exit
