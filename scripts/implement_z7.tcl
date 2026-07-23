set root [file normalize [file join [file dirname [info script]] ..]]
set build [file join $root build implementation]
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
synth_design -top tensorwright_top -part xc7z020clg400-1 \
    -flatten_hierarchy rebuilt -mode out_of_context
set clock_period_ns 10.000
create_clock -name clk_i -period $clock_period_ns [get_ports clk_i]
opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force [file join $build tensorwright_routed.dcp]
report_utilization -hierarchical -file [file join $build utilization.rpt]
report_timing_summary -delay_type min_max -max_paths 20 \
    -file [file join $build timing_summary.rpt]
report_route_status -file [file join $build route_status.rpt]
report_drc -file [file join $build drc.rpt]
report_power -file [file join $build power.rpt]

set timing_paths [get_timing_paths -setup -max_paths 1]
set worst_slack [get_property SLACK $timing_paths]
set route_status [get_property ROUTE_STATUS [current_design]]
set summary [open [file join $build implementation_status.txt] w]
puts $summary "status=complete"
puts $summary "part=xc7z020clg400-1"
puts $summary "top=tensorwright_top"
puts $summary "source_count=[llength $sources]"
puts $summary "clock_period_ns=$clock_period_ns"
puts $summary "worst_setup_slack_ns=$worst_slack"
puts $summary "route_status=$route_status"
close $summary
exit
