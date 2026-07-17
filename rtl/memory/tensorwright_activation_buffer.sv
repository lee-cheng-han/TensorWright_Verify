`timescale 1ns/1ps
`default_nettype none

module tensorwright_activation_buffer #(
    parameter int DATA_WIDTH = 8,
    parameter int DEPTH = 64,
    localparam int COUNT_WIDTH = $clog2(DEPTH + 1)
) (
    input  logic                  clk_i,
    input  logic                  rst_ni,
    input  logic                  s_tvalid_i,
    output logic                  s_tready_o,
    input  logic [DATA_WIDTH-1:0] s_tdata_i,
    input  logic                  s_tlast_i,
    output logic                  m_tvalid_o,
    input  logic                  m_tready_i,
    output logic [DATA_WIDTH-1:0] m_tdata_o,
    output logic                  m_tlast_o,
    output logic [COUNT_WIDTH-1:0] count_o
);
    tensorwright_stream_fifo #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH)
    ) fifo (.*);
endmodule

`default_nettype wire
