`timescale 1ns/1ps
`default_nettype none

interface tensorwright_stream_if #(
    parameter int DATA_WIDTH = 8
) (
    input logic clk_i,
    input logic rst_ni
);
    logic                  tvalid;
    logic                  tready;
    logic [DATA_WIDTH-1:0] tdata;
    logic                  tlast;

    modport source (
        input  tready,
        output tvalid,
        output tdata,
        output tlast
    );

    modport sink (
        output tready,
        input  tvalid,
        input  tdata,
        input  tlast
    );
endinterface

`default_nettype wire
