`timescale 1ns/1ps
`default_nettype none

interface tensorwright_axil_if #(
    parameter int ADDRESS_WIDTH = 12,
    parameter int DATA_WIDTH = 32
) (
    input logic clk_i,
    input logic rst_ni
);
    logic [ADDRESS_WIDTH-1:0] awaddr;
    logic awvalid, awready;
    logic [DATA_WIDTH-1:0] wdata;
    logic [(DATA_WIDTH/8)-1:0] wstrb;
    logic wvalid, wready;
    logic [1:0] bresp;
    logic bvalid, bready;
    logic [ADDRESS_WIDTH-1:0] araddr;
    logic arvalid, arready;
    logic [DATA_WIDTH-1:0] rdata;
    logic [1:0] rresp;
    logic rvalid, rready;

    modport manager (output awaddr, awvalid, wdata, wstrb, wvalid, bready,
        araddr, arvalid, rready, input awready, wready, bresp, bvalid,
        arready, rdata, rresp, rvalid);
    modport subordinate (input awaddr, awvalid, wdata, wstrb, wvalid, bready,
        araddr, arvalid, rready, output awready, wready, bresp, bvalid,
        arready, rdata, rresp, rvalid);
endinterface

`default_nettype wire
