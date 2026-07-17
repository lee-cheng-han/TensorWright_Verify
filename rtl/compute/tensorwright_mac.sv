`timescale 1ns/1ps
`default_nettype none

module tensorwright_mac #(
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH = 32
) (
    input  logic                         clk_i,
    input  logic                         rst_ni,
    input  logic                         valid_i,
    input  logic                         clear_i,
    input  logic signed [DATA_WIDTH-1:0] activation_i,
    input  logic signed [DATA_WIDTH-1:0] weight_i,
    output logic                         valid_o,
    output logic signed [ACC_WIDTH-1:0]  accumulator_o,
    output logic                         overflow_o
);
    logic signed [(2*DATA_WIDTH)-1:0] product;
    logic signed [ACC_WIDTH:0] next_accumulator;

    always_comb begin
        product = activation_i * weight_i;
        if (clear_i) begin
            next_accumulator = {
                {(ACC_WIDTH+1-(2*DATA_WIDTH)){product[(2*DATA_WIDTH)-1]}},
                product
            };
        end else begin
            next_accumulator =
                {accumulator_o[ACC_WIDTH-1], accumulator_o} +
                {
                    {(ACC_WIDTH+1-(2*DATA_WIDTH)){product[(2*DATA_WIDTH)-1]}},
                    product
                };
        end
    end

    always_ff @(posedge clk_i) begin
        if (!rst_ni) begin
            valid_o      <= 1'b0;
            accumulator_o <= '0;
            overflow_o   <= 1'b0;
        end else begin
            valid_o    <= valid_i;
            overflow_o <= 1'b0;
            if (valid_i) begin
                accumulator_o <= next_accumulator[ACC_WIDTH-1:0];
                overflow_o <= next_accumulator[ACC_WIDTH] != next_accumulator[ACC_WIDTH-1];
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk_i) begin
        if (rst_ni && valid_i) begin
            assert (!$isunknown({clear_i, activation_i, weight_i}));
        end
    end
`endif
endmodule

`default_nettype wire
