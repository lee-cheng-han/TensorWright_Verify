`timescale 1ns/1ps
`default_nettype none

module tensorwright_postprocess #(
    parameter int ACC_WIDTH = 32,
    parameter int MULTIPLIER_WIDTH = 31,
    parameter int SHIFT_WIDTH = 7,
    parameter int OUTPUT_WIDTH = 8
) (
    input  logic                              clk_i,
    input  logic                              rst_ni,
    input  logic                              valid_i,
    input  logic signed [ACC_WIDTH-1:0]       accumulator_i,
    input  logic signed [ACC_WIDTH-1:0]       bias_i,
    input  logic        [MULTIPLIER_WIDTH-1:0] multiplier_i,
    input  logic        [SHIFT_WIDTH-1:0]      shift_i,
    input  logic                              relu_i,
    output logic                              valid_o,
    output logic signed [OUTPUT_WIDTH-1:0]    result_o,
    output logic                              overflow_o
);
    logic signed [ACC_WIDTH:0] biased_ext;
    logic signed [ACC_WIDTH-1:0] biased;
    logic signed [63:0] product;
    logic [63:0] magnitude;
    logic [63:0] rounded_magnitude;
    logic signed [63:0] rounded;
    logic signed [OUTPUT_WIDTH-1:0] next_result;
    logic bias_overflow;

    always_comb begin
        biased_ext = $signed(accumulator_i) + $signed(bias_i);
        biased = biased_ext[ACC_WIDTH-1:0];
        bias_overflow = biased_ext[ACC_WIDTH] != biased_ext[ACC_WIDTH-1];
        product = $signed(biased) * $signed({1'b0, multiplier_i});

        if (product < 0) begin
            magnitude = $unsigned(-product);
        end else begin
            magnitude = $unsigned(product);
        end

        if (shift_i == 0) begin
            rounded_magnitude = magnitude;
        end else if (shift_i >= 64) begin
            rounded_magnitude = '0;
        end else begin
`ifdef TENSORWRIGHT_DEMO_FAULT_REQUANT_ROUND
            // Demo-only defect: truncate instead of adding the round-to-nearest bias.
            rounded_magnitude = magnitude >> shift_i;
`else
            rounded_magnitude =
                (magnitude + (64'd1 << (shift_i - 1'b1))) >> shift_i;
`endif
        end

        if (product < 0) begin
            rounded = -$signed(rounded_magnitude);
        end else begin
            rounded = $signed(rounded_magnitude);
        end

        if (bias_overflow) begin
            next_result = '0;
        end else if (relu_i && rounded < 0) begin
            next_result = '0;
        end else if (rounded > 127) begin
            next_result = 8'sd127;
        end else if (rounded < -128) begin
            next_result = 8'sh80;
        end else begin
            next_result = rounded[OUTPUT_WIDTH-1:0];
        end
    end

    always_ff @(posedge clk_i) begin
        if (!rst_ni) begin
            valid_o    <= 1'b0;
            result_o   <= '0;
            overflow_o <= 1'b0;
        end else begin
            valid_o <= valid_i;
            if (valid_i) begin
                result_o   <= next_result;
                overflow_o <= bias_overflow;
            end else begin
                overflow_o <= 1'b0;
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk_i) begin
        if (rst_ni && valid_i) begin
            assert (!$isunknown({accumulator_i, bias_i, multiplier_i, shift_i, relu_i}));
            assert (!bias_overflow);
        end
    end

    initial begin
        assert (ACC_WIDTH == 32);
        assert (MULTIPLIER_WIDTH == 31);
        assert (OUTPUT_WIDTH == 8);
    end
`endif
endmodule

`default_nettype wire
