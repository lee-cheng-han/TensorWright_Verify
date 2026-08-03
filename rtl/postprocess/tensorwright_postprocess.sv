`timescale 1ns/1ps
`default_nettype none

module tensorwright_postprocess #(
    parameter int ACC_WIDTH = 32,
    parameter int MULTIPLIER_WIDTH = 31,
    parameter int SHIFT_WIDTH = 7,
    parameter int OUTPUT_WIDTH = 8
) (
    input  wire logic                              clk_i,
    input  wire logic                              rst_ni,
    input  wire logic                              valid_i,
    input  wire logic signed [ACC_WIDTH-1:0]       accumulator_i,
    input  wire logic signed [ACC_WIDTH-1:0]       bias_i,
    input  wire logic        [MULTIPLIER_WIDTH-1:0] multiplier_i,
    input  wire logic        [SHIFT_WIDTH-1:0]      shift_i,
    input  wire logic                              relu_i,
    output logic                              valid_o,
    output logic signed [OUTPUT_WIDTH-1:0]    result_o,
    output logic                              overflow_o
);
    logic signed [ACC_WIDTH:0] biased_ext;
    logic signed [ACC_WIDTH-1:0] biased;
    logic signed [ACC_WIDTH-1:0] biased_q;
    logic signed [63:0] product;
    logic signed [63:0] product_q;
    logic [63:0] magnitude;
    logic [63:0] magnitude_q;
    logic [63:0] rounded_magnitude;
    logic [63:0] rounded_magnitude_q;
    logic signed [OUTPUT_WIDTH-1:0] next_result;
    logic bias_overflow;
    logic bias_overflow_q;
    logic biased_valid_q;
    logic biased_overflow_q;
    logic [MULTIPLIER_WIDTH-1:0] biased_multiplier_q;
    logic [SHIFT_WIDTH-1:0] biased_shift_q;
    logic biased_relu_q;
    logic valid_q;
    logic magnitude_valid_q;
    logic rounded_valid_q;
    logic [SHIFT_WIDTH-1:0] shift_q;
    logic relu_q;
    logic magnitude_negative_q;
    logic [SHIFT_WIDTH-1:0] magnitude_shift_q;
    logic magnitude_relu_q;
    logic magnitude_overflow_q;
    logic rounded_negative_q;
    logic rounded_relu_q;
    logic rounded_overflow_q;
    // RTL testbenches sample these transaction-aligned observability registers.
    /* verilator lint_off UNUSEDSIGNAL */
    logic signed [ACC_WIDTH-1:0] trace_accumulator_q;
    logic signed [ACC_WIDTH-1:0] trace_bias_q;
    logic signed [ACC_WIDTH-1:0] trace_biased_q;
    logic [MULTIPLIER_WIDTH-1:0] trace_multiplier_q;
    logic [SHIFT_WIDTH-1:0] trace_shift_q;
    /* verilator lint_on UNUSEDSIGNAL */

    always_comb begin
        biased_ext = $signed(accumulator_i) + $signed(bias_i);
        biased = biased_ext[ACC_WIDTH-1:0];
        bias_overflow = biased_ext[ACC_WIDTH] != biased_ext[ACC_WIDTH-1];
    end

    always_comb begin
        product = $signed(biased_q) * $signed({1'b0, biased_multiplier_q});
    end

    always_comb begin
        if (product_q < 0) begin
            magnitude = $unsigned(-product_q);
        end else begin
            magnitude = $unsigned(product_q);
        end
    end

    always_comb begin
        if (magnitude_shift_q == 0) begin
            rounded_magnitude = magnitude_q;
        end else if (magnitude_shift_q >= 64) begin
            rounded_magnitude = '0;
        end else begin
`ifdef TENSORWRIGHT_DEMO_FAULT_REQUANT_ROUND
            // Demo-only defect: truncate instead of adding the round-to-nearest bias.
            rounded_magnitude = magnitude_q >> magnitude_shift_q[5:0];
`else
            // Equivalent to adding 2**(shift-1) before shifting, but avoids
            // placing a wide carry chain in front of the variable shifter.
            rounded_magnitude =
                (magnitude_q >> magnitude_shift_q[5:0]) +
                {{63{1'b0}}, magnitude_q[magnitude_shift_q[5:0] - 1'b1]};
`endif
        end
    end

    always_comb begin
        if (rounded_overflow_q) begin
            next_result = '0;
        end else if (rounded_relu_q && rounded_negative_q) begin
            next_result = '0;
        end else if (!rounded_negative_q && rounded_magnitude_q > 127) begin
            next_result = 8'sd127;
        end else if (rounded_negative_q && rounded_magnitude_q > 128) begin
            next_result = 8'sh80;
        end else if (rounded_negative_q) begin
            next_result = -$signed(rounded_magnitude_q[OUTPUT_WIDTH-1:0]);
        end else begin
            next_result = rounded_magnitude_q[OUTPUT_WIDTH-1:0];
        end
    end

    always_ff @(posedge clk_i) begin
        if (!rst_ni) begin
            valid_q    <= 1'b0;
            biased_valid_q <= 1'b0;
            magnitude_valid_q <= 1'b0;
            rounded_valid_q <= 1'b0;
            valid_o    <= 1'b0;
            result_o   <= '0;
            overflow_o <= 1'b0;
            product_q <= '0;
            biased_q <= '0;
            biased_overflow_q <= 1'b0;
            biased_multiplier_q <= '0;
            biased_shift_q <= '0;
            biased_relu_q <= 1'b0;
            bias_overflow_q <= 1'b0;
            shift_q <= '0;
            relu_q <= 1'b0;
            magnitude_q <= '0;
            magnitude_negative_q <= 1'b0;
            magnitude_shift_q <= '0;
            magnitude_relu_q <= 1'b0;
            magnitude_overflow_q <= 1'b0;
            rounded_magnitude_q <= '0;
            rounded_negative_q <= 1'b0;
            rounded_relu_q <= 1'b0;
            rounded_overflow_q <= 1'b0;
            trace_accumulator_q <= '0;
            trace_bias_q <= '0;
            trace_biased_q <= '0;
            trace_multiplier_q <= '0;
            trace_shift_q <= '0;
        end else begin
            biased_valid_q <= valid_i;
            valid_q <= biased_valid_q;
            magnitude_valid_q <= valid_q;
            rounded_valid_q <= magnitude_valid_q;
            valid_o <= rounded_valid_q;
            if (valid_i) begin
                biased_q <= biased;
                biased_overflow_q <= bias_overflow;
                biased_multiplier_q <= multiplier_i;
                biased_shift_q <= shift_i;
                biased_relu_q <= relu_i;
                trace_accumulator_q <= accumulator_i;
                trace_bias_q <= bias_i;
                trace_biased_q <= biased;
                trace_multiplier_q <= multiplier_i;
                trace_shift_q <= shift_i;
            end
            if (biased_valid_q) begin
                product_q <= product;
                bias_overflow_q <= biased_overflow_q;
                shift_q <= biased_shift_q;
                relu_q <= biased_relu_q;
            end
            if (valid_q) begin
                magnitude_q <= magnitude;
                magnitude_negative_q <= product_q < 0;
                magnitude_shift_q <= shift_q;
                magnitude_relu_q <= relu_q;
                magnitude_overflow_q <= bias_overflow_q;
            end
            if (magnitude_valid_q) begin
                rounded_magnitude_q <= rounded_magnitude;
                rounded_negative_q <= magnitude_negative_q;
                rounded_relu_q <= magnitude_relu_q;
                rounded_overflow_q <= magnitude_overflow_q;
            end
            if (rounded_valid_q) begin
                result_o   <= next_result;
                overflow_o <= rounded_overflow_q;
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
