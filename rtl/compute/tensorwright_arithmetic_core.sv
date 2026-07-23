`timescale 1ns/1ps
`default_nettype none

module tensorwright_arithmetic_core #(
    parameter int LANES = 9,
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH = 32
) (
    input  wire logic                                clk_i,
    input  wire logic                                rst_ni,
    input  wire logic                                valid_i,
    input  wire logic                                clear_i,
    input  wire logic                                last_i,
    input  wire logic signed [LANES-1:0][DATA_WIDTH-1:0] activations_i,
    input  wire logic signed [LANES-1:0][DATA_WIDTH-1:0] weights_i,
    input  wire logic signed [ACC_WIDTH-1:0]          bias_i,
    input  wire logic        [30:0]                   multiplier_i,
    input  wire logic        [6:0]                    shift_i,
    input  wire logic                                relu_i,
    output logic                                valid_o,
    output logic signed [7:0]                   result_o,
    output logic                                overflow_o
);
    logic signed [LANES-1:0][(2*DATA_WIDTH)-1:0] products;
    logic signed [LANES-1:0][(2*DATA_WIDTH)-1:0] products_q;
    logic signed [LANES-1:0][DATA_WIDTH-1:0] activations_q;
    logic signed [LANES-1:0][DATA_WIDTH-1:0] weights_q;
    logic signed [ACC_WIDTH-1:0] lane_sum;
    logic signed [ACC_WIDTH-1:0] lane_sum_q;
    logic signed [ACC_WIDTH-1:0] accumulator;
    logic signed [ACC_WIDTH:0] next_accumulator;
    logic accumulator_overflow;
    logic accumulator_overflow_q;
    logic postprocess_overflow;
    logic postprocess_valid;
    logic valid_q, clear_q, last_q;
    logic input_valid_q, input_clear_q, input_last_q;
    logic product_valid_q, product_clear_q, product_last_q;
    logic signed [ACC_WIDTH-1:0] bias_q;
    logic signed [ACC_WIDTH-1:0] input_bias_q;
    logic signed [ACC_WIDTH-1:0] product_bias_q;
    logic [30:0] multiplier_q;
    logic [30:0] input_multiplier_q;
    logic [30:0] product_multiplier_q;
    logic [6:0] shift_q;
    logic [6:0] input_shift_q;
    logic [6:0] product_shift_q;
    logic relu_q;
    logic input_relu_q;
    logic product_relu_q;
    generate
        genvar index;
        for (index = 0; index < LANES; index = index + 1) begin : gen_multipliers
            tensorwright_multiplier #(
                .DATA_WIDTH(DATA_WIDTH)
            ) multiplier (
                .activation_i(activations_q[index]),
                .weight_i(weights_q[index]),
                .product_o(products[index])
            );
        end
    endgenerate

    tensorwright_adder_tree #(
        .LANES(LANES),
        .INPUT_WIDTH(2 * DATA_WIDTH),
        .OUTPUT_WIDTH(ACC_WIDTH)
    ) adder_tree (
        .values_i(products_q),
        .sum_o(lane_sum)
    );

    always_comb begin
        if (clear_q) begin
            next_accumulator = {lane_sum_q[ACC_WIDTH-1], lane_sum_q};
        end else begin
            next_accumulator = $signed(accumulator) + $signed(lane_sum_q);
        end
        accumulator_overflow =
            next_accumulator[ACC_WIDTH] != next_accumulator[ACC_WIDTH-1];
    end

    always_ff @(posedge clk_i) begin
        if (!rst_ni) begin
            accumulator <= '0;
            input_valid_q <= 1'b0;
            product_valid_q <= 1'b0;
            activations_q <= '0;
            weights_q <= '0;
            products_q <= '0;
            input_clear_q <= 1'b0;
            input_last_q <= 1'b0;
            input_bias_q <= '0;
            input_multiplier_q <= '0;
            input_shift_q <= '0;
            input_relu_q <= 1'b0;
            product_clear_q <= 1'b0;
            product_last_q <= 1'b0;
            product_bias_q <= '0;
            product_multiplier_q <= '0;
            product_shift_q <= '0;
            product_relu_q <= 1'b0;
            valid_q <= 1'b0;
            lane_sum_q <= '0;
            clear_q <= 1'b0;
            last_q <= 1'b0;
            bias_q <= '0;
            multiplier_q <= '0;
            shift_q <= '0;
            relu_q <= 1'b0;
        end else begin
            input_valid_q <= valid_i;
            if (valid_i) begin
                activations_q <= activations_i;
                weights_q <= weights_i;
                input_clear_q <= clear_i;
                input_last_q <= last_i;
                input_bias_q <= bias_i;
                input_multiplier_q <= multiplier_i;
                input_shift_q <= shift_i;
                input_relu_q <= relu_i;
            end
            product_valid_q <= input_valid_q;
            if (input_valid_q) begin
                products_q <= products;
                product_clear_q <= input_clear_q;
                product_last_q <= input_last_q;
                product_bias_q <= input_bias_q;
                product_multiplier_q <= input_multiplier_q;
                product_shift_q <= input_shift_q;
                product_relu_q <= input_relu_q;
            end
            valid_q <= product_valid_q;
            if (product_valid_q) begin
                lane_sum_q <= lane_sum;
                clear_q <= product_clear_q;
                last_q <= product_last_q;
                bias_q <= product_bias_q;
                multiplier_q <= product_multiplier_q;
                shift_q <= product_shift_q;
                relu_q <= product_relu_q;
            end
            if (valid_q) begin
              if (last_q) begin
                accumulator <= '0;
              end else begin
                accumulator <= next_accumulator[ACC_WIDTH-1:0];
              end
            end
        end
    end

    tensorwright_postprocess postprocess (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .valid_i(valid_q && last_q && !accumulator_overflow),
        .accumulator_i(next_accumulator[ACC_WIDTH-1:0]),
        .bias_i(bias_q),
        .multiplier_i(multiplier_q),
        .shift_i(shift_q),
        .relu_i(relu_q),
        .valid_o(postprocess_valid),
        .result_o(result_o),
        .overflow_o(postprocess_overflow)
    );

    always_ff @(posedge clk_i) begin
        if (!rst_ni) begin
            accumulator_overflow_q <= 1'b0;
        end else begin
            accumulator_overflow_q <= valid_q && accumulator_overflow;
        end
    end
    assign valid_o = postprocess_valid;
    assign overflow_o = accumulator_overflow_q || postprocess_overflow;

`ifndef SYNTHESIS
    always_ff @(posedge clk_i) begin
        if (rst_ni && valid_i) begin
            assert (!$isunknown({clear_i, last_i, activations_i, weights_i}));
            assert (!accumulator_overflow);
        end
    end
`endif
endmodule

`default_nettype wire
