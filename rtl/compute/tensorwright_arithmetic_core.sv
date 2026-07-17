`default_nettype none

module tensorwright_arithmetic_core #(
    parameter int LANES = 9,
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH = 32
) (
    input  logic                                clk_i,
    input  logic                                rst_ni,
    input  logic                                valid_i,
    input  logic                                clear_i,
    input  logic                                last_i,
    input  logic signed [LANES-1:0][DATA_WIDTH-1:0] activations_i,
    input  logic signed [LANES-1:0][DATA_WIDTH-1:0] weights_i,
    input  logic signed [ACC_WIDTH-1:0]          bias_i,
    input  logic        [30:0]                   multiplier_i,
    input  logic        [6:0]                    shift_i,
    input  logic                                relu_i,
    output logic                                valid_o,
    output logic signed [7:0]                   result_o,
    output logic                                overflow_o
);
    logic signed [LANES-1:0][(2*DATA_WIDTH)-1:0] products;
    logic signed [ACC_WIDTH-1:0] lane_sum;
    logic signed [ACC_WIDTH-1:0] accumulator;
    logic signed [ACC_WIDTH:0] next_accumulator;
    logic accumulator_overflow;
    logic accumulator_overflow_q;
    logic postprocess_overflow;
    logic postprocess_valid;
    generate
        genvar index;
        for (index = 0; index < LANES; index = index + 1) begin : gen_multipliers
            tensorwright_multiplier #(
                .DATA_WIDTH(DATA_WIDTH)
            ) multiplier (
                .activation_i(activations_i[index]),
                .weight_i(weights_i[index]),
                .product_o(products[index])
            );
        end
    endgenerate

    tensorwright_adder_tree #(
        .LANES(LANES),
        .INPUT_WIDTH(2 * DATA_WIDTH),
        .OUTPUT_WIDTH(ACC_WIDTH)
    ) adder_tree (
        .values_i(products),
        .sum_o(lane_sum)
    );

    always_comb begin
        if (clear_i) begin
            next_accumulator = {lane_sum[ACC_WIDTH-1], lane_sum};
        end else begin
            next_accumulator = $signed(accumulator) + $signed(lane_sum);
        end
        accumulator_overflow =
            next_accumulator[ACC_WIDTH] != next_accumulator[ACC_WIDTH-1];
    end

    always_ff @(posedge clk_i) begin
        if (!rst_ni) begin
            accumulator <= '0;
        end else if (valid_i) begin
            if (last_i) begin
                accumulator <= '0;
            end else begin
                accumulator <= next_accumulator[ACC_WIDTH-1:0];
            end
        end
    end

    tensorwright_postprocess postprocess (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .valid_i(valid_i && last_i && !accumulator_overflow),
        .accumulator_i(next_accumulator[ACC_WIDTH-1:0]),
        .bias_i(bias_i),
        .multiplier_i(multiplier_i),
        .shift_i(shift_i),
        .relu_i(relu_i),
        .valid_o(postprocess_valid),
        .result_o(result_o),
        .overflow_o(postprocess_overflow)
    );

    always_ff @(posedge clk_i) begin
        if (!rst_ni) begin
            accumulator_overflow_q <= 1'b0;
        end else begin
            accumulator_overflow_q <= valid_i && accumulator_overflow;
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
