`timescale 1ns/1ps
`default_nettype none

module tensorwright_convolution_engine #(
    parameter int IMAGE_WIDTH = 5,
    parameter int IMAGE_HEIGHT = 5,
    parameter int INPUT_CHANNELS = 2,
    parameter int OUTPUT_CHANNELS = 2,
    localparam int OUTPUT_WIDTH = IMAGE_WIDTH - 2,
    localparam int OUTPUT_HEIGHT = IMAGE_HEIGHT - 2,
    localparam int WEIGHT_COUNT = OUTPUT_CHANNELS * INPUT_CHANNELS * 9,
    localparam int ACTIVATION_COUNT = INPUT_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH,
    localparam int WEIGHT_ADDRESS_WIDTH = $clog2(WEIGHT_COUNT + 1),
    localparam int ACTIVATION_ADDRESS_WIDTH = $clog2(ACTIVATION_COUNT + 1),
    localparam int INPUT_CHANNEL_WIDTH = (INPUT_CHANNELS <= 1) ? 1 : $clog2(INPUT_CHANNELS),
    localparam int OUTPUT_CHANNEL_WIDTH = (OUTPUT_CHANNELS <= 1) ? 1 : $clog2(OUTPUT_CHANNELS),
    localparam int OUTPUT_X_WIDTH = (OUTPUT_WIDTH <= 1) ? 1 : $clog2(OUTPUT_WIDTH),
    localparam int OUTPUT_Y_WIDTH = (OUTPUT_HEIGHT <= 1) ? 1 : $clog2(OUTPUT_HEIGHT)
) (
    input logic clk_i, input logic rst_ni, input logic start_i, input logic soft_reset_i,
    input logic weight_tvalid_i, output logic weight_tready_o,
    input logic signed [7:0] weight_tdata_i, input logic weight_tlast_i,
    input logic activation_tvalid_i, output logic activation_tready_o,
    input logic signed [7:0] activation_tdata_i, input logic activation_tlast_i,
    output logic output_tvalid_o, input logic output_tready_i,
    output logic signed [7:0] output_tdata_o, output logic output_tlast_o,
    input logic signed [OUTPUT_CHANNELS-1:0][31:0] biases_i,
    input logic [OUTPUT_CHANNELS-1:0][30:0] multipliers_i,
    input logic [OUTPUT_CHANNELS-1:0][6:0] shifts_i,
    input logic [OUTPUT_CHANNELS-1:0] relu_i,
    output logic busy_o, output logic done_o, output logic overflow_o,
    output logic compute_active_o, output logic [31:0] macs_executed_o
);
    typedef enum logic [2:0] {IDLE, LOAD_WEIGHTS, LOAD_ACTIVATIONS, COMPUTE, WAIT_RESULT, OUTPUT} state_t;
    state_t state;
    logic signed [7:0] weights [0:WEIGHT_COUNT-1];
    logic signed [7:0] activations [0:ACTIVATION_COUNT-1];
    logic [WEIGHT_ADDRESS_WIDTH-1:0] weight_count;
    logic [ACTIVATION_ADDRESS_WIDTH-1:0] activation_count;
    logic [INPUT_CHANNEL_WIDTH-1:0] input_channel;
    logic [OUTPUT_CHANNEL_WIDTH-1:0] output_channel;
    logic [OUTPUT_X_WIDTH-1:0] output_x;
    logic [OUTPUT_Y_WIDTH-1:0] output_y;
    logic signed [8:0][7:0] core_activations, core_weights;
    logic core_valid, core_result_valid, core_overflow;
    logic signed [7:0] core_result;
    localparam logic [WEIGHT_ADDRESS_WIDTH-1:0] LAST_WEIGHT = WEIGHT_ADDRESS_WIDTH'(WEIGHT_COUNT - 1);
    localparam logic [ACTIVATION_ADDRESS_WIDTH-1:0] LAST_ACTIVATION = ACTIVATION_ADDRESS_WIDTH'(ACTIVATION_COUNT - 1);
    localparam logic [INPUT_CHANNEL_WIDTH-1:0] LAST_INPUT_CHANNEL = INPUT_CHANNEL_WIDTH'(INPUT_CHANNELS - 1);
    localparam logic [OUTPUT_CHANNEL_WIDTH-1:0] LAST_OUTPUT_CHANNEL = OUTPUT_CHANNEL_WIDTH'(OUTPUT_CHANNELS - 1);
    localparam logic [OUTPUT_X_WIDTH-1:0] LAST_OUTPUT_X = OUTPUT_X_WIDTH'(OUTPUT_WIDTH - 1);
    localparam logic [OUTPUT_Y_WIDTH-1:0] LAST_OUTPUT_Y = OUTPUT_Y_WIDTH'(OUTPUT_HEIGHT - 1);

    assign weight_tready_o = state == LOAD_WEIGHTS;
    assign activation_tready_o = state == LOAD_ACTIVATIONS;
    assign busy_o = state != IDLE;
    assign compute_active_o = state == COMPUTE;
    assign macs_executed_o = state == COMPUTE ? 32'd9 : 32'd0;
    assign core_valid = state == COMPUTE;

    always_comb begin
        for (integer lane = 0; lane < 9; lane++) begin
            core_activations[lane] = activations[
                input_channel * (IMAGE_HEIGHT * IMAGE_WIDTH) +
                (int'(output_y) + lane / 3) * IMAGE_WIDTH + int'(output_x) + lane % 3
            ];
            core_weights[lane] = weights[
                output_channel * (INPUT_CHANNELS * 9) + input_channel * 9 + lane
            ];
        end
    end

    tensorwright_arithmetic_core arithmetic_core (
        .clk_i(clk_i), .rst_ni(rst_ni && !soft_reset_i), .valid_i(core_valid),
        .clear_i(input_channel == 0), .last_i(input_channel == LAST_INPUT_CHANNEL),
        .activations_i(core_activations), .weights_i(core_weights),
        .bias_i(biases_i[output_channel]), .multiplier_i(multipliers_i[output_channel]),
        .shift_i(shifts_i[output_channel]), .relu_i(relu_i[output_channel]),
        .valid_o(core_result_valid), .result_o(core_result), .overflow_o(core_overflow)
    );

    always_ff @(posedge clk_i) begin
        if (!rst_ni || soft_reset_i) begin
            state <= IDLE; weight_count <= 0; activation_count <= 0; input_channel <= 0;
            output_channel <= 0; output_x <= 0; output_y <= 0; output_tvalid_o <= 0;
            output_tdata_o <= 0; output_tlast_o <= 0; done_o <= 0; overflow_o <= 0;
        end else begin
            done_o <= 0;
            case (state)
                IDLE: if (start_i) begin
                    state <= LOAD_WEIGHTS; weight_count <= 0; activation_count <= 0;
                    output_channel <= 0; output_x <= 0; output_y <= 0; overflow_o <= 0;
                end
                LOAD_WEIGHTS: if (weight_tvalid_i) begin
                    weights[weight_count] <= weight_tdata_i;
                    if (weight_count == LAST_WEIGHT) begin state <= LOAD_ACTIVATIONS; weight_count <= 0; end
                    else weight_count <= weight_count + 1'b1;
                end
                LOAD_ACTIVATIONS: if (activation_tvalid_i) begin
                    activations[activation_count] <= activation_tdata_i;
                    if (activation_count == LAST_ACTIVATION) begin
                        state <= COMPUTE; activation_count <= 0; input_channel <= 0;
                    end else activation_count <= activation_count + 1'b1;
                end
                COMPUTE: begin
                    if (input_channel == LAST_INPUT_CHANNEL) begin input_channel <= 0; state <= WAIT_RESULT; end
                    else input_channel <= input_channel + 1'b1;
                end
                WAIT_RESULT: begin
                    if (core_overflow) overflow_o <= 1;
                    if (core_result_valid) begin
`ifdef TENSORWRIGHT_DEMO_FAULT_DROPPED_TRANSFER
                        // Demo-only protocol defect: logical output 4 is consumed
                        // internally without ever presenting a valid transfer.
                        if (output_channel == 0 && output_y == 1 && output_x == 1) begin
                            output_tvalid_o <= 0;
                            output_x <= output_x + 1'b1;
                            state <= COMPUTE;
                        end else begin
                            output_tdata_o <= core_result;
                            output_tvalid_o <= 1;
                            output_tlast_o <= output_channel == LAST_OUTPUT_CHANNEL &&
                                output_y == LAST_OUTPUT_Y && output_x == LAST_OUTPUT_X;
                            state <= OUTPUT;
                        end
`else
                        output_tdata_o <= core_result;
                        output_tvalid_o <= 1;
                        output_tlast_o <= output_channel == LAST_OUTPUT_CHANNEL &&
                            output_y == LAST_OUTPUT_Y && output_x == LAST_OUTPUT_X;
                        state <= OUTPUT;
`endif
                    end
                end
                OUTPUT: if (output_tvalid_o && output_tready_i) begin
                    output_tvalid_o <= 0;
                    if (output_tlast_o) begin state <= IDLE; done_o <= 1; end
                    else begin
                        if (output_x == LAST_OUTPUT_X) begin
                            output_x <= 0;
                            if (output_y == LAST_OUTPUT_Y) begin output_y <= 0; output_channel <= output_channel + 1'b1; end
                            else output_y <= output_y + 1'b1;
                        end else output_x <= output_x + 1'b1;
                        state <= COMPUTE;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    logic stalled_q;
    logic signed [7:0] stalled_data_q;
    logic stalled_last_q;
    always_ff @(posedge clk_i) begin
        if (!rst_ni || soft_reset_i) begin
            stalled_q <= 1'b0;
        end else begin
            if (state == LOAD_WEIGHTS && weight_tvalid_i)
                assert (weight_tlast_i == (weight_count == LAST_WEIGHT));
            if (state == LOAD_ACTIVATIONS && activation_tvalid_i)
                assert (activation_tlast_i == (activation_count == LAST_ACTIVATION));
            if (stalled_q) begin
                assert (output_tvalid_o); assert (output_tdata_o == stalled_data_q);
                assert (output_tlast_o == stalled_last_q);
            end
            stalled_q <= output_tvalid_o && !output_tready_i;
            stalled_data_q <= output_tdata_o; stalled_last_q <= output_tlast_o;
        end
    end
`endif
    initial begin assert (IMAGE_WIDTH >= 3); assert (IMAGE_HEIGHT >= 3); assert (INPUT_CHANNELS > 0); assert (OUTPUT_CHANNELS > 0); end
endmodule

`default_nettype wire
