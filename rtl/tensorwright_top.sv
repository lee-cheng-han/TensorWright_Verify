`timescale 1ns/1ps
`default_nettype none

module tensorwright_top #(
    parameter int IMAGE_WIDTH = 5, parameter int IMAGE_HEIGHT = 5,
    parameter int INPUT_CHANNELS = 3, parameter int OUTPUT_CHANNELS = 2
) (
    input wire logic clk_i, input wire logic rst_ni,
    input wire logic [11:0] s_axil_awaddr_i, input wire logic s_axil_awvalid_i, output logic s_axil_awready_o,
    input wire logic [31:0] s_axil_wdata_i, input wire logic [3:0] s_axil_wstrb_i,
    input wire logic s_axil_wvalid_i, output logic s_axil_wready_o,
    output logic [1:0] s_axil_bresp_o, output logic s_axil_bvalid_o, input wire logic s_axil_bready_i,
    input wire logic [11:0] s_axil_araddr_i, input wire logic s_axil_arvalid_i, output logic s_axil_arready_o,
    output logic [31:0] s_axil_rdata_o, output logic [1:0] s_axil_rresp_o,
    output logic s_axil_rvalid_o, input wire logic s_axil_rready_i,
    input wire logic weight_tvalid_i, output logic weight_tready_o,
    input wire logic signed [7:0] weight_tdata_i, input wire logic weight_tlast_i,
    input wire logic activation_tvalid_i, output logic activation_tready_o,
    input wire logic signed [7:0] activation_tdata_i, input wire logic activation_tlast_i,
    output logic output_tvalid_o, input wire logic output_tready_i,
    output logic signed [7:0] output_tdata_o, output logic output_tlast_o,
    input wire logic signed [OUTPUT_CHANNELS-1:0][31:0] biases_i,
    input wire logic [OUTPUT_CHANNELS-1:0][30:0] multipliers_i,
    input wire logic [OUTPUT_CHANNELS-1:0][6:0] shifts_i,
    input wire logic [OUTPUT_CHANNELS-1:0] relu_i,
    output logic irq_o
);
    logic start, soft_reset, engine_done, engine_busy, engine_overflow;
    logic compute_active; logic [31:0] macs_executed;
    logic [31:0] input_height, input_width, input_channels, output_channels;
    logic [31:0] kernel_config, output_height, output_width, flags;
    logic [31:0] input_length, weight_length, output_length;

    tensorwright_control control (.*,
        .engine_done_i(engine_done), .compute_active_i(compute_active),
        .input_tvalid_i(activation_tvalid_i), .input_tready_i(activation_tready_o),
        .weight_tvalid_i(weight_tvalid_i), .weight_tready_i(weight_tready_o),
        .output_tvalid_i(output_tvalid_o), .output_tready_i(output_tready_i),
        .macs_executed_i(macs_executed), .start_o(start), .soft_reset_o(soft_reset),
        .input_height_o(input_height), .input_width_o(input_width),
        .input_channels_o(input_channels), .output_channels_o(output_channels),
        .kernel_config_o(kernel_config), .output_height_o(output_height),
        .output_width_o(output_width), .flags_o(flags), .input_length_o(input_length),
        .weight_length_o(weight_length), .output_length_o(output_length)
    );

    tensorwright_convolution_engine #(
        .IMAGE_WIDTH(IMAGE_WIDTH), .IMAGE_HEIGHT(IMAGE_HEIGHT),
        .INPUT_CHANNELS(INPUT_CHANNELS), .OUTPUT_CHANNELS(OUTPUT_CHANNELS)
    ) engine (
        .clk_i(clk_i), .rst_ni(rst_ni), .start_i(start), .soft_reset_i(soft_reset),
        .weight_tvalid_i(weight_tvalid_i), .weight_tready_o(weight_tready_o),
        .weight_tdata_i(weight_tdata_i), .weight_tlast_i(weight_tlast_i),
        .activation_tvalid_i(activation_tvalid_i), .activation_tready_o(activation_tready_o),
        .activation_tdata_i(activation_tdata_i), .activation_tlast_i(activation_tlast_i),
        .output_tvalid_o(output_tvalid_o), .output_tready_i(output_tready_i),
        .output_tdata_o(output_tdata_o), .output_tlast_o(output_tlast_o),
        .biases_i(biases_i), .multipliers_i(multipliers_i), .shifts_i(shifts_i), .relu_i(relu_i),
        .busy_o(engine_busy), .done_o(engine_done), .overflow_o(engine_overflow),
        .compute_active_o(compute_active), .macs_executed_o(macs_executed)
    );

`ifndef SYNTHESIS
    always_ff @(posedge clk_i) if (rst_ni && start) begin
        assert (input_width == IMAGE_WIDTH && input_height == IMAGE_HEIGHT);
        assert (input_channels == INPUT_CHANNELS && output_channels == OUTPUT_CHANNELS);
        assert (output_width == IMAGE_WIDTH - 2 && output_height == IMAGE_HEIGHT - 2);
        assert (kernel_config == 32'h0000_1133 && flags[31:1] == 0 && !$isunknown(flags[0]));
        assert (input_length == IMAGE_WIDTH * IMAGE_HEIGHT * INPUT_CHANNELS);
        assert (weight_length == OUTPUT_CHANNELS * INPUT_CHANNELS * 9);
        assert (output_length == (IMAGE_WIDTH - 2) * (IMAGE_HEIGHT - 2) * OUTPUT_CHANNELS);
    end
    always_ff @(posedge clk_i) if (rst_ni) begin
        assert (!engine_overflow);
        if (engine_done) assert (!engine_busy);
    end
`endif
endmodule

`default_nettype wire
