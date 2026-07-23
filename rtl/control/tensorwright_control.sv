`timescale 1ns/1ps
`default_nettype none

module tensorwright_control (
    input wire logic clk_i, input wire logic rst_ni,
    input wire logic [11:0] s_axil_awaddr_i, input wire logic s_axil_awvalid_i, output logic s_axil_awready_o,
    input wire logic [31:0] s_axil_wdata_i, input wire logic [3:0] s_axil_wstrb_i,
    input wire logic s_axil_wvalid_i, output logic s_axil_wready_o,
    output logic [1:0] s_axil_bresp_o, output logic s_axil_bvalid_o, input wire logic s_axil_bready_i,
    input wire logic [11:0] s_axil_araddr_i, input wire logic s_axil_arvalid_i, output logic s_axil_arready_o,
    output logic [31:0] s_axil_rdata_o, output logic [1:0] s_axil_rresp_o,
    output logic s_axil_rvalid_o, input wire logic s_axil_rready_i,
    input wire logic engine_done_i, input wire logic compute_active_i,
    input wire logic input_tvalid_i, input wire logic input_tready_i,
    input wire logic weight_tvalid_i, input wire logic weight_tready_i,
    input wire logic output_tvalid_i, input wire logic output_tready_i,
    input wire logic [31:0] macs_executed_i,
    output logic start_o, output logic soft_reset_o, output logic irq_o,
    output logic [31:0] input_height_o, output logic [31:0] input_width_o,
    output logic [31:0] input_channels_o, output logic [31:0] output_channels_o,
    output logic [31:0] kernel_config_o, output logic [31:0] output_height_o,
    output logic [31:0] output_width_o, output logic [31:0] flags_o,
    output logic [31:0] input_length_o, output logic [31:0] weight_length_o,
    output logic [31:0] output_length_o
);
    import tensorwright_registers_pkg::*;
    logic busy, done, error;
    logic [1:0] irq_status, irq_enable;
    logic [31:0] error_code, compute_active_count, input_stalls, output_stalls;
    logic [31:0] weight_load_cycles, output_count, input_count, layer_invocations, error_count;
    logic [63:0] cycle_count, executed_macs;
    logic aw_pending, w_pending;
    logic [11:0] awaddr_q;
    logic [31:0] wdata_q;
    logic [3:0] wstrb_q;
    logic write_commit, input_transfer, weight_transfer, output_transfer;
    logic config_valid;

    function automatic logic [31:0] saturating_increment(input logic [31:0] value);
        return value == 32'hffff_ffff ? value : value + 1'b1;
    endfunction

    function automatic logic [31:0] apply_strobes(
        input logic [31:0] old_value,
        input logic [31:0] new_value,
        input logic [3:0] strobes
    );
        logic [31:0] result;
        result = old_value;
        for (integer index = 0; index < 4; index++)
            if (strobes[index]) result[index*8 +: 8] = new_value[index*8 +: 8];
        return result;
    endfunction

    assign s_axil_awready_o = !aw_pending && !s_axil_bvalid_o;
    assign s_axil_wready_o = !w_pending && !s_axil_bvalid_o;
    assign write_commit = aw_pending && w_pending && !s_axil_bvalid_o;
    assign s_axil_arready_o = !s_axil_rvalid_o;
    assign input_transfer = input_tvalid_i && input_tready_i;
    assign weight_transfer = weight_tvalid_i && weight_tready_i;
    assign output_transfer = output_tvalid_i && output_tready_i;
    assign config_valid = input_height_o != 0 && input_width_o != 0 && input_channels_o != 0 &&
        output_channels_o != 0 && output_height_o != 0 && output_width_o != 0 &&
        input_length_o != 0 && weight_length_o != 0 && output_length_o != 0 &&
        kernel_config_o[3:0] == 3 && kernel_config_o[7:4] == 3 &&
        kernel_config_o[11:8] == 1 && kernel_config_o[15:12] == 1 &&
        kernel_config_o[31:16] == 0 && flags_o[31:1] == 0;
    assign irq_o = |(irq_status & irq_enable);

    task automatic record_error(input logic [31:0] code);
        begin
            error <= 1'b1; error_code <= code; irq_status[1] <= 1'b1;
            error_count <= saturating_increment(error_count);
        end
    endtask

    always_ff @(posedge clk_i) begin
        if (!rst_ni) begin
            aw_pending <= 0; w_pending <= 0; s_axil_bvalid_o <= 0; s_axil_bresp_o <= 0;
            s_axil_rvalid_o <= 0; s_axil_rdata_o <= 0; s_axil_rresp_o <= 0;
            busy <= 0; done <= 0; error <= 0; irq_status <= 0; irq_enable <= 0;
            error_code <= ERROR_NONE; start_o <= 0; soft_reset_o <= 0;
            input_height_o <= 0; input_width_o <= 0; input_channels_o <= 0; output_channels_o <= 0;
            kernel_config_o <= 0; output_height_o <= 0; output_width_o <= 0; flags_o <= 0;
            input_length_o <= 0; weight_length_o <= 0; output_length_o <= 0;
            cycle_count <= 0; compute_active_count <= 0; input_stalls <= 0; output_stalls <= 0;
            weight_load_cycles <= 0; output_count <= 0; input_count <= 0;
            layer_invocations <= 0; executed_macs <= 0; error_count <= 0;
        end else begin
            start_o <= 0; soft_reset_o <= 0;
            if (s_axil_awready_o && s_axil_awvalid_i) begin aw_pending <= 1; awaddr_q <= s_axil_awaddr_i; end
            if (s_axil_wready_o && s_axil_wvalid_i) begin w_pending <= 1; wdata_q <= s_axil_wdata_i; wstrb_q <= s_axil_wstrb_i; end
            if (s_axil_bvalid_o && s_axil_bready_i) s_axil_bvalid_o <= 0;
            if (write_commit) begin
                aw_pending <= 0; w_pending <= 0; s_axil_bvalid_o <= 1; s_axil_bresp_o <= 2'b00;
                case (awaddr_q)
                    ADDR_CONTROL: begin
                        if (wstrb_q[0] && wdata_q[1]) begin
                            busy <= 0; done <= 0; error <= 0; error_code <= 0; irq_status <= 0;
                            cycle_count <= 0; compute_active_count <= 0; input_stalls <= 0; output_stalls <= 0;
                            weight_load_cycles <= 0; output_count <= 0; input_count <= 0; executed_macs <= 0;
                            soft_reset_o <= 1;
                        end else begin
                            if (wstrb_q[0] && wdata_q[2]) begin error <= 0; error_code <= 0; irq_status[1] <= 0; end
                            if (wstrb_q[0] && wdata_q[0]) begin
                                if (busy) record_error(ERROR_START_WHILE_BUSY);
                                else if (!config_valid) record_error(ERROR_INVALID_CONFIG);
                                else begin
                                    busy <= 1; done <= 0; error <= 0; error_code <= 0; irq_status <= 0;
                                    cycle_count <= 0; compute_active_count <= 0; input_stalls <= 0; output_stalls <= 0;
                                    weight_load_cycles <= 0; output_count <= 0; input_count <= 0; executed_macs <= 0;
                                    layer_invocations <= saturating_increment(layer_invocations); start_o <= 1;
                                end
                            end
                        end
                    end
                    ADDR_IRQ_STATUS: irq_status <= irq_status & ~wdata_q[1:0];
                    ADDR_IRQ_ENABLE:
                        if (wstrb_q[0]) irq_enable <= wdata_q[1:0];
                    ADDR_INPUT_HEIGHT, ADDR_INPUT_WIDTH, ADDR_INPUT_CHANNELS, ADDR_OUTPUT_CHANNELS,
                    ADDR_KERNEL_CONFIG, ADDR_OUTPUT_HEIGHT, ADDR_OUTPUT_WIDTH, ADDR_FLAGS,
                    ADDR_INPUT_LENGTH, ADDR_WEIGHT_LENGTH, ADDR_OUTPUT_LENGTH: begin
                        if (busy) begin s_axil_bresp_o <= 2'b10; record_error(ERROR_BUSY_WRITE); end
                        else case (awaddr_q)
                            ADDR_INPUT_HEIGHT: input_height_o <= apply_strobes(input_height_o, wdata_q, wstrb_q);
                            ADDR_INPUT_WIDTH: input_width_o <= apply_strobes(input_width_o, wdata_q, wstrb_q);
                            ADDR_INPUT_CHANNELS: input_channels_o <= apply_strobes(input_channels_o, wdata_q, wstrb_q);
                            ADDR_OUTPUT_CHANNELS: output_channels_o <= apply_strobes(output_channels_o, wdata_q, wstrb_q);
                            ADDR_KERNEL_CONFIG: kernel_config_o <= apply_strobes(kernel_config_o, wdata_q, wstrb_q);
                            ADDR_OUTPUT_HEIGHT: output_height_o <= apply_strobes(output_height_o, wdata_q, wstrb_q);
                            ADDR_OUTPUT_WIDTH: output_width_o <= apply_strobes(output_width_o, wdata_q, wstrb_q);
                            ADDR_FLAGS: flags_o <= apply_strobes(flags_o, wdata_q, wstrb_q);
                            ADDR_INPUT_LENGTH: input_length_o <= apply_strobes(input_length_o, wdata_q, wstrb_q);
                            ADDR_WEIGHT_LENGTH: weight_length_o <= apply_strobes(weight_length_o, wdata_q, wstrb_q);
                            default: output_length_o <= apply_strobes(output_length_o, wdata_q, wstrb_q);
                        endcase
                    end
                    default: s_axil_bresp_o <= 2'b11;
                endcase
            end

            if (s_axil_rvalid_o && s_axil_rready_i) s_axil_rvalid_o <= 0;
            if (s_axil_arready_o && s_axil_arvalid_i) begin
                s_axil_rvalid_o <= 1; s_axil_rresp_o <= 2'b00;
                case (s_axil_araddr_i)
                    ADDR_DEVICE_ID: s_axil_rdata_o <= DEVICE_ID; ADDR_VERSION: s_axil_rdata_o <= INTERFACE_VERSION;
                    ADDR_CONTROL: s_axil_rdata_o <= 0; ADDR_STATUS: s_axil_rdata_o <= {29'd0, error, done, busy};
                    ADDR_IRQ_STATUS: s_axil_rdata_o <= {30'd0, irq_status}; ADDR_IRQ_ENABLE: s_axil_rdata_o <= {30'd0, irq_enable};
                    ADDR_INPUT_HEIGHT: s_axil_rdata_o <= input_height_o; ADDR_INPUT_WIDTH: s_axil_rdata_o <= input_width_o;
                    ADDR_INPUT_CHANNELS: s_axil_rdata_o <= input_channels_o; ADDR_OUTPUT_CHANNELS: s_axil_rdata_o <= output_channels_o;
                    ADDR_KERNEL_CONFIG: s_axil_rdata_o <= kernel_config_o; ADDR_OUTPUT_HEIGHT: s_axil_rdata_o <= output_height_o;
                    ADDR_OUTPUT_WIDTH: s_axil_rdata_o <= output_width_o; ADDR_FLAGS: s_axil_rdata_o <= flags_o;
                    ADDR_INPUT_LENGTH: s_axil_rdata_o <= input_length_o; ADDR_WEIGHT_LENGTH: s_axil_rdata_o <= weight_length_o;
                    ADDR_OUTPUT_LENGTH: s_axil_rdata_o <= output_length_o; ADDR_CYCLE_COUNT_LOW: s_axil_rdata_o <= cycle_count[31:0];
                    ADDR_CYCLE_COUNT_HIGH: s_axil_rdata_o <= cycle_count[63:32]; ADDR_COMPUTE_ACTIVE: s_axil_rdata_o <= compute_active_count;
                    ADDR_INPUT_STALLS: s_axil_rdata_o <= input_stalls; ADDR_OUTPUT_STALLS: s_axil_rdata_o <= output_stalls;
                    ADDR_ERROR_CODE: s_axil_rdata_o <= error_code; ADDR_WEIGHT_LOAD_CYCLES: s_axil_rdata_o <= weight_load_cycles;
                    ADDR_OUTPUT_COUNT: s_axil_rdata_o <= output_count; ADDR_INPUT_COUNT: s_axil_rdata_o <= input_count;
                    ADDR_LAYER_INVOCATIONS: s_axil_rdata_o <= layer_invocations; ADDR_EXECUTED_MACS_LOW: s_axil_rdata_o <= executed_macs[31:0];
                    ADDR_EXECUTED_MACS_HIGH: s_axil_rdata_o <= executed_macs[63:32]; ADDR_ERROR_COUNT: s_axil_rdata_o <= error_count;
                    default: begin s_axil_rdata_o <= 0; s_axil_rresp_o <= 2'b11; end
                endcase
            end

            if (busy) begin
                cycle_count <= cycle_count + 1'b1;
                if (compute_active_i) compute_active_count <= saturating_increment(compute_active_count);
                if (input_tvalid_i && !input_tready_i) input_stalls <= saturating_increment(input_stalls);
                if (output_tvalid_i && !output_tready_i) output_stalls <= saturating_increment(output_stalls);
                if (weight_transfer) weight_load_cycles <= saturating_increment(weight_load_cycles);
                if (input_transfer) input_count <= saturating_increment(input_count);
                if (output_transfer) output_count <= saturating_increment(output_count);
                executed_macs <= executed_macs + 64'(macs_executed_i);
                if (engine_done_i) begin
                    busy <= 0;
                    if ((output_count + (output_transfer ? 1 : 0)) < output_length_o) record_error(ERROR_EARLY_COMPLETION);
                    else begin done <= 1; irq_status[0] <= 1; end
                end
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk_i) if (rst_ni) begin
        if (start_o) assert (busy);
    end
`endif
endmodule

`default_nettype wire
