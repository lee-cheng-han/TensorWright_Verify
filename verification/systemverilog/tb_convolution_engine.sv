`timescale 1ns/1ps
`default_nettype none

module tb_convolution_engine;
    logic clk = 0, rst_n = 0, start = 0, soft_reset = 0;
    always #5 clk = ~clk;
    logic weight_valid, weight_ready, weight_last; logic signed [7:0] weight_data;
    logic activation_valid, activation_ready, activation_last; logic signed [7:0] activation_data;
    logic output_valid, output_ready, output_last; logic signed [7:0] output_data;
    logic signed [1:0][31:0] biases; logic [1:0][30:0] multipliers;
    logic [1:0][6:0] shifts; logic [1:0] relu;
    logic busy, done, overflow, compute_active; logic [31:0] macs;
    integer weights [0:53]; integer activations [0:74]; integer expected [0:17];

    tensorwright_convolution_engine #(
        .IMAGE_WIDTH(5), .IMAGE_HEIGHT(5), .INPUT_CHANNELS(3), .OUTPUT_CHANNELS(2)
    ) dut (
        .clk_i(clk), .rst_ni(rst_n), .start_i(start), .soft_reset_i(soft_reset),
        .weight_tvalid_i(weight_valid), .weight_tready_o(weight_ready),
        .weight_tdata_i(weight_data), .weight_tlast_i(weight_last),
        .activation_tvalid_i(activation_valid), .activation_tready_o(activation_ready),
        .activation_tdata_i(activation_data), .activation_tlast_i(activation_last),
        .output_tvalid_o(output_valid), .output_tready_i(output_ready),
        .output_tdata_o(output_data), .output_tlast_o(output_last),
        .biases_i(biases), .multipliers_i(multipliers), .shifts_i(shifts), .relu_i(relu),
        .busy_o(busy), .done_o(done), .overflow_o(overflow),
        .compute_active_o(compute_active), .macs_executed_o(macs)
    );

    initial begin : test
        integer file, trace_file, arithmetic_trace_file, scanned, case_count, case_index, index, sent, received, cycles, simulation_cycle;
        integer expected_transfers, logical_sequence;
        integer bias_value, multiplier_value, shift_value, relu_value;
        logic [31:0] lfsr; logic source_fire, sink_fire;
        logic allow_mismatch;
        string vector_file, trace_path, arithmetic_trace_path;
        allow_mismatch = $test$plusargs("ALLOW_MISMATCH");
        expected_transfers = $test$plusargs("EXPECT_DROPPED_TRANSFER") ? 17 : 18;
        if (!$value$plusargs("VECTOR_FILE=%s", vector_file)) $fatal(1, "VECTOR_FILE missing");
        file = $fopen(vector_file, "r"); if (!file) $fatal(1, "cannot open vectors");
        trace_file = 0;
        if ($value$plusargs("TRACE_FILE=%s", trace_path)) begin
            trace_file = $fopen(trace_path, "w");
            if (!trace_file) $fatal(1, "cannot open trace output");
        end
        arithmetic_trace_file = 0;
        if ($value$plusargs("ARITH_TRACE_FILE=%s", arithmetic_trace_path)) begin
            arithmetic_trace_file = $fopen(arithmetic_trace_path, "w");
            if (!arithmetic_trace_file) $fatal(1, "cannot open arithmetic trace output");
        end
        scanned = $fscanf(file, "%d", case_count);
        weight_valid = 0; activation_valid = 0; output_ready = 0; weight_data = 0;
        activation_data = 0; weight_last = 0; activation_last = 0;
        repeat (3) @(posedge clk); rst_n = 1; @(posedge clk);
        lfsr = 32'h243f6a88; simulation_cycle = 0;
        for (case_index = 0; case_index < case_count; case_index++) begin
            for (index = 0; index < 2; index++) begin scanned = $fscanf(file, "%d", bias_value); biases[index] = bias_value; end
            for (index = 0; index < 2; index++) begin scanned = $fscanf(file, "%d", multiplier_value); multipliers[index] = multiplier_value; end
            for (index = 0; index < 2; index++) begin scanned = $fscanf(file, "%d", shift_value); shifts[index] = shift_value; end
            for (index = 0; index < 2; index++) begin scanned = $fscanf(file, "%d", relu_value); relu[index] = relu_value; end
            for (index = 0; index < 54; index++) scanned = $fscanf(file, "%d", weights[index]);
            for (index = 0; index < 75; index++) scanned = $fscanf(file, "%d", activations[index]);
            for (index = 0; index < 18; index++) scanned = $fscanf(file, "%d", expected[index]);
            @(negedge clk); start = 1; @(posedge clk); #1; start = 0;

            sent = 0; cycles = 0;
            while (sent < 54) begin
                @(negedge clk); lfsr = {lfsr[30:0], lfsr[31]^lfsr[21]^lfsr[1]^lfsr[0]};
                if (!weight_valid && lfsr[2]) begin weight_valid = 1; weight_data = 8'(weights[sent]); weight_last = sent == 53; end
                #1; source_fire = weight_valid && weight_ready; @(posedge clk); #1;
                if (source_fire) begin sent++; weight_valid = 0; end
                cycles++; if (cycles > 2000) $fatal(1, "weight timeout");
            end
            sent = 0;
            while (sent < 75) begin
                @(negedge clk); lfsr = {lfsr[30:0], lfsr[31]^lfsr[21]^lfsr[1]^lfsr[0]};
                if (!activation_valid && lfsr[3]) begin activation_valid = 1; activation_data = 8'(activations[sent]); activation_last = sent == 74; end
                #1; source_fire = activation_valid && activation_ready; @(posedge clk); #1;
                if (source_fire) begin sent++; activation_valid = 0; end
                cycles++; if (cycles > 4000) $fatal(1, "activation timeout");
            end
            received = 0;
            while (received < expected_transfers) begin
                @(negedge clk); lfsr = {lfsr[30:0], lfsr[31]^lfsr[21]^lfsr[1]^lfsr[0]};
                output_ready = lfsr[4] | lfsr[7]; #1; sink_fire = output_valid && output_ready;
                if ((arithmetic_trace_file != 0) && (case_index == 0) &&
                    dut.arithmetic_core.postprocess.rounded_valid_q) begin
                    logical_sequence = int'(dut.output_channel) * 9 +
                        int'(dut.output_y) * 3 + int'(dut.output_x);
                    $fdisplay(arithmetic_trace_file,
                        "%0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                        logical_sequence, simulation_cycle,
                        $signed(dut.arithmetic_core.postprocess.trace_accumulator_q),
                        $signed(dut.arithmetic_core.postprocess.trace_bias_q),
                        $signed(dut.arithmetic_core.postprocess.trace_biased_q),
                        $signed(dut.arithmetic_core.postprocess.trace_multiplier_q),
                        dut.arithmetic_core.postprocess.trace_shift_q,
                        $signed(dut.arithmetic_core.postprocess.product_q),
                        $signed(dut.arithmetic_core.postprocess.rounded_negative_q ?
                            -dut.arithmetic_core.postprocess.rounded_magnitude_q :
                            dut.arithmetic_core.postprocess.rounded_magnitude_q),
                        $signed(dut.arithmetic_core.postprocess.next_result));
                end
                if (sink_fire) begin
                    logical_sequence = int'(dut.output_channel) * 9 +
                        int'(dut.output_y) * 3 + int'(dut.output_x);
                    if (!allow_mismatch && output_data !== 8'(expected[logical_sequence]))
                        $fatal(1, "case %0d output %0d: got %0d expected %0d", case_index, logical_sequence, output_data, expected[logical_sequence]);
                    if (!allow_mismatch && output_last !== (logical_sequence == 17))
                        $fatal(1, "TLAST mismatch");
                end
                @(posedge clk); #1;
                simulation_cycle++;
                if (sink_fire) begin
                    if ((trace_file != 0) && (case_index == 0))
                        $fdisplay(trace_file, "%0d %0d %0d 1 1 %0d", logical_sequence, simulation_cycle, $signed(output_data), output_last);
                    received++;
                end
                cycles++; if (cycles > 8000) $fatal(1, "output timeout");
            end
            if (!done || busy || overflow) $fatal(1, "completion state mismatch");
        end
        $fclose(file);
        if (trace_file != 0) $fclose(trace_file);
        if (arithmetic_trace_file != 0) $fclose(arithmetic_trace_file);
        $display("Convolution engine passed %0d randomized layers", case_count);
        $finish;
    end
endmodule

`default_nettype wire
