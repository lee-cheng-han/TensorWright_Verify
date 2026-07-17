`timescale 1ns/1ps
`default_nettype none

module tb_control;
    import tensorwright_registers_pkg::*;
    logic clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    logic [11:0] awaddr, araddr; logic awvalid, awready, wvalid, wready, bvalid, bready;
    logic [31:0] wdata, rdata; logic [3:0] wstrb; logic [1:0] bresp, rresp;
    logic arvalid, arready, rvalid, rready;
    logic engine_done, compute_active, input_valid, input_ready, weight_valid, weight_ready;
    logic output_valid, output_ready; logic [31:0] macs;
    logic start, soft_reset, irq;
    logic saw_soft_reset = 0;
    logic [31:0] ih, iw, ic, oc, kc, oh, ow, flags, ilen, wlen, olen;
    always_ff @(posedge clk) if (soft_reset) saw_soft_reset <= 1;

    tensorwright_control dut (
        .clk_i(clk), .rst_ni(rst_n),
        .s_axil_awaddr_i(awaddr), .s_axil_awvalid_i(awvalid), .s_axil_awready_o(awready),
        .s_axil_wdata_i(wdata), .s_axil_wstrb_i(wstrb), .s_axil_wvalid_i(wvalid), .s_axil_wready_o(wready),
        .s_axil_bresp_o(bresp), .s_axil_bvalid_o(bvalid), .s_axil_bready_i(bready),
        .s_axil_araddr_i(araddr), .s_axil_arvalid_i(arvalid), .s_axil_arready_o(arready),
        .s_axil_rdata_o(rdata), .s_axil_rresp_o(rresp), .s_axil_rvalid_o(rvalid), .s_axil_rready_i(rready),
        .engine_done_i(engine_done), .compute_active_i(compute_active),
        .input_tvalid_i(input_valid), .input_tready_i(input_ready),
        .weight_tvalid_i(weight_valid), .weight_tready_i(weight_ready),
        .output_tvalid_i(output_valid), .output_tready_i(output_ready), .macs_executed_i(macs),
        .start_o(start), .soft_reset_o(soft_reset), .irq_o(irq),
        .input_height_o(ih), .input_width_o(iw), .input_channels_o(ic), .output_channels_o(oc),
        .kernel_config_o(kc), .output_height_o(oh), .output_width_o(ow), .flags_o(flags),
        .input_length_o(ilen), .weight_length_o(wlen), .output_length_o(olen)
    );

    task automatic axil_write(input logic [11:0] address, input logic [31:0] data,
                              input logic [1:0] expected_response);
        begin
            @(negedge clk); awaddr = address; awvalid = 1; wdata = data; wstrb = 4'hf; wvalid = 1;
            do @(posedge clk); while (!(awready && wready));
            #1; awvalid = 0; wvalid = 0;
            while (!bvalid) @(posedge clk);
            if (bresp !== expected_response) $fatal(1, "write response at %03x", address);
            @(negedge clk); bready = 1; @(posedge clk); #1; bready = 0;
        end
    endtask

    task automatic axil_read(input logic [11:0] address, input logic [31:0] expected,
                             input logic [1:0] expected_response);
        begin
            @(negedge clk); araddr = address; arvalid = 1;
            do @(posedge clk); while (!arready);
            #1; arvalid = 0;
            while (!rvalid) @(posedge clk);
            if (rresp !== expected_response || rdata !== expected)
                $fatal(1, "read mismatch at %03x: got %08x/%0d", address, rdata, rresp);
            @(negedge clk); rready = 1; @(posedge clk); #1; rready = 0;
        end
    endtask

    task automatic activity_cycle(input logic cv, iv, ir, wv, wr, ov, or_, done_i,
                                  input logic [31:0] mac_count);
        begin
            @(negedge clk); compute_active = cv; input_valid = iv; input_ready = ir;
            weight_valid = wv; weight_ready = wr; output_valid = ov; output_ready = or_;
            engine_done = done_i; macs = mac_count; @(posedge clk); #1;
            compute_active = 0; input_valid = 0; input_ready = 0; weight_valid = 0; weight_ready = 0;
            output_valid = 0; output_ready = 0; engine_done = 0; macs = 0;
        end
    endtask

    initial begin
        awaddr = 0; awvalid = 0; wdata = 0; wstrb = 0; wvalid = 0; bready = 0;
        araddr = 0; arvalid = 0; rready = 0; engine_done = 0; compute_active = 0;
        input_valid = 0; input_ready = 0; weight_valid = 0; weight_ready = 0;
        output_valid = 0; output_ready = 0; macs = 0;
        repeat (3) @(posedge clk); rst_n = 1; @(posedge clk);
        axil_read(ADDR_DEVICE_ID, DEVICE_ID, 0); axil_read(ADDR_VERSION, INTERFACE_VERSION, 0);
        axil_read(12'hffc, 0, 3);

        axil_write(ADDR_CONTROL, 1, 0);
        axil_read(ADDR_ERROR_CODE, ERROR_INVALID_CONFIG, 0);
        axil_write(ADDR_CONTROL, 4, 0);
        axil_write(ADDR_IRQ_ENABLE, 3, 0);
        axil_write(ADDR_INPUT_HEIGHT, 5, 0); axil_write(ADDR_INPUT_WIDTH, 5, 0);
        axil_write(ADDR_INPUT_CHANNELS, 1, 0); axil_write(ADDR_OUTPUT_CHANNELS, 1, 0);
        axil_write(ADDR_KERNEL_CONFIG, 32'h0000_1133, 0);
        axil_write(ADDR_OUTPUT_HEIGHT, 3, 0); axil_write(ADDR_OUTPUT_WIDTH, 3, 0);
        axil_write(ADDR_INPUT_LENGTH, 2, 0); axil_write(ADDR_WEIGHT_LENGTH, 1, 0);
        axil_write(ADDR_OUTPUT_LENGTH, 3, 0); axil_write(ADDR_CONTROL, 1, 0);
        axil_read(ADDR_STATUS, 1, 0);
        axil_write(ADDR_INPUT_HEIGHT, 7, 2);
        axil_read(ADDR_ERROR_CODE, ERROR_BUSY_WRITE, 0);
        axil_write(ADDR_CONTROL, 4, 0);
        axil_read(ADDR_INPUT_HEIGHT, 5, 0);

        activity_cycle(1, 1, 1, 1, 1, 0, 0, 0, 9);
        activity_cycle(1, 1, 0, 0, 0, 1, 0, 0, 9);
        activity_cycle(1, 0, 0, 0, 0, 1, 1, 0, 9);
        activity_cycle(0, 0, 0, 0, 0, 1, 1, 0, 0);
        activity_cycle(1, 0, 0, 0, 0, 1, 1, 1, 9);
        axil_read(ADDR_STATUS, 2, 0); axil_read(ADDR_COMPUTE_ACTIVE, 4, 0);
        axil_read(ADDR_INPUT_STALLS, 1, 0); axil_read(ADDR_OUTPUT_STALLS, 1, 0);
        axil_read(ADDR_WEIGHT_LOAD_CYCLES, 1, 0); axil_read(ADDR_INPUT_COUNT, 1, 0);
        axil_read(ADDR_OUTPUT_COUNT, 3, 0); axil_read(ADDR_EXECUTED_MACS_LOW, 36, 0);
        axil_read(ADDR_LAYER_INVOCATIONS, 1, 0);
        if (!irq) $fatal(1, "completion interrupt missing");
        axil_write(ADDR_IRQ_STATUS, 1, 0);
        if (irq) $fatal(1, "completion interrupt did not clear");

        axil_write(ADDR_CONTROL, 1, 0);
        activity_cycle(0, 0, 0, 0, 0, 0, 0, 1, 0);
        axil_read(ADDR_ERROR_CODE, ERROR_EARLY_COMPLETION, 0);
        axil_read(ADDR_ERROR_COUNT, 3, 0);
        if (!irq) $fatal(1, "error interrupt missing");
        axil_write(12'hffc, 0, 3);
        axil_write(ADDR_CONTROL, 2, 0);
        if (!saw_soft_reset) $fatal(1, "soft reset pulse missing");
        $display("Control tests passed: configuration, errors, IRQs, counters");
        $finish;
    end
endmodule

`default_nettype wire
