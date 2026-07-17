`timescale 1ns/1ps
`default_nettype none

module tb_streaming;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic av_s_valid, av_s_ready, av_s_last;
    logic [7:0] av_s_data;
    logic av_m_valid, av_m_ready, av_m_last;
    logic [7:0] av_m_data;
    logic [3:0] av_count;

    logic wt_clear, wt_s_valid, wt_s_ready, wt_s_last;
    logic [7:0] wt_s_data, wt_read_data;
    logic [3:0] wt_read_address;
    logic wt_read_valid, wt_loaded;
    logic [4:0] wt_count;

    logic win_s_valid, win_s_ready, win_s_last;
    logic [7:0] win_s_data;
    logic win_m_valid, win_m_ready, win_m_last;
    logic [71:0] win_m_data;

    tensorwright_activation_buffer #(.DATA_WIDTH(8), .DEPTH(8)) activation_buffer (
        .clk_i(clk), .rst_ni(rst_n), .s_tvalid_i(av_s_valid), .s_tready_o(av_s_ready),
        .s_tdata_i(av_s_data), .s_tlast_i(av_s_last), .m_tvalid_o(av_m_valid),
        .m_tready_i(av_m_ready), .m_tdata_o(av_m_data), .m_tlast_o(av_m_last),
        .count_o(av_count)
    );

    tensorwright_weight_buffer #(.DATA_WIDTH(8), .DEPTH(16)) weight_buffer (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(wt_clear), .s_tvalid_i(wt_s_valid),
        .s_tready_o(wt_s_ready), .s_tdata_i(wt_s_data), .s_tlast_i(wt_s_last),
        .read_address_i(wt_read_address), .read_data_o(wt_read_data),
        .read_valid_o(wt_read_valid), .loaded_o(wt_loaded), .count_o(wt_count)
    );

    tensorwright_window_generator_3x3 #(.DATA_WIDTH(8), .IMAGE_WIDTH(5), .IMAGE_HEIGHT(5)) window_generator (
        .clk_i(clk), .rst_ni(rst_n), .s_tvalid_i(win_s_valid), .s_tready_o(win_s_ready),
        .s_tdata_i(win_s_data), .s_tlast_i(win_s_last), .m_tvalid_o(win_m_valid),
        .m_tready_i(win_m_ready), .m_tdata_o(win_m_data), .m_tlast_o(win_m_last)
    );

    task automatic reset_dut;
        begin
            av_s_valid = 0; av_s_data = 0; av_s_last = 0; av_m_ready = 0;
            wt_clear = 0; wt_s_valid = 0; wt_s_data = 0; wt_s_last = 0; wt_read_address = 0;
            win_s_valid = 0; win_s_data = 0; win_s_last = 0; win_m_ready = 0;
            rst_n = 0;
            repeat (3) @(posedge clk);
            rst_n = 1;
            @(posedge clk);
        end
    endtask

    initial begin : test_sequence
        integer sent, received, cycles, window_index, lane, output_row, output_column;
        logic [31:0] lfsr;
        logic [7:0] expected;
        logic source_fire, sink_fire;
        reset_dut();

        // The source holds each item until accepted; the sink applies seeded stalls.
        sent = 0; received = 0; cycles = 0; lfsr = 32'h6a09e667;
        while (received < 40) begin
            @(negedge clk);
            lfsr = {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
            if (!av_s_valid && sent < 40 && lfsr[2]) begin
                av_s_valid = 1; av_s_data = 8'(sent); av_s_last = sent == 39;
            end
            av_m_ready = lfsr[5] | lfsr[9];
            #1; source_fire = av_s_valid && av_s_ready; sink_fire = av_m_valid && av_m_ready;
            if (sink_fire) begin
                if (av_m_data !== 8'(received) || av_m_last !== (received == 39))
                    $fatal(1, "activation FIFO mismatch at %0d", received);
            end
            @(posedge clk);
            #1;
            if (source_fire) begin sent++; av_s_valid = 0; end
            if (sink_fire) received++;
            cycles++;
            if (cycles > 2000) $fatal(1, "activation FIFO timeout");
        end

        // Load a complete weight packet, then verify random-access contents.
        av_m_ready = 0; sent = 0; cycles = 0;
        while (!wt_loaded) begin
            @(negedge clk);
            lfsr = {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
            if (!wt_s_valid && sent < 16 && lfsr[3]) begin
                wt_s_valid = 1; wt_s_data = 8'(8'h80 + sent); wt_s_last = sent == 15;
            end
            #1; source_fire = wt_s_valid && wt_s_ready;
            @(posedge clk);
            #1;
            if (source_fire) begin sent++; wt_s_valid = 0; end
            cycles++;
            if (cycles > 1000) $fatal(1, "weight buffer timeout");
        end
        if (wt_count != 16) $fatal(1, "weight count mismatch");
        for (sent = 0; sent < 16; sent++) begin
            wt_read_address = 4'(sent); #1;
            if (!wt_read_valid || wt_read_data !== 8'(8'h80 + sent))
                $fatal(1, "weight read mismatch at %0d", sent);
        end

        // A 5x5 raster produces nine valid, unpadded 3x3 windows.
        sent = 0; window_index = 0; cycles = 0;
        while (window_index < 9) begin
            @(negedge clk);
            lfsr = {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
            if (!win_s_valid && sent < 25 && lfsr[4]) begin
                win_s_valid = 1; win_s_data = 8'(sent); win_s_last = sent == 24;
            end
            win_m_ready = lfsr[6] | lfsr[11];
            #1; source_fire = win_s_valid && win_s_ready; sink_fire = win_m_valid && win_m_ready;
            if (sink_fire) begin
                output_row = window_index / 3;
                output_column = window_index % 3;
                for (lane = 0; lane < 9; lane++) begin
                    expected = 8'((output_row + lane / 3) * 5 + output_column + lane % 3);
                    if (win_m_data[(lane*8)+:8] !== expected)
                        $fatal(1, "window %0d lane %0d mismatch", window_index, lane);
                end
                if (win_m_last !== (window_index == 8)) $fatal(1, "window TLAST mismatch");
            end
            @(posedge clk);
            #1;
            if (source_fire) begin sent++; win_s_valid = 0; end
            if (sink_fire) window_index++;
            cycles++;
            if (cycles > 2000) $fatal(1, "window generator timeout");
        end
        $display("Streaming tests passed: fifo_items=40 weights=16 windows=9");
        $finish;
    end
endmodule

`default_nettype wire
