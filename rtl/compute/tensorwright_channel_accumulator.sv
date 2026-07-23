`timescale 1ns/1ps
`default_nettype none

module tensorwright_channel_accumulator #(
    parameter int ACC_WIDTH = 32
) (
    input  wire logic                        clk_i,
    input  wire logic                        rst_ni,
    input  wire logic                        valid_i,
    input  wire logic                        clear_i,
    input  wire logic signed [ACC_WIDTH-1:0] partial_sum_i,
    output logic                        valid_o,
    output logic signed [ACC_WIDTH-1:0] accumulator_o,
    output logic                        overflow_o
);
    logic signed [ACC_WIDTH:0] next_accumulator;

    always_comb begin
        if (clear_i) begin
            next_accumulator = {partial_sum_i[ACC_WIDTH-1], partial_sum_i};
        end else begin
            next_accumulator = $signed(accumulator_o) + $signed(partial_sum_i);
        end
    end

    always_ff @(posedge clk_i) begin
        if (!rst_ni) begin
            valid_o       <= 1'b0;
            accumulator_o <= '0;
            overflow_o    <= 1'b0;
        end else begin
            valid_o    <= valid_i;
            overflow_o <= 1'b0;
            if (valid_i) begin
                accumulator_o <= next_accumulator[ACC_WIDTH-1:0];
                overflow_o <= next_accumulator[ACC_WIDTH] != next_accumulator[ACC_WIDTH-1];
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk_i) begin
        if (rst_ni && valid_i) begin
            assert (!$isunknown({clear_i, partial_sum_i}));
        end
    end
`endif
endmodule

`default_nettype wire
