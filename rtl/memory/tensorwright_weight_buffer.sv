`timescale 1ns/1ps
`default_nettype none

module tensorwright_weight_buffer #(
    parameter int DATA_WIDTH = 8,
    parameter int DEPTH = 64,
    localparam int ADDRESS_WIDTH = (DEPTH <= 2) ? 1 : $clog2(DEPTH),
    localparam int COUNT_WIDTH = $clog2(DEPTH + 1)
) (
    input  logic                     clk_i,
    input  logic                     rst_ni,
    input  logic                     clear_i,
    input  logic                     s_tvalid_i,
    output logic                     s_tready_o,
    input  logic [DATA_WIDTH-1:0]    s_tdata_i,
    input  logic                     s_tlast_i,
    input  logic [ADDRESS_WIDTH-1:0] read_address_i,
    output logic [DATA_WIDTH-1:0]    read_data_o,
    output logic                     read_valid_o,
    output logic                     loaded_o,
    output logic [COUNT_WIDTH-1:0]   count_o
);
    logic [DATA_WIDTH-1:0] memory [0:DEPTH-1];
    logic write_transfer;
    localparam logic [COUNT_WIDTH-1:0] DEPTH_COUNT = COUNT_WIDTH'(DEPTH);
    localparam logic [COUNT_WIDTH-1:0] LAST_COUNT = COUNT_WIDTH'(DEPTH - 1);

    assign s_tready_o = !loaded_o && count_o < DEPTH_COUNT;
    assign write_transfer = s_tvalid_i && s_tready_o;
    assign read_valid_o = COUNT_WIDTH'(read_address_i) < count_o;
    assign read_data_o = memory[read_address_i];

    always_ff @(posedge clk_i) begin
        if (!rst_ni || clear_i) begin
            count_o  <= '0;
            loaded_o <= 1'b0;
        end else if (write_transfer) begin
            memory[count_o[ADDRESS_WIDTH-1:0]] <= s_tdata_i;
            count_o <= count_o + 1'b1;
            if (s_tlast_i || count_o == LAST_COUNT) begin
                loaded_o <= 1'b1;
            end
        end
    end

    initial begin
        assert (DEPTH > 0);
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk_i) begin
        if (rst_ni && !clear_i) begin
            assert (count_o <= DEPTH_COUNT);
            if (s_tvalid_i) begin
                assert (!$isunknown({s_tdata_i, s_tlast_i}));
            end
        end
    end
`endif
endmodule

`default_nettype wire
