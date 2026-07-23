`timescale 1ns/1ps
`default_nettype none

module tensorwright_stream_fifo #(
    parameter int DATA_WIDTH = 8,
    parameter int DEPTH = 16,
    localparam int POINTER_WIDTH = (DEPTH <= 2) ? 1 : $clog2(DEPTH),
    localparam int COUNT_WIDTH = $clog2(DEPTH + 1)
) (
    input  wire logic                  clk_i,
    input  wire logic                  rst_ni,
    input  wire logic                  s_tvalid_i,
    output logic                  s_tready_o,
    input  wire logic [DATA_WIDTH-1:0] s_tdata_i,
    input  wire logic                  s_tlast_i,
    output logic                  m_tvalid_o,
    input  wire logic                  m_tready_i,
    output logic [DATA_WIDTH-1:0] m_tdata_o,
    output logic                  m_tlast_o,
    output logic [COUNT_WIDTH-1:0] count_o
);
    logic [DATA_WIDTH-1:0] data_memory [0:DEPTH-1];
    logic                  last_memory [0:DEPTH-1];
    logic [POINTER_WIDTH-1:0] write_pointer;
    logic [POINTER_WIDTH-1:0] read_pointer;
    logic write_transfer;
    logic read_transfer;
    localparam logic [COUNT_WIDTH-1:0] DEPTH_COUNT = COUNT_WIDTH'(DEPTH);
    localparam logic [POINTER_WIDTH-1:0] LAST_POINTER = POINTER_WIDTH'(DEPTH - 1);

    assign s_tready_o = count_o < DEPTH_COUNT;
    assign m_tvalid_o = count_o != 0;
    assign m_tdata_o = data_memory[read_pointer];
    assign m_tlast_o = last_memory[read_pointer];
    assign write_transfer = s_tvalid_i && s_tready_o;
    assign read_transfer = m_tvalid_o && m_tready_i;

    always_ff @(posedge clk_i) begin
        if (!rst_ni) begin
            write_pointer <= '0;
            read_pointer  <= '0;
            count_o       <= '0;
        end else begin
            if (write_transfer) begin
                data_memory[write_pointer] <= s_tdata_i;
                last_memory[write_pointer] <= s_tlast_i;
                if (write_pointer == LAST_POINTER) begin
                    write_pointer <= '0;
                end else begin
                    write_pointer <= write_pointer + 1'b1;
                end
            end
            if (read_transfer) begin
                if (read_pointer == LAST_POINTER) begin
                    read_pointer <= '0;
                end else begin
                    read_pointer <= read_pointer + 1'b1;
                end
            end
            case ({write_transfer, read_transfer})
                2'b10: count_o <= count_o + 1'b1;
                2'b01: count_o <= count_o - 1'b1;
                default: count_o <= count_o;
            endcase
        end
    end

    initial begin
        assert (DEPTH > 0);
    end

`ifndef SYNTHESIS
    logic stalled_q;
    logic [DATA_WIDTH-1:0] stalled_data_q;
    logic stalled_last_q;
    always_ff @(posedge clk_i) begin
        if (!rst_ni) begin
            stalled_q <= 1'b0;
        end else begin
            if (stalled_q) begin
                assert (m_tvalid_o);
                assert (m_tdata_o == stalled_data_q);
                assert (m_tlast_o == stalled_last_q);
            end
            assert (count_o <= DEPTH_COUNT);
            if (s_tvalid_i) begin
                assert (!$isunknown({s_tdata_i, s_tlast_i}));
            end
            stalled_q      <= m_tvalid_o && !m_tready_i;
            stalled_data_q <= m_tdata_o;
            stalled_last_q <= m_tlast_o;
        end
    end
`endif
endmodule

`default_nettype wire
