`timescale 1ns/1ps
`default_nettype none

module tensorwright_window_generator_3x3 #(
    parameter int DATA_WIDTH = 8,
    parameter int IMAGE_WIDTH = 5,
    parameter int IMAGE_HEIGHT = 5,
    localparam int COLUMN_WIDTH = (IMAGE_WIDTH <= 2) ? 1 : $clog2(IMAGE_WIDTH),
    localparam int ROW_WIDTH = (IMAGE_HEIGHT <= 2) ? 1 : $clog2(IMAGE_HEIGHT)
) (
    input  logic                       clk_i,
    input  logic                       rst_ni,
    input  logic                       s_tvalid_i,
    output logic                       s_tready_o,
    input  logic [DATA_WIDTH-1:0]      s_tdata_i,
    input  logic                       s_tlast_i,
    output logic                       m_tvalid_o,
    input  logic                       m_tready_i,
    output logic [(9*DATA_WIDTH)-1:0]  m_tdata_o,
    output logic                       m_tlast_o
);
    logic [DATA_WIDTH-1:0] previous_row [0:IMAGE_WIDTH-1];
    logic [DATA_WIDTH-1:0] two_rows_back [0:IMAGE_WIDTH-1];
    logic [DATA_WIDTH-1:0] top_left;
    logic [DATA_WIDTH-1:0] top_middle;
    logic [DATA_WIDTH-1:0] middle_left;
    logic [DATA_WIDTH-1:0] middle_middle;
    logic [DATA_WIDTH-1:0] bottom_left;
    logic [DATA_WIDTH-1:0] bottom_middle;
    logic [COLUMN_WIDTH-1:0] column;
    logic [ROW_WIDTH-1:0] row;
    logic input_transfer;
    logic produces_window;
    localparam logic [COLUMN_WIDTH-1:0] LAST_COLUMN = COLUMN_WIDTH'(IMAGE_WIDTH - 1);
    localparam logic [ROW_WIDTH-1:0] LAST_ROW = ROW_WIDTH'(IMAGE_HEIGHT - 1);

    assign s_tready_o = !m_tvalid_o || m_tready_i;
    assign input_transfer = s_tvalid_i && s_tready_o;
    assign produces_window = row >= 2 && column >= 2;

    always_ff @(posedge clk_i) begin
        if (!rst_ni) begin
            m_tvalid_o   <= 1'b0;
            m_tdata_o    <= '0;
            m_tlast_o    <= 1'b0;
            column       <= '0;
            row          <= '0;
            top_left     <= '0;
            top_middle   <= '0;
            middle_left  <= '0;
            middle_middle <= '0;
            bottom_left  <= '0;
            bottom_middle <= '0;
        end else begin
            if (m_tvalid_o && m_tready_i) begin
                m_tvalid_o <= 1'b0;
            end
            if (input_transfer) begin
                if (produces_window) begin
                    m_tdata_o[(0*DATA_WIDTH)+:DATA_WIDTH] <= top_left;
                    m_tdata_o[(1*DATA_WIDTH)+:DATA_WIDTH] <= top_middle;
                    m_tdata_o[(2*DATA_WIDTH)+:DATA_WIDTH] <= two_rows_back[column];
                    m_tdata_o[(3*DATA_WIDTH)+:DATA_WIDTH] <= middle_left;
                    m_tdata_o[(4*DATA_WIDTH)+:DATA_WIDTH] <= middle_middle;
                    m_tdata_o[(5*DATA_WIDTH)+:DATA_WIDTH] <= previous_row[column];
                    m_tdata_o[(6*DATA_WIDTH)+:DATA_WIDTH] <= bottom_left;
                    m_tdata_o[(7*DATA_WIDTH)+:DATA_WIDTH] <= bottom_middle;
                    m_tdata_o[(8*DATA_WIDTH)+:DATA_WIDTH] <= s_tdata_i;
                    m_tvalid_o <= 1'b1;
                    m_tlast_o <= row == LAST_ROW && column == LAST_COLUMN;
                end

                two_rows_back[column] <= previous_row[column];
                previous_row[column] <= s_tdata_i;
                top_left <= top_middle;
                top_middle <= two_rows_back[column];
                middle_left <= middle_middle;
                middle_middle <= previous_row[column];
                bottom_left <= bottom_middle;
                bottom_middle <= s_tdata_i;

                if (column == LAST_COLUMN) begin
                    column <= '0;
                    if (row == LAST_ROW) begin
                        row <= '0;
                    end else begin
                        row <= row + 1'b1;
                    end
                    top_left <= '0;
                    top_middle <= '0;
                    middle_left <= '0;
                    middle_middle <= '0;
                    bottom_left <= '0;
                    bottom_middle <= '0;
                end else begin
                    column <= column + 1'b1;
                end
            end
        end
    end

    initial begin
        assert (IMAGE_WIDTH >= 3);
        assert (IMAGE_HEIGHT >= 3);
    end

`ifndef SYNTHESIS
    logic stalled_q;
    logic [(9*DATA_WIDTH)-1:0] stalled_data_q;
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
            if (input_transfer) begin
                assert (s_tlast_i == (row == LAST_ROW && column == LAST_COLUMN));
            end
            stalled_q      <= m_tvalid_o && !m_tready_i;
            stalled_data_q <= m_tdata_o;
            stalled_last_q <= m_tlast_o;
        end
    end
`endif
endmodule

`default_nettype wire
