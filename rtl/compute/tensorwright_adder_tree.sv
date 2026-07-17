`default_nettype none

module tensorwright_adder_tree #(
    parameter int LANES = 9,
    parameter int INPUT_WIDTH = 16,
    parameter int OUTPUT_WIDTH = 32
) (
    input  logic signed [LANES-1:0][INPUT_WIDTH-1:0] values_i,
    output logic signed [OUTPUT_WIDTH-1:0]           sum_o
);
    integer lane;
    always_comb begin
        sum_o = '0;
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            sum_o = sum_o + {
                {(OUTPUT_WIDTH-INPUT_WIDTH){values_i[lane][INPUT_WIDTH-1]}},
                values_i[lane]
            };
        end
    end

    initial begin
        assert (LANES > 0);
        assert (OUTPUT_WIDTH >= INPUT_WIDTH);
    end
endmodule

`default_nettype wire
