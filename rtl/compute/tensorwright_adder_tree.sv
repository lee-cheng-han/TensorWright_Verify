`timescale 1ns/1ps
`default_nettype none

module tensorwright_adder_tree #(
    parameter int LANES = 9,
    parameter int INPUT_WIDTH = 16,
    parameter int OUTPUT_WIDTH = 32
) (
    input  wire logic signed [LANES-1:0][INPUT_WIDTH-1:0] values_i,
    output logic signed [OUTPUT_WIDTH-1:0]           sum_o
);
    generate
        if (LANES == 9) begin : gen_balanced_nine
            logic signed [OUTPUT_WIDTH-1:0] extended [0:8];
            logic signed [OUTPUT_WIDTH-1:0] level_one [0:4];
            logic signed [OUTPUT_WIDTH-1:0] level_two [0:2];
            logic signed [OUTPUT_WIDTH-1:0] level_three [0:1];
            always_comb begin
                for (integer lane = 0; lane < 9; lane++) begin
                    extended[lane] = {
                        {(OUTPUT_WIDTH-INPUT_WIDTH){values_i[lane][INPUT_WIDTH-1]}},
                        values_i[lane]
                    };
                end
                level_one[0] = extended[0] + extended[1];
                level_one[1] = extended[2] + extended[3];
                level_one[2] = extended[4] + extended[5];
                level_one[3] = extended[6] + extended[7];
                level_one[4] = extended[8];
                level_two[0] = level_one[0] + level_one[1];
                level_two[1] = level_one[2] + level_one[3];
                level_two[2] = level_one[4];
                level_three[0] = level_two[0] + level_two[1];
                level_three[1] = level_two[2];
                sum_o = level_three[0] + level_three[1];
            end
        end else begin : gen_generic
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
        end
    endgenerate

    initial begin
        assert (LANES > 0);
        assert (OUTPUT_WIDTH >= INPUT_WIDTH);
    end
endmodule

`default_nettype wire
