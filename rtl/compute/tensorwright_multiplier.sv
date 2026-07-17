`default_nettype none

module tensorwright_multiplier #(
    parameter int DATA_WIDTH = 8,
    parameter int PRODUCT_WIDTH = 2 * DATA_WIDTH
) (
    input  logic signed [DATA_WIDTH-1:0]    activation_i,
    input  logic signed [DATA_WIDTH-1:0]    weight_i,
    output logic signed [PRODUCT_WIDTH-1:0] product_o
);
    always_comb begin
        product_o = activation_i * weight_i;
    end
endmodule

`default_nettype wire
