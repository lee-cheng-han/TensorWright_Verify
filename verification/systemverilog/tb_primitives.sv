`default_nettype none

module tb_primitives;
    logic clk_i = 1'b0;
    logic rst_ni = 1'b0;
    logic valid_i = 1'b0;
    logic clear_i = 1'b0;
    logic signed [7:0] activation_i;
    logic signed [7:0] weight_i;
    logic signed [15:0] product_o;
    logic mac_valid;
    logic signed [31:0] mac_accumulator;
    logic mac_overflow;
    logic signed [31:0] partial_sum_i;
    logic channel_valid;
    logic signed [31:0] channel_accumulator;
    logic channel_overflow;
    logic signed [8:0][15:0] tree_values;
    logic signed [31:0] tree_sum;
    integer activation;
    integer weight;
    integer expected;
    integer lane;

    /* verilator lint_off BLKSEQ */
    always #5 clk_i = ~clk_i;
    /* verilator lint_on BLKSEQ */

    tensorwright_multiplier multiplier (.*);
    tensorwright_mac mac (
        .clk_i,
        .rst_ni,
        .valid_i,
        .clear_i,
        .activation_i,
        .weight_i,
        .valid_o(mac_valid),
        .accumulator_o(mac_accumulator),
        .overflow_o(mac_overflow)
    );
    tensorwright_channel_accumulator channel_accumulator_unit (
        .clk_i,
        .rst_ni,
        .valid_i,
        .clear_i,
        .partial_sum_i,
        .valid_o(channel_valid),
        .accumulator_o(channel_accumulator),
        .overflow_o(channel_overflow)
    );
    tensorwright_adder_tree adder_tree (
        .values_i(tree_values),
        .sum_o(tree_sum)
    );

    initial begin
        for (activation = -128; activation <= 127; activation = activation + 1) begin
            for (weight = -128; weight <= 127; weight = weight + 1) begin
                activation_i = activation[7:0];
                weight_i = weight[7:0];
                #1;
                expected = activation * weight;
                assert ($signed(product_o) == expected) else
                    $fatal(1, "multiply expected %0d actual %0d", expected, product_o);
            end
        end

        expected = 0;
        for (lane = 0; lane < 9; lane = lane + 1) begin
            tree_values[lane] = (lane - 4) * 100;
            expected = expected + ((lane - 4) * 100);
        end
        #1;
        assert ($signed(tree_sum) == expected) else $fatal(1, "adder tree mismatch");

        repeat (2) @(posedge clk_i);
        rst_ni = 1'b1;
        activation_i = 8'sd127;
        weight_i = 8'sd127;
        partial_sum_i = 32'sd100;
        clear_i = 1'b1;
        valid_i = 1'b1;
        @(posedge clk_i);
        #1;
        assert (mac_valid && mac_accumulator == 16129 && !mac_overflow);
        assert (channel_valid && channel_accumulator == 100 && !channel_overflow);

        activation_i = 8'sh80;
        weight_i = 8'sh80;
        partial_sum_i = -32'sd40;
        clear_i = 1'b0;
        @(posedge clk_i);
        #1;
        assert (mac_valid && mac_accumulator == 32513 && !mac_overflow);
        assert (channel_valid && channel_accumulator == 60 && !channel_overflow);

        valid_i = 1'b0;
        $display("PASS primitives multiplier_vectors=65536");
        $finish;
    end
endmodule

`default_nettype wire
