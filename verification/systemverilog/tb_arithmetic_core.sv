`default_nettype none

module tb_arithmetic_core;
    localparam int LANES = 9;
    logic clk_i = 1'b0;
    logic rst_ni = 1'b0;
    logic valid_i = 1'b0;
    logic clear_i;
    logic last_i;
    logic signed [LANES-1:0][7:0] activations_i;
    logic signed [LANES-1:0][7:0] weights_i;
    logic signed [31:0] bias_i;
    logic [30:0] multiplier_i;
    logic [6:0] shift_i;
    logic relu_i;
    logic valid_o;
    logic signed [7:0] result_o;
    logic overflow_o;

    integer vectors;
    integer status;
    integer cycle_count;
    integer expected;
    integer relu_value;
    integer activation_value;
    integer weight_value;
    integer vector_count = 0;
    integer cycle;
    integer lane;
    string vector_file;

    always #5 clk_i = ~clk_i;

    tensorwright_arithmetic_core #(.LANES(LANES)) dut (.*);

    initial begin
        if (!$value$plusargs("VECTOR_FILE=%s", vector_file)) begin
            $fatal(1, "VECTOR_FILE plusarg is required");
        end
        vectors = $fopen(vector_file, "r");
        if (vectors == 0) begin
            $fatal(1, "could not open %s", vector_file);
        end
        repeat (2) @(posedge clk_i);
        rst_ni = 1'b1;

        while (!$feof(vectors)) begin
            status = $fscanf(
                vectors,
                "%d %d %d %d %d %d\n",
                cycle_count,
                bias_i,
                multiplier_i,
                shift_i,
                relu_value,
                expected
            );
            if (status == 6) begin
                relu_i = relu_value[0];
                for (cycle = 0; cycle < cycle_count; cycle = cycle + 1) begin
                    for (lane = 0; lane < LANES; lane = lane + 1) begin
                        status = $fscanf(
                            vectors,
                            "%d %d",
                            activation_value,
                            weight_value
                        );
                        assert (status == 2) else $fatal(1, "truncated vector file");
                        activations_i[lane] = activation_value[7:0];
                        weights_i[lane] = weight_value[7:0];
                    end
                    status = $fscanf(vectors, "\n");
                    clear_i = cycle == 0;
                    last_i = cycle == cycle_count - 1;
                    valid_i = 1'b1;
                    @(posedge clk_i);
                    #1;
                end
                valid_i = 1'b0;
                while (!valid_o) begin
                    @(posedge clk_i);
                    #1;
                end
                assert (valid_o) else
                    $fatal(1, "missing valid at vector %0d", vector_count);
                assert (!overflow_o) else
                    $fatal(1, "overflow at vector %0d", vector_count);
                assert ($signed(result_o) == expected) else
                    $fatal(
                        1,
                        "vector %0d expected %0d actual %0d",
                        vector_count,
                        expected,
                        $signed(result_o)
                    );
                vector_count = vector_count + 1;
            end
        end
        valid_i = 1'b0;
        $display("PASS arithmetic_core vectors=%0d", vector_count);
        $finish;
    end
endmodule

`default_nettype wire
