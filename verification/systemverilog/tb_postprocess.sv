`default_nettype none

module tb_postprocess;
    logic clk_i = 1'b0;
    logic rst_ni = 1'b0;
    logic valid_i = 1'b0;
    logic signed [31:0] accumulator_i;
    logic signed [31:0] bias_i;
    logic [30:0] multiplier_i;
    logic [6:0] shift_i;
    logic relu_i;
    logic valid_o;
    logic signed [7:0] result_o;
    logic overflow_o;

    integer vectors;
    integer status;
    integer expected;
    integer relu_value;
    integer accumulator_value;
    integer bias_value;
    integer multiplier_value;
    integer shift_value;
    integer count = 0;
    string vector_file;

    always #5 clk_i = ~clk_i;

    tensorwright_postprocess dut (.*);

    initial begin
        if (!$value$plusargs("VECTOR_FILE=%s", vector_file)) begin
            $fatal(1, "VECTOR_FILE plusarg is required");
        end
        vectors = $fopen(vector_file, "r");
        if (vectors == 0) begin
            $fatal(1, "could not open %s", vector_file);
        end
        // Drive reset and transaction inputs away from the DUT sampling edge.
        // Changing them on a positive edge creates an event-ordering race whose
        // outcome differs between Verilator releases.
        repeat (2) @(posedge clk_i);
        @(negedge clk_i);
        rst_ni = 1'b1;

        while (!$feof(vectors)) begin
            status = $fscanf(
                vectors,
                "%d %d %d %d %d %d\n",
                accumulator_value,
                bias_value,
                multiplier_value,
                shift_value,
                relu_value,
                expected
            );
            if (status == 6) begin
                @(negedge clk_i);
                accumulator_i = accumulator_value[31:0];
                bias_i = bias_value[31:0];
                multiplier_i = multiplier_value[30:0];
                shift_i = shift_value[6:0];
                relu_i = relu_value[0];
                valid_i = 1'b1;
                @(posedge clk_i);
                @(negedge clk_i);
                valid_i = 1'b0;
                while (!valid_o) begin
                    @(posedge clk_i);
                    #1;
                end
                assert (valid_o) else $fatal(1, "missing valid at vector %0d", count);
                assert (!overflow_o) else $fatal(1, "overflow at vector %0d", count);
                assert ($signed(result_o) == expected) else
                    $fatal(
                        1,
                        "vector %0d expected %0d actual %0d",
                        count,
                        expected,
                        $signed(result_o)
                    );
                count = count + 1;
                @(posedge clk_i);
                #1;
            end
        end
        valid_i = 1'b0;
        $display("PASS postprocess vectors=%0d", count);
        $finish;
    end
endmodule

`default_nettype wire
