`timescale 1ns/1ps
`default_nettype none

package tensorwright_registers_pkg;
    localparam logic [31:0] DEVICE_ID = 32'h5457_0001;
    localparam logic [31:0] INTERFACE_VERSION = 32'h0001_0000;

    localparam logic [11:0] ADDR_DEVICE_ID          = 12'h000;
    localparam logic [11:0] ADDR_VERSION            = 12'h004;
    localparam logic [11:0] ADDR_CONTROL            = 12'h008;
    localparam logic [11:0] ADDR_STATUS             = 12'h00c;
    localparam logic [11:0] ADDR_IRQ_STATUS         = 12'h010;
    localparam logic [11:0] ADDR_IRQ_ENABLE         = 12'h014;
    localparam logic [11:0] ADDR_INPUT_HEIGHT       = 12'h020;
    localparam logic [11:0] ADDR_INPUT_WIDTH        = 12'h024;
    localparam logic [11:0] ADDR_INPUT_CHANNELS     = 12'h028;
    localparam logic [11:0] ADDR_OUTPUT_CHANNELS    = 12'h02c;
    localparam logic [11:0] ADDR_KERNEL_CONFIG      = 12'h030;
    localparam logic [11:0] ADDR_OUTPUT_HEIGHT      = 12'h034;
    localparam logic [11:0] ADDR_OUTPUT_WIDTH       = 12'h038;
    localparam logic [11:0] ADDR_FLAGS              = 12'h03c;
    localparam logic [11:0] ADDR_INPUT_LENGTH       = 12'h040;
    localparam logic [11:0] ADDR_WEIGHT_LENGTH      = 12'h044;
    localparam logic [11:0] ADDR_OUTPUT_LENGTH      = 12'h048;
    localparam logic [11:0] ADDR_CYCLE_COUNT_LOW    = 12'h050;
    localparam logic [11:0] ADDR_CYCLE_COUNT_HIGH   = 12'h054;
    localparam logic [11:0] ADDR_COMPUTE_ACTIVE     = 12'h058;
    localparam logic [11:0] ADDR_INPUT_STALLS       = 12'h05c;
    localparam logic [11:0] ADDR_OUTPUT_STALLS      = 12'h060;
    localparam logic [11:0] ADDR_ERROR_CODE         = 12'h064;
    localparam logic [11:0] ADDR_WEIGHT_LOAD_CYCLES = 12'h068;
    localparam logic [11:0] ADDR_OUTPUT_COUNT       = 12'h06c;
    localparam logic [11:0] ADDR_INPUT_COUNT        = 12'h070;
    localparam logic [11:0] ADDR_LAYER_INVOCATIONS  = 12'h074;
    localparam logic [11:0] ADDR_EXECUTED_MACS_LOW  = 12'h078;
    localparam logic [11:0] ADDR_EXECUTED_MACS_HIGH = 12'h07c;
    localparam logic [11:0] ADDR_ERROR_COUNT        = 12'h080;

    localparam logic [31:0] ERROR_NONE            = 32'd0;
    localparam logic [31:0] ERROR_INVALID_CONFIG  = 32'd1;
    localparam logic [31:0] ERROR_START_WHILE_BUSY = 32'd2;
    localparam logic [31:0] ERROR_BUSY_WRITE       = 32'd3;
    localparam logic [31:0] ERROR_EARLY_COMPLETION = 32'd4;
endpackage

`default_nettype wire
