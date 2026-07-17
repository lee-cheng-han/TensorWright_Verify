# Register and control contract

Milestone 7 freezes interface version `1.0` and the logical AXI4-Lite register model
used by both simulation and the future board runtime. The bus is 32 bits wide with
12-bit byte addresses. Registers are word aligned and little-endian. Reads and writes
to unspecified addresses return AXI `DECERR`; writes to read-only registers also return
`DECERR`. Configuration writes while busy return `SLVERR` and record `BUSY_WRITE`.

| Offset | Register | Access | Reset | Meaning |
| --- | --- | --- | --- | --- |
| `0x000` | `DEVICE_ID` | RO | `0x54570001` | TensorWright accelerator identity |
| `0x004` | `VERSION` | RO | `0x00010000` | Major 1, minor 0 interface |
| `0x008` | `CONTROL` | WO | 0 | Bit 0 start, bit 1 soft reset, bit 2 clear error |
| `0x00C` | `STATUS` | RO | 0 | Bit 0 busy, bit 1 done, bit 2 error |
| `0x010` | `IRQ_STATUS` | RW1C | 0 | Bit 0 completion, bit 1 error |
| `0x014` | `IRQ_ENABLE` | RW | 0 | Enables corresponding IRQ status bits |
| `0x020` | `INPUT_HEIGHT` | RW | 0 | Nonzero input rows |
| `0x024` | `INPUT_WIDTH` | RW | 0 | Nonzero input columns |
| `0x028` | `INPUT_CHANNELS` | RW | 0 | Nonzero input channels |
| `0x02C` | `OUTPUT_CHANNELS` | RW | 0 | Nonzero output channels |
| `0x030` | `KERNEL_CONFIG` | RW | 0 | `[3:0]` KW, `[7:4]` KH, `[11:8]` SW, `[15:12]` SH, `[23:16]` padding |
| `0x034` | `OUTPUT_HEIGHT` | RW | 0 | Nonzero output rows |
| `0x038` | `OUTPUT_WIDTH` | RW | 0 | Nonzero output columns |
| `0x03C` | `FLAGS` | RW | 0 | Bit 0 enables ReLU; other bits reserved |
| `0x040` | `INPUT_LENGTH` | RW | 0 | Expected input transfers |
| `0x044` | `WEIGHT_LENGTH` | RW | 0 | Expected weight transfers |
| `0x048` | `OUTPUT_LENGTH` | RW | 0 | Required output transfers before completion |
| `0x050` | `CYCLE_COUNT_LOW` | RO | 0 | Busy-cycle counter, low word |
| `0x054` | `CYCLE_COUNT_HIGH` | RO | 0 | Busy-cycle counter, high word |
| `0x058` | `COMPUTE_ACTIVE` | RO | 0 | Cycles with compute active |
| `0x05C` | `INPUT_STALLS` | RO | 0 | Busy cycles with input valid and not ready |
| `0x060` | `OUTPUT_STALLS` | RO | 0 | Busy cycles with output valid and not ready |
| `0x064` | `ERROR_CODE` | RO | 0 | Most recent control error |
| `0x068` | `WEIGHT_LOAD_CYCLES` | RO | 0 | Accepted weight transfers |
| `0x06C` | `OUTPUT_COUNT` | RO | 0 | Accepted output transfers |
| `0x070` | `INPUT_COUNT` | RO | 0 | Accepted input transfers |
| `0x074` | `LAYER_INVOCATIONS` | RO | 0 | Accepted starts since hard reset |
| `0x078` | `EXECUTED_MACS_LOW` | RO | 0 | Executed MAC count, low word |
| `0x07C` | `EXECUTED_MACS_HIGH` | RO | 0 | Executed MAC count, high word |
| `0x080` | `ERROR_COUNT` | RO | 0 | Recorded errors since hard reset |

Only the M8-supported `3x3`, stride-one, zero-padding kernel encoding is currently a
legal start configuration. Dimensions and transfer lengths must be nonzero. Unsupported
flag bits are reserved and software must write them as zero.

## Lifecycle and recovery

An accepted start pulses `start_o`, clears per-run counters and stale done/error/IRQ
state, increments `LAYER_INVOCATIONS`, and enters busy. Configuration is immutable
while busy. `engine_done_i` produces successful completion only after the configured
number of output transfers; otherwise it records `EARLY_COMPLETION`. Done and IRQ
status are sticky until the next accepted start, soft reset, or explicit W1C operation.

Soft reset aborts a run, pulses `soft_reset_o`, and clears busy, status, IRQs, errors,
and per-run counters. It preserves configuration and lifetime invocation/error counts.
Clear-error removes error status, code, and error IRQ without changing configuration or
other counters. A start while busy records an error without aborting the current run.

The 64-bit cycle and MAC counters wrap modulo 2^64. All 32-bit activity, transfer,
invocation, and error counters saturate at `0xffffffff`. Counters advance only while
busy; transfer counters advance only on legal ready/valid handshakes. `irq_o` is the OR
of enabled sticky status bits.

Error codes are `0 NONE`, `1 INVALID_CONFIG`, `2 START_WHILE_BUSY`, `3 BUSY_WRITE`, and
`4 EARLY_COMPLETION`. An error never creates successful completion.
