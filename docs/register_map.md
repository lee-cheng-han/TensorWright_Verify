# Register map draft

The planned AXI4-Lite interface reserves these word-aligned offsets. This is a draft,
not an implemented hardware interface.

| Offset | Register | Purpose |
| --- | --- | --- |
| `0x000` | `DEVICE_ID` | Accelerator identifier |
| `0x004` | `VERSION` | Interface version |
| `0x008` | `CONTROL` | Start, reset, clear error |
| `0x00C` | `STATUS` | Busy, done, error |
| `0x010` | `IRQ_STATUS` | Interrupt status |
| `0x014` | `IRQ_ENABLE` | Interrupt enable |
| `0x020` | `INPUT_HEIGHT` | Input height |
| `0x024` | `INPUT_WIDTH` | Input width |
| `0x028` | `INPUT_CHANNELS` | Input channels |
| `0x02C` | `OUTPUT_CHANNELS` | Output channels |
| `0x030` | `KERNEL_CONFIG` | Kernel, stride, padding |
| `0x034` | `OUTPUT_HEIGHT` | Output height |
| `0x038` | `OUTPUT_WIDTH` | Output width |
| `0x03C` | `FLAGS` | ReLU and feature flags |
| `0x040` | `INPUT_LENGTH` | Expected input bytes |
| `0x044` | `WEIGHT_LENGTH` | Expected weight bytes |
| `0x048` | `OUTPUT_LENGTH` | Expected output bytes |
| `0x050` | `CYCLE_COUNT_LOW` | Total cycles, low word |
| `0x054` | `CYCLE_COUNT_HIGH` | Total cycles, high word |
| `0x058` | `COMPUTE_ACTIVE` | Useful compute cycles |
| `0x05C` | `INPUT_STALLS` | Input-stall cycles |
| `0x060` | `OUTPUT_STALLS` | Output-stall cycles |
| `0x064` | `ERROR_CODE` | Last error |
| `0x068` | `WEIGHT_LOAD_CYCLES` | Weight-loading cycles |
| `0x06C` | `OUTPUT_COUNT` | Produced output values |
| `0x070` | `INPUT_COUNT` | Consumed input values |
| `0x074` | `LAYER_INVOCATIONS` | Accepted layer commands |
| `0x078` | `EXECUTED_MACS_LOW` | Executed MACs, low word |
| `0x07C` | `EXECUTED_MACS_HIGH` | Executed MACs, high word |
| `0x080` | `ERROR_COUNT` | Detected command errors |

Before RTL implementation, each register must define reset value, access type, bit
fields, illegal values, busy-time write behavior, write-one-to-clear behavior, counter
overflow, software ownership, and recovery. Unspecified offsets must return a defined
bus response rather than aliasing a register.

This same logical register model must be used by cocotb simulation and future AXI4-Lite
hardware access. Milestone 7 must freeze `DEVICE_ID`, interface-version compatibility,
start/completion semantics, timeout-visible state, and every counter's enable, reset,
saturation or wrap behavior. Simulation-only shortcuts must not add hidden control
paths. The simulation runtime will read counters through register transactions, not
direct hierarchical RTL access.
