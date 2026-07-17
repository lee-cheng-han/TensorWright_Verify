# Streaming and buffering contract

Milestone 6 uses an AXI4-Stream-style subset: `tvalid`, `tready`, `tdata`, and
`tlast`. A transfer occurs only on a rising clock edge where both `tvalid` and
`tready` are high. A producer must hold `tvalid`, `tdata`, and `tlast` stable
while stalled. Reset is synchronous, active-low, and clears all visible valid
and occupancy state.

`tensorwright_stream_fifo` is a parameterized synchronous FIFO that preserves
packet boundaries through its stored `tlast` bit. The current implementation
deasserts input ready while full; it deliberately does not provide a combinational
full-FIFO bypass. `tensorwright_activation_buffer` gives that FIFO a domain-specific
name without changing the protocol.

`tensorwright_weight_buffer` accepts one packet into addressable local storage.
It asserts `loaded_o` on an accepted `tlast` or when its configured depth is
filled, then rejects further writes until `clear_i`. Reads are combinational and
`read_valid_o` distinguishes initialized packet addresses.

## 3x3 window layout

The window generator accepts one single-channel image in raster order. It emits
valid, stride-one, unpadded 3x3 windows after the first two rows and columns.
Lane zero occupies the least-significant bits:

```text
lane 0  lane 1  lane 2
lane 3  lane 4  lane 5
lane 6  lane 7  lane 8
```

`m_tlast_o` marks the bottom-right window. When its output is stalled, the entire
input and line-buffer state freezes, so no pixels or windows are lost. Image width
and height are elaboration-time parameters and must each be at least three.

This milestone does not implement padding, stride other than one, multiple input
channels, convolution accumulation, DMA, or an integrated accelerator. Those
features are introduced only through later scheduling and convolution milestones.
