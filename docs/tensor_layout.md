# Tensor layout

The ONNX frontend and initial compiler IR use NCHW ordering: batch, channel, height,
width. Batch size is one for the MVP. A tensor's layout is explicit metadata and layout
changes must be represented by compiler or deployment operations; they may not be
hidden in packing code.

RTL activation packets contain signed INT8 values in `[channel][row][column]` order,
with column varying fastest. Weight packets use
`[output channel][input channel][kernel row][kernel column]`, with kernel column varying
fastest. Output packets use `[output channel][row][column]`. Batch size is one and is
therefore not represented in a stream index.

Every INT8 element occupies one stream beat and retains its two's-complement byte
representation. There are no implicit padding beats: the current engine implements
valid, zero-padding-free 3x3 convolution. `tlast` is asserted only on the final weight,
activation, or output beat. Quantization metadata packing and multibyte bundle byte
order are frozen by Milestone 9; the M8 RTL exposes those values as typed ports.
