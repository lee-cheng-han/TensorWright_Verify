# Tensor layout

The ONNX frontend and initial compiler IR use NCHW ordering: batch, channel, height,
width. Batch size is one for the MVP. A tensor's layout is explicit metadata and layout
changes must be represented by compiler or deployment operations; they may not be
hidden in packing code.

The bundle's activation, weight, and stream layouts remain draft interfaces until the
buffering architecture is selected. Before RTL streaming begins, this document must
define byte order, channel interleaving, weight order, padding representation, AXI
Stream beat order, `tlast` placement, and output ordering. Reference vectors will use
the exact same definitions.
