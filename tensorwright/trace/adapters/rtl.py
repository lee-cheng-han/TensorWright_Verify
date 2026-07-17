"""Canonical adapter for accepted RTL streaming transfers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tensorwright.trace.schema import TRACE_VERSION, TraceEvent, write_trace


@dataclass(frozen=True)
class RtlTransfer:
    """A sampled AXI-stream-like output transfer."""

    sequence: int
    cycle: int
    value: int
    valid: bool
    ready: bool
    last: bool


class RtlTraceCapture:
    """Collect accepted output transfers without importing a simulator package."""

    def __init__(
        self,
        *,
        enabled: bool,
        run_id: str,
        model_id: str,
        source_operation_id: str,
        compiled_operation_id: str,
        operation_name: str,
        tensor_name: str,
        shape: list[int],
        fused_source_operation_ids: list[str] | None = None,
        graph_stage: str = "rtl_execution",
        operation_type: str = "Conv",
        hardware_stage: str = "convolution_output_stream",
        layout: str = "NCHW",
        dtype: str = "int8",
        source_backend: str = "tensorwright.cocotb_rtl",
    ) -> None:
        self.enabled = enabled
        self._identity = {
            "trace_version": TRACE_VERSION,
            "source_backend": source_backend,
            "run_id": run_id,
            "model_id": model_id,
            "source_operation_id": source_operation_id,
            "compiled_operation_id": compiled_operation_id,
            "fused_source_operation_ids": fused_source_operation_ids or [],
            "graph_stage": graph_stage,
            "operation_name": operation_name,
            "operation_type": operation_type,
            "hardware_stage": hardware_stage,
            "trace_point": "stream_transfer",
            "tensor_name": tensor_name,
            "shape": shape,
            "layout": layout,
            "dtype": dtype,
        }
        self.events: list[TraceEvent] = []

    def record(self, transfer: RtlTransfer) -> None:
        """Record a transfer only when the ready/valid handshake accepts it."""
        if not self.enabled or not (transfer.valid and transfer.ready):
            return
        expected = len(self.events)
        if transfer.sequence != expected:
            raise ValueError(f"RTL transfer sequence {transfer.sequence} != {expected}")
        shape = self._identity["shape"]
        assert isinstance(shape, list)
        coordinate = _unravel(transfer.sequence, shape)
        self.events.append(
            TraceEvent(
                event_type="scalar",
                coordinate=coordinate,
                value=transfer.value,
                cycle=transfer.cycle,
                metadata={
                    "valid": transfer.valid,
                    "ready": transfer.ready,
                    "tlast": transfer.last,
                    "sequence": transfer.sequence,
                },
                **self._identity,
            )
        )

    def write(self, path: str | Path) -> Path | None:
        """Write a canonical trace when capture was enabled and data was accepted."""
        if not self.enabled:
            return None
        if not self.events:
            raise ValueError("RTL trace capture contains no accepted transfers")
        expected_last = len(self.events) - 1
        for index, event in enumerate(self.events):
            if bool(event.metadata["tlast"]) != (index == expected_last):
                raise ValueError("RTL TLAST must identify exactly the final transfer")
        return write_trace(path, self.events)


def read_transfer_log(path: str | Path) -> list[RtlTransfer]:
    """Read the compact whitespace log emitted by the Verilator testbench."""
    transfers: list[RtlTransfer] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"Malformed RTL transfer log line {line_number}")
        sequence, cycle, value, valid, ready, last = (int(field) for field in fields)
        transfers.append(
            RtlTransfer(sequence, cycle, value, bool(valid), bool(ready), bool(last))
        )
    return transfers


def _unravel(index: int, shape: list[int]) -> list[int]:
    size = 1
    for dimension in shape:
        size *= dimension
    if index < 0 or index >= size:
        raise ValueError("RTL transfer sequence exceeds the declared tensor shape")
    coordinate = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        coordinate[axis] = index % shape[axis]
        index //= shape[axis]
    return coordinate
