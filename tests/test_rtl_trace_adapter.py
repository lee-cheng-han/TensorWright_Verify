import tempfile
import unittest
from pathlib import Path

from tensorwright.trace.adapters.rtl import (
    RtlTraceCapture,
    RtlTransfer,
    read_transfer_log,
)
from tensorwright.trace.schema import read_trace


def _capture(
    *, enabled: bool = True, shape: list[int] | None = None
) -> RtlTraceCapture:
    return RtlTraceCapture(
        enabled=enabled,
        run_id="run_rtl_1",
        model_id="tiny_conv",
        source_operation_id="onnx:Conv_0",
        compiled_operation_id="compiled:op_0000",
        operation_name="conv_0",
        tensor_name="output",
        shape=shape or [1, 1, 1, 2],
    )


class RtlTraceAdapterTest(unittest.TestCase):
    def test_capture_records_only_accepted_transfers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capture = _capture()
            capture.record(RtlTransfer(0, 10, 99, True, False, False))
            capture.record(RtlTransfer(0, 11, -7, True, True, False))
            capture.record(RtlTransfer(1, 15, 8, True, True, True))

            output = capture.write(Path(directory) / "rtl.jsonl")
            self.assertIsNotNone(output)
            assert output is not None
            events = read_trace(output).events
            self.assertEqual([event.value for event in events], [-7, 8])
            self.assertEqual(
                [event.coordinate for event in events],
                [[0, 0, 0, 0], [0, 0, 0, 1]],
            )
            self.assertEqual([event.cycle for event in events], [11, 15])
            self.assertEqual(
                events[1].metadata,
                {"valid": True, "ready": True, "tlast": True, "sequence": 1},
            )
            self.assertEqual(events[0].source_backend, "tensorwright.cocotb_rtl")
            self.assertEqual(events[0].trace_point, "stream_transfer")

    def test_disabled_capture_has_no_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capture = _capture(enabled=False)
            capture.record(RtlTransfer(0, 1, 1, True, True, True))
            destination = Path(directory) / "disabled.jsonl"
            self.assertIsNone(capture.write(destination))
            self.assertFalse(destination.exists())

    def test_capture_rejects_sequence_and_tlast_errors(self) -> None:
        capture = _capture()
        with self.assertRaisesRegex(ValueError, "sequence"):
            capture.record(RtlTransfer(1, 1, 1, True, True, True))

        capture.record(RtlTransfer(0, 1, 1, True, True, True))
        capture.record(RtlTransfer(1, 2, 2, True, True, True))
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "TLAST"),
        ):
            capture.write(Path(directory) / "bad.jsonl")

    def test_compact_verilator_log_is_converted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "transfers.txt"
            source.write_text("0 41 -3 1 1 0\n1 45 4 1 1 1\n", encoding="utf-8")
            self.assertEqual(
                read_transfer_log(source),
                [
                    RtlTransfer(0, 41, -3, True, True, False),
                    RtlTransfer(1, 45, 4, True, True, True),
                ],
            )

    def test_sequence_must_fit_declared_tensor(self) -> None:
        capture = _capture(shape=[1])
        capture.record(RtlTransfer(0, 1, 1, True, True, False))
        with self.assertRaisesRegex(ValueError, "shape"):
            capture.record(RtlTransfer(1, 2, 2, True, True, True))


if __name__ == "__main__":
    unittest.main()
