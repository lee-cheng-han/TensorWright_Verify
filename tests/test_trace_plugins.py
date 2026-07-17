from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tensorwright.cli import main
from tensorwright.trace import (
    ADAPTER_API_VERSION,
    AdapterDescriptor,
    AdapterError,
    AdapterRegistry,
    AdapterRequest,
    read_trace,
    write_trace,
)
from tensorwright.trace.plugins import default_adapter_registry
from tests.test_trace_comparison import _event


class CustomAdapter:
    descriptor = AdapterDescriptor(
        "custom.example",
        "1.2.3",
        ADAPTER_API_VERSION,
        ("example-v1",),
        ("operation_output",),
        "Test adapter.",
    )

    def convert(self, request: AdapterRequest) -> Path:
        event = _event(3, [0, 0, 0, 0], backend=self.descriptor.name)
        return write_trace(request.destination, [event])


class TracePluginTest(unittest.TestCase):
    def test_registry_is_sorted_and_rejects_duplicates(self) -> None:
        registry = AdapterRegistry()
        registry.register(CustomAdapter())
        self.assertEqual(registry.names(), ["custom.example"])
        self.assertIsInstance(registry.get("custom.example"), CustomAdapter)
        with self.assertRaisesRegex(AdapterError, "Duplicate"):
            registry.register(CustomAdapter())
        with self.assertRaisesRegex(AdapterError, "Unknown"):
            registry.get("custom.missing")

    def test_descriptor_contract_is_strict(self) -> None:
        invalid = [
            AdapterDescriptor(
                "bad", "1.0.0", 1, ("x",), ("operation_output",), "bad name"
            ),
            AdapterDescriptor(
                "custom.bad", "v1", 1, ("x",), ("operation_output",), "bad version"
            ),
            AdapterDescriptor(
                "custom.bad", "1.0.0", 99, ("x",), ("operation_output",), "bad API"
            ),
            AdapterDescriptor(
                "custom.bad", "1.0.0", 1, ("x",), ("unknown",), "bad point"
            ),
        ]
        for descriptor in invalid:
            with self.subTest(descriptor=descriptor), self.assertRaises(AdapterError):
                descriptor.validate()

    def test_conversion_revalidates_backend_identity(self) -> None:
        registry = AdapterRegistry()
        registry.register(CustomAdapter())
        with tempfile.TemporaryDirectory() as directory:
            output = registry.convert(
                "custom.example",
                AdapterRequest(
                    Path(directory) / "unused", Path(directory) / "trace.jsonl", {}
                ),
            )
            event = read_trace(output).events[0]
        self.assertEqual(event.source_backend, "custom.example")

        class WrongBackendAdapter(CustomAdapter):
            descriptor = AdapterDescriptor(
                "custom.wrong", "1.0.0", 1, ("x",), ("operation_output",), "wrong"
            )

            def convert(self, request: AdapterRequest) -> Path:
                return write_trace(
                    request.destination,
                    [_event(3, [0, 0, 0, 0], backend="custom.example")],
                )

        registry = AdapterRegistry()
        registry.register(WrongBackendAdapter())
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(AdapterError, "emitted backend"),
        ):
            registry.convert(
                "custom.wrong",
                AdapterRequest(Path("unused"), Path(directory) / "trace.jsonl", {}),
            )

    def test_opt_in_entry_point_discovery(self) -> None:
        class FakeEntryPoint:
            name = "custom_example"

            @staticmethod
            def load() -> type[CustomAdapter]:
                return CustomAdapter

        registry = AdapterRegistry()
        with patch(
            "tensorwright.trace.plugins.metadata.entry_points",
            return_value=[FakeEntryPoint()],
        ) as entry_points:
            registry.discover()
        entry_points.assert_called_once_with(group="tensorwright.trace_adapters")
        self.assertEqual(registry.names(), ["custom.example"])

    def test_builtin_verilator_adapter_converts_transfer_log(self) -> None:
        options = {
            "run_id": "run_1",
            "model_id": "tiny_conv",
            "source_operation_id": "onnx:Conv_0",
            "compiled_operation_id": "compiled:op_0000",
            "operation_name": "conv_0",
            "tensor_name": "output",
            "shape": [1, 1, 1, 2],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "transfers.txt"
            source.write_text("0 10 -3 1 1 0\n1 14 4 1 1 1\n", encoding="utf-8")
            output = default_adapter_registry().convert(
                "tensorwright.verilator_rtl",
                AdapterRequest(source, root / "trace.jsonl", options),
            )
            events = read_trace(output).events
        self.assertEqual([event.value for event in events], [-3, 4])
        self.assertEqual([event.cycle for event in events], [10, 14])

    def test_builtin_finn_adapter_converts_full_execution_context(self) -> None:
        options = {
            "run_id": "finn_run_1",
            "model_id": "finn_tiny",
            "graph_stage": "post_streamlining",
            "scalar_event_limit": 4,
            "tensors": [
                {
                    "tensor_name": "global_out",
                    "source_operation_id": "qonnx:MultiThreshold_0",
                    "compiled_operation_id": "finn:StreamingMaxPool_0",
                    "fused_source_operation_ids": ["qonnx:Relu_0"],
                    "operation_name": "StreamingMaxPool_0",
                    "operation_type": "StreamingMaxPool",
                    "layout": "NHWC",
                },
                {
                    "tensor_name": "small_out",
                    "source_operation_id": "qonnx:Add_1",
                    "compiled_operation_id": "finn:AddStreams_1",
                    "operation_name": "AddStreams_1",
                    "operation_type": "AddStreams",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "verify_step_1_SUCCESS.npz"
            np.savez(
                source,
                global_out=np.arange(6, dtype=np.float32).reshape(1, 2, 3, 1),
                small_out=np.array([[7, -2]], dtype=np.int8),
            )
            output = default_adapter_registry().convert(
                "finn.dataflow",
                AdapterRequest(source, root / "trace.jsonl", options),
            )
            events = read_trace(output).events
            payload = np.load(root / str(events[0].data_file), allow_pickle=False)
        self.assertEqual(events[0].event_type, "tensor_chunk")
        self.assertEqual(events[0].source_operation_id, "qonnx:MultiThreshold_0")
        self.assertEqual(events[0].fused_source_operation_ids, ["qonnx:Relu_0"])
        self.assertEqual(
            payload.tolist(),
            [[[[0.0], [1.0], [2.0]], [[3.0], [4.0], [5.0]]]],
        )
        self.assertEqual([event.value for event in events[1:]], [7, -2])

    def test_finn_adapter_rejects_missing_and_duplicate_mappings(self) -> None:
        base = {
            "tensor_name": "out",
            "source_operation_id": "qonnx:Add_0",
            "compiled_operation_id": "finn:AddStreams_0",
            "operation_name": "AddStreams_0",
            "operation_type": "AddStreams",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "context.npz"
            np.savez(source, out=np.array([1], dtype=np.int8))
            request = AdapterRequest(
                source,
                root / "trace.jsonl",
                {
                    "run_id": "run",
                    "model_id": "model",
                    "tensors": [base, dict(base)],
                },
            )
            with self.assertRaisesRegex(AdapterError, "Duplicate"):
                default_adapter_registry().convert("finn.dataflow", request)
            missing = dict(base, tensor_name="absent")
            with self.assertRaisesRegex(AdapterError, "has no tensor"):
                default_adapter_registry().convert(
                    "finn.dataflow",
                    AdapterRequest(
                        source,
                        root / "missing.jsonl",
                        {"run_id": "run", "model_id": "model", "tensors": [missing]},
                    ),
                )

    def test_builtin_hls4ml_adapter_converts_csim_layer_trace(self) -> None:
        options = {
            "run_id": "hls4ml_run_1",
            "model_id": "hls4ml_tiny",
            "tensors": [
                {
                    "tensor_name": "dense",
                    "source_operation_id": "keras:dense",
                    "compiled_operation_id": "hls4ml:dense",
                    "operation_name": "dense",
                    "operation_type": "Dense",
                    "layout": "NC",
                },
                {
                    "tensor_name": "relu",
                    "source_operation_id": "keras:relu",
                    "compiled_operation_id": "hls4ml:relu",
                    "operation_name": "relu",
                    "operation_type": "Activation",
                    "layout": "NC",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hls4ml_trace.npz"
            np.savez(
                source,
                dense=np.array([[12.0, 15.0]], dtype=np.float64),
                relu=np.array([[12.0, 15.0]], dtype=np.float64),
            )
            output = default_adapter_registry().convert(
                "hls4ml.csim",
                AdapterRequest(source, root / "trace.jsonl", options),
            )
            events = read_trace(output).events
        self.assertEqual([event.value for event in events], [12.0, 15.0] * 2)
        self.assertTrue(
            all(event.source_backend == "hls4ml.csim" for event in events)
        )
        self.assertTrue(
            all(event.hardware_stage == "hls4ml_layer_output" for event in events)
        )
        self.assertEqual(events[0].metadata, {"hls4ml_trace_key": "dense"})

    def test_cli_lists_and_converts_with_options_file(self) -> None:
        listing = StringIO()
        with redirect_stdout(listing):
            status = main(["trace", "adapters"])
        self.assertEqual(status, 0)
        self.assertIn("tensorwright.verilator_rtl 1.0.0", listing.getvalue())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "transfers.txt"
            source.write_text("0 10 3 1 1 1\n", encoding="utf-8")
            options = root / "options.json"
            options.write_text(
                json.dumps(
                    {
                        "run_id": "run_1",
                        "model_id": "model",
                        "source_operation_id": "onnx:Conv_0",
                        "compiled_operation_id": "compiled:op_0000",
                        "operation_name": "conv_0",
                        "tensor_name": "output",
                        "shape": [1, 1, 1, 1],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "trace.jsonl"
            stdout = StringIO()
            with redirect_stdout(stdout):
                status = main(
                    [
                        "trace",
                        "convert",
                        str(source),
                        str(output),
                        "--adapter",
                        "tensorwright.verilator_rtl",
                        "--options",
                        f"@{options}",
                    ]
                )
        self.assertEqual(status, 0)
        self.assertIn("Generated canonical trace", stdout.getvalue())

    def test_cli_reports_adapter_errors(self) -> None:
        error = StringIO()
        with redirect_stderr(error):
            status = main(
                [
                    "trace",
                    "convert",
                    "missing",
                    "output.jsonl",
                    "--adapter",
                    "custom.missing",
                ]
            )
        self.assertEqual(status, 1)
        self.assertIn("trace conversion failed", error.getvalue())


if __name__ == "__main__":
    unittest.main()
