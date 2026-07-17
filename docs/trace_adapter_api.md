# Trace-adapter plugin API version 1

Milestone 19 defines the extension boundary that converts backend-specific artifacts into
TensorWright canonical trace schema version 2.

## CLI

List maintained built-ins:

```bash
tensorwright trace adapters
```

Installed third-party entry points are loaded only when explicitly requested:

```bash
tensorwright trace adapters --discover
tensorwright trace convert source.log output.jsonl \
  --adapter custom.my_backend \
  --options @options.json \
  --discover
```

The built-in `tensorwright.verilator_rtl` adapter accepts TensorWright compact transfer-log v1.
Its required options are `run_id`, `model_id`, `source_operation_id`,
`compiled_operation_id`, `operation_name`, `tensor_name`, and `shape`. Optional provenance,
layout, dtype, operation type, and hardware-stage fields have documented TensorWright defaults.
Unknown options are rejected.

## Python contract

An adapter exposes an `AdapterDescriptor` and synchronous `convert(AdapterRequest) -> Path`:

```python
from tensorwright.trace import (
    ADAPTER_API_VERSION,
    AdapterDescriptor,
    AdapterRequest,
)

class MyAdapter:
    descriptor = AdapterDescriptor(
        name="custom.my_backend",
        version="1.0.0",
        api_version=ADAPTER_API_VERSION,
        input_formats=("my-trace-v1",),
        trace_points=("operation_output",),
        description="Convert my backend trace format.",
    )

    def convert(self, request: AdapterRequest):
        # Parse request.source and write canonical JSONL to request.destination.
        return request.destination
```

Distributions register a class or instance under this entry-point group:

```toml
[project.entry-points."tensorwright.trace_adapters"]
my_backend = "my_package.adapter:MyAdapter"
```

Names must be extensible dotted backend identifiers, versions use `MAJOR.MINOR.PATCH`, and the
declared API version must match TensorWright. Input formats and canonical trace points cannot be
empty. Registries reject duplicate names. After conversion TensorWright reads and validates the
entire canonical trace and requires `source_backend` to equal the registered adapter name.

Entry-point discovery executes installed Python code, so it is opt-in in the CLI. An adapter
failure names its entry point or registered backend and does not leave the output accepted as a
valid conversion. A backend name alone does not imply support; a source becomes supported
only after real, backend-tested adapters exist.
