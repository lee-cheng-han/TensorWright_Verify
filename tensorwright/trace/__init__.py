"""Canonical TensorWright Verify trace interfaces."""

from tensorwright.trace.compare import (
    AlignmentError,
    ComparisonReport,
    Divergence,
    compare_trace_files,
)
from tensorwright.trace.diagnosis import (
    DIAGNOSIS_RULESET_VERSION,
    Diagnosis,
    DiagnosisReport,
    diagnose_comparison,
)
from tensorwright.trace.plugins import (
    ADAPTER_API_VERSION,
    ENTRY_POINT_GROUP,
    AdapterDescriptor,
    AdapterError,
    AdapterRegistry,
    AdapterRequest,
    TraceAdapter,
    default_adapter_registry,
)
from tensorwright.trace.protocol import (
    PROTOCOL_RULESET_VERSION,
    ProtocolFinding,
    ProtocolReport,
    analyze_protocol_files,
)
from tensorwright.trace.reference import write_reference_trace
from tensorwright.trace.schema import (
    TRACE_VERSION,
    QuantizationMetadata,
    TraceError,
    TraceEvent,
    TraceSet,
    read_trace,
    write_trace,
)

__all__ = [
    "TRACE_VERSION",
    "ADAPTER_API_VERSION",
    "ENTRY_POINT_GROUP",
    "AdapterDescriptor",
    "AdapterError",
    "AdapterRegistry",
    "AdapterRequest",
    "AlignmentError",
    "ComparisonReport",
    "DIAGNOSIS_RULESET_VERSION",
    "Diagnosis",
    "DiagnosisReport",
    "Divergence",
    "QuantizationMetadata",
    "PROTOCOL_RULESET_VERSION",
    "ProtocolFinding",
    "ProtocolReport",
    "TraceError",
    "TraceEvent",
    "TraceSet",
    "TraceAdapter",
    "read_trace",
    "compare_trace_files",
    "diagnose_comparison",
    "default_adapter_registry",
    "analyze_protocol_files",
    "write_reference_trace",
    "write_trace",
]
