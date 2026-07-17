"""Self-contained HTML debugging dashboard generation."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensorwright.trace import (
    ComparisonReport,
    DiagnosisReport,
    ProtocolReport,
    analyze_protocol_files,
    compare_trace_files,
    diagnose_comparison,
)

DASHBOARD_FORMAT_VERSION = 1


class DashboardError(RuntimeError):
    """Raised when dashboard inputs or output are invalid."""


@dataclass(frozen=True)
class DashboardResult:
    """Generated dashboard identity and summary."""

    path: Path
    comparison: ComparisonReport
    diagnosis: DiagnosisReport
    protocol: ProtocolReport


def generate_dashboard(
    reference_trace: str | Path,
    candidate_trace: str | Path,
    output_path: str | Path,
    *,
    minimization_report: str | Path | None = None,
    regression_manifest: str | Path | None = None,
) -> DashboardResult:
    """Generate one deterministic, offline HTML investigation report."""
    output = Path(output_path)
    if output.suffix.lower() != ".html":
        raise DashboardError("Dashboard output must use the .html extension")
    comparison = compare_trace_files(reference_trace, candidate_trace)
    diagnosis = diagnose_comparison(comparison)
    protocol = analyze_protocol_files(reference_trace, candidate_trace)
    minimization = _optional_json(minimization_report, "minimization report")
    regression = _optional_json(regression_manifest, "regression manifest")
    document = _render(comparison, diagnosis, protocol, minimization, regression)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return DashboardResult(output, comparison, diagnosis, protocol)


def _optional_json(path: str | Path | None, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DashboardError(f"Invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise DashboardError(f"Invalid {label}: root must be a JSON object")
    return value


def _render(
    comparison: ComparisonReport,
    diagnosis: DiagnosisReport,
    protocol: ProtocolReport,
    minimization: dict[str, Any] | None,
    regression: dict[str, Any] | None,
) -> str:
    status = "MATCH" if comparison.matched else "DIVERGENCE"
    status_class = "pass" if comparison.matched else "fail"
    protocol_status = "PASS" if protocol.protocol_ok else "FAIL"
    protocol_class = "pass" if protocol.protocol_ok else "fail"
    divergence = comparison.first_divergence
    if divergence is None:
        divergence_panel = "<p>No divergent value was found.</p>"
    else:
        divergence_panel = _definition_list(
            [
                ("Kind", divergence.kind),
                ("Operation", divergence.compiled_operation_id),
                ("Source operation", divergence.source_operation_id),
                ("Tensor", divergence.tensor_name),
                ("Trace point", divergence.trace_point),
                ("Coordinate", divergence.coordinate),
                ("Reference", divergence.reference_value),
                ("Candidate", divergence.candidate_value),
                ("Candidate cycle", divergence.candidate_cycle),
            ]
        )
    numerical = diagnosis.diagnosis
    if numerical is None:
        diagnosis_panel = "<p>No numerical diagnosis is required.</p>"
    else:
        diagnosis_panel = (
            _definition_list(
                [
                    ("Rule", numerical.rule_id),
                    ("Likely cause", numerical.title),
                    ("Confidence", numerical.confidence),
                ]
            )
            + _string_list("Evidence", numerical.evidence)
            + _string_list("Recommended checks", numerical.recommended_checks)
        )
    findings = "".join(
        "<tr>"
        f"<td>{_escape(item.severity)}</td>"
        f"<td><code>{_escape(item.rule_id)}</code></td>"
        f"<td>{_escape(item.event_index)}</td>"
        f"<td>{_escape(item.cycle)}</td>"
        f"<td>{_escape(item.evidence)}</td>"
        "</tr>"
        for item in protocol.findings
    )
    if not findings:
        findings = '<tr><td colspan="5">No protocol findings.</td></tr>'
    optional_sections = ""
    if minimization is not None:
        optional_sections += _json_section("Minimization", minimization)
    if regression is not None:
        optional_sections += _json_section("Generated regression", regression)
    raw_report = {
        "dashboard_format_version": DASHBOARD_FORMAT_VERSION,
        "comparison": comparison.to_dict(),
        "diagnosis": diagnosis.to_dict(),
        "protocol": protocol.to_dict(),
        "minimization": minimization,
        "regression": regression,
    }
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TensorWright Verify — {_escape(comparison.model_id)}</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ margin: 0 auto; max-width: 1100px; padding: 2rem; line-height: 1.45; }}
header, section {{
  border: 1px solid #7776; border-radius: .7rem;
  padding: 1rem 1.2rem; margin: 1rem 0;
}}
h1, h2 {{ margin-top: 0; }}
.summary {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: .8rem;
}}
.metric {{ background: #7771; border-radius: .5rem; padding: .8rem; }}
.metric strong {{ display: block; font-size: 1.35rem; }}
.pass {{ color: #17833b; }} .fail {{ color: #c0392b; }}
dl {{ display: grid; grid-template-columns: minmax(150px, 1fr) 3fr; gap: .35rem 1rem; }}
dt {{ font-weight: 700; }} dd {{ margin: 0; overflow-wrap: anywhere; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{
  border-bottom: 1px solid #7775; padding: .5rem;
  text-align: left; vertical-align: top;
}}
pre {{
  white-space: pre-wrap; overflow-wrap: anywhere;
  background: #7771; padding: 1rem; border-radius: .4rem;
}}
code {{ font-family: ui-monospace, monospace; }}
</style>
</head>
<body>
<header>
<h1>TensorWright Verify</h1>
<p>Offline debugging dashboard · format version {DASHBOARD_FORMAT_VERSION}</p>
<div class="summary">
<div class="metric"><span>Result</span>
<strong class="{status_class}">{status}</strong></div>
<div class="metric"><span>Matched values</span>
<strong>{comparison.matched_values}</strong></div>
<div class="metric"><span>Reference values</span>
<strong>{comparison.reference_values}</strong></div>
<div class="metric"><span>Candidate values</span>
<strong>{comparison.candidate_values}</strong></div>
<div class="metric"><span>Protocol</span>
<strong class="{protocol_class}">{protocol_status}</strong></div>
</div>
</header>
<section><h2>First divergence</h2>{divergence_panel}</section>
<section><h2>Numerical diagnosis</h2>{diagnosis_panel}</section>
<section><h2>Protocol findings</h2>
<table><thead><tr><th>Severity</th><th>Rule</th><th>Event</th><th>Cycle</th><th>Evidence</th></tr></thead>
<tbody>{findings}</tbody></table></section>
{optional_sections}
{_json_section("Complete machine-readable report", raw_report)}
</body>
</html>
"""


def _definition_list(items: list[tuple[str, object]]) -> str:
    return (
        "<dl>"
        + "".join(
            f"<dt>{_escape(label)}</dt><dd>{_escape(value)}</dd>"
            for label, value in items
        )
        + "</dl>"
    )


def _string_list(title: str, values: list[str]) -> str:
    return (
        f"<h3>{_escape(title)}</h3><ul>"
        + "".join(f"<li>{_escape(value)}</li>" for value in values)
        + "</ul>"
    )


def _json_section(title: str, value: dict[str, Any]) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True)
    return (
        f"<section><h2>{_escape(title)}</h2><details><summary>Show JSON</summary>"
        f"<pre>{_escape(payload)}</pre></details></section>"
    )


def _escape(value: object) -> str:
    if value is None:
        return "—"
    return html.escape(str(value), quote=True)
