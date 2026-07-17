"""Self-contained HTML debugging dashboard generation."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tensorwright.trace import (
    ComparisonReport,
    DiagnosisReport,
    ProtocolReport,
    TraceEvent,
    analyze_protocol_files,
    compare_trace_files,
    diagnose_comparison,
    read_trace,
)

DASHBOARD_FORMAT_VERSION = 2
PREVIEW_CHARACTER_LIMIT = 12_000


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
    baseline_candidate_trace: str | Path | None = None,
    scenario_note: str | None = None,
    arithmetic_evidence: dict[str, Any] | None = None,
    generated_regression: str | Path | None = None,
) -> DashboardResult:
    """Generate one deterministic, offline HTML investigation report."""
    output = Path(output_path)
    if output.suffix.lower() != ".html":
        raise DashboardError("Dashboard output must use the .html extension")
    comparison = compare_trace_files(reference_trace, candidate_trace)
    diagnosis = diagnose_comparison(comparison)
    protocol = analyze_protocol_files(reference_trace, candidate_trace)
    baseline = (
        compare_trace_files(reference_trace, baseline_candidate_trace)
        if baseline_candidate_trace is not None
        else None
    )
    tensor_slice = _tensor_slice(reference_trace, candidate_trace, comparison)
    comparison_stats = _scalar_comparison_stats(reference_trace, candidate_trace)
    regression_source = _optional_text(generated_regression, "generated regression")
    minimization = _optional_json(minimization_report, "minimization report")
    regression = _optional_json(regression_manifest, "regression manifest")
    document = _render(
        comparison,
        diagnosis,
        protocol,
        minimization,
        regression,
        baseline,
        scenario_note,
        tensor_slice,
        arithmetic_evidence,
        regression_source,
        comparison_stats,
    )
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


def _optional_text(path: str | Path | None, label: str) -> tuple[str, str] | None:
    if path is None:
        return None
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            content = stream.read(PREVIEW_CHARACTER_LIMIT + 1)
        if len(content) > PREVIEW_CHARACTER_LIMIT:
            content = (
                content[:PREVIEW_CHARACTER_LIMIT]
                + "\n... preview truncated by TensorWright ...\n"
            )
        return str(source), content
    except OSError as error:
        raise DashboardError(f"Invalid {label}: {error}") from error


def _presentation_panel(
    comparison: ComparisonReport,
    baseline: ComparisonReport | None,
    scenario_note: str | None,
) -> str:
    if baseline is None and scenario_note is None:
        return ""
    scenario = ""
    if scenario_note is not None:
        scenario = (
            '<section class="scenario"><strong>Controlled demo fault</strong>'
            f"<span>{_escape(scenario_note)}</span></section>"
        )
    baseline_panel = ""
    if baseline is not None:
        baseline_status = "PASS" if baseline.matched else "UNEXPECTED DIVERGENCE"
        baseline_class = "pass" if baseline.matched else "fail"
        baseline_panel = (
            '<div class="baseline flow-node"><span>Known-good RTL baseline</span>'
            f'<strong class="{baseline_class}">{baseline_status}</strong>'
            f"<span>{baseline.matched_values}/{baseline.reference_values} values "
            "matched</span></div>"
        )
    divergence = comparison.first_divergence
    reveal = ""
    if divergence is not None:
        reveal = f"""
<div>
  <div class="value-reveal">
    <span><small class="value-label">SOFTWARE</small>
      {_escape(divergence.reference_value)}</span>
    <span class="arrow">→</span>
    <span><small class="value-label">RTL</small>
      {_escape(divergence.candidate_value)}</span>
  </div>
  <div class="fault-facts">
    <div class="fact"><span>Coordinate</span>
      <strong>{_escape(divergence.coordinate)}</strong></div>
    <div class="fact"><span>Accepted cycle</span>
      <strong>{_escape(divergence.candidate_cycle)}</strong></div>
    <div class="fact"><span>First mismatch</span>
      <strong>#{comparison.matched_values + 1}</strong></div>
  </div>
</div>"""
    return (
        f'{scenario}<section id="experiment">'
        "<h2>Clean baseline → controlled failure</h2>"
        f'<div class="reveal">{baseline_panel}{reveal}</div></section>'
    )


def _arithmetic_panel(evidence: dict[str, Any] | None) -> str:
    if evidence is None:
        return ""
    required = {
        "accumulator",
        "bias",
        "biased",
        "multiplier",
        "shift",
        "product",
        "rounding_offset",
        "software_result",
        "rtl_result",
    }
    if not required <= evidence.keys():
        raise DashboardError("Arithmetic evidence is missing required fields")
    difference = evidence["rtl_result"] - evidence["software_result"]
    return f"""
<section id="arithmetic-evidence">
<h2>Why this value differs</h2>
<p>The accumulator, bias, multiplier, and product are identical. The first error is
introduced when the faulty RTL omits the rounding offset before the right shift.</p>
<div class="arithmetic">
  <div><span>Accumulator</span><strong>{_escape(evidence['accumulator'])}</strong></div>
  <div class="arrow">+</div>
  <div><span>Bias</span><strong>{_escape(evidence['bias'])}</strong></div>
  <div class="arrow">=</div>
  <div><span>Post-bias</span><strong>{_escape(evidence['biased'])}</strong></div>
  <div class="arrow">× {_escape(evidence['multiplier'])}</div>
  <div><span>Product</span><strong>{_escape(evidence['product'])}</strong></div>
</div>
<div class="rounding-compare">
  <div class="pass-box"><strong>Software · round to nearest</strong>
    <code>(|{_escape(evidence['product'])}| + {_escape(evidence['rounding_offset'])})
    &gt;&gt; {_escape(evidence['shift'])} = {abs(evidence['software_result'])}</code>
    <b>{_escape(evidence['software_result'])}</b></div>
  <div class="fail-box"><strong>RTL fault · truncate</strong>
    <code>|{_escape(evidence['product'])}| &gt;&gt; {_escape(evidence['shift'])}
    = {abs(evidence['rtl_result'])}</code>
    <b>{_escape(evidence['rtl_result'])}</b></div>
</div>
<p class="impact"><strong>Difference: {difference:+} quantized unit.</strong>
One operation, one coordinate, and one RTL stage now define the search space.</p>
<div class="pipeline">
  <span>Accumulator <b>✓ MATCH</b></span><i>→</i>
  <span>Bias addition <b>✓ MATCH</b></span><i>→</i>
  <span>Fixed-point multiply <b>✓ MATCH</b></span><i>→</i>
  <span class="pipeline-fail">Rounding + shift <b>✕ FIRST ERROR</b></span><i>→</i>
  <span>Saturation</span>
</div>
<details><summary>Suggested implementation inspection</summary>
  <p><code>rtl/postprocess/tensorwright_postprocess.sv</code></p>
  <p>Inspect <code>product</code>, <code>rounded_magnitude</code>,
  <code>shift_i</code>, and <code>next_result</code>. Assert that negative halfway
  cases follow the bit-accurate round-to-nearest rule.</p>
</details>
<div class="suggested-fix"><strong>Recommended fix</strong>
  <p>Restore the round-to-nearest offset before shifting:</p>
  <code>rounded_magnitude =
  (magnitude + (64'd1 &lt;&lt; (shift_i - 1'b1))) &gt;&gt; shift_i;</code>
  <p>Keep the generated negative-halfway regression in the suite to prevent the
  truncation behavior from returning.</p>
</div>
</section>"""


def _regression_panel(regression: tuple[str, str] | None) -> str:
    if regression is None:
        return ""
    path, source = regression
    module = Path(path).with_suffix("").as_posix().replace("/", ".")
    command = ".venv/bin/python -m unittest " + module
    return f"""
<section id="regression">
<h2>Bug locked into regression</h2>
<p>Generated: <code>{_escape(path)}</code></p>
<div class="before-after">
  <div class="fail-box"><span>Before fix</span><strong>FAIL</strong>
    <small>Truncating RTL returns -116</small></div>
  <div class="pass-box"><span>After fix</span><strong>PASS</strong>
    <small>Corrected RTL returns -117</small></div>
</div>
<button type="button" onclick="navigator.clipboard.writeText('{_escape(command)}')">
Copy regression command</button>
<code class="command">{_escape(command)}</code>
<details><summary>Preview generated test</summary><pre>{_escape(source)}</pre></details>
</section>"""


def _tensor_slice(
    reference_path: str | Path,
    candidate_path: str | Path,
    comparison: ComparisonReport,
) -> str:
    divergence = comparison.first_divergence
    if divergence is None or divergence.coordinate is None:
        return ""
    coordinate = divergence.coordinate
    if len(coordinate) < 2:
        return ""
    reference_events = read_trace(reference_path).events
    candidate_events = read_trace(candidate_path).events
    matching_reference = [
        event
        for event in reference_events
        if event.source_operation_id == divergence.source_operation_id
        and event.tensor_name == divergence.tensor_name
    ]
    if not matching_reference:
        return ""
    shape = matching_reference[0].shape
    if len(shape) != len(coordinate):
        return ""
    prefix = coordinate[:-2]
    row_start = max(0, coordinate[-2] - 2)
    row_stop = min(shape[-2], coordinate[-2] + 3)
    column_start = max(0, coordinate[-1] - 2)
    column_stop = min(shape[-1], coordinate[-1] + 3)
    rows_to_show = range(row_start, row_stop)
    columns_to_show = range(column_start, column_stop)
    reference_values = _window_values(
        matching_reference,
        Path(reference_path).parent,
        prefix,
        rows_to_show,
        columns_to_show,
    )
    candidate_values = _window_values(
        [
            event
            for event in candidate_events
            if event.source_operation_id == divergence.source_operation_id
            and event.tensor_name == divergence.tensor_name
        ],
        Path(candidate_path).parent,
        prefix,
        rows_to_show,
        columns_to_show,
    )
    heading = "<tr><th></th>" + "".join(
        f"<th>Column {column}</th>" for column in columns_to_show
    ) + "</tr>"
    rows = heading
    for row in rows_to_show:
        cells = ""
        for column in columns_to_show:
            key = (row, column)
            reference = reference_values.get(key)
            candidate = candidate_values.get(key)
            css_class = ' class="mismatch"' if reference != candidate else ""
            relation = "=" if reference == candidate else "≠"
            delta = ""
            if (
                reference is not None
                and candidate is not None
                and reference != candidate
            ):
                delta = f"<b>Δ {candidate - reference:+}</b>"
            cells += (
                f"<td{css_class}><small>[{row}, {column}]</small>"
                f"{_escape(reference)} {relation} {_escape(candidate)}{delta}</td>"
            )
        rows += f"<tr><th>Row {row}</th>{cells}</tr>"
    return (
        '<section id="tensor-window"><h2>Bounded tensor window</h2>'
        f"<p>Fixed leading coordinate: <code>{_escape(prefix)}</code>. "
        f"Showing at most 5×5 values around {_escape(coordinate)}—not the full "
        f"{_escape(shape)} tensor. Each cell shows software versus RTL.</p>"
        f'<div class="table-scroll"><table class="tensor-grid"><tbody>{rows}'
        "</tbody></table></div></section>"
    )


def _window_values(
    events: list[TraceEvent],
    payload_directory: Path,
    prefix: list[int],
    rows: range,
    columns: range,
) -> dict[tuple[int, int], int | float]:
    values: dict[tuple[int, int], int | float] = {}
    for event in events:
        if event.event_type == "scalar":
            assert event.coordinate is not None and event.value is not None
            if (
                event.coordinate[:-2] == prefix
                and event.coordinate[-2] in rows
                and event.coordinate[-1] in columns
            ):
                values[(event.coordinate[-2], event.coordinate[-1])] = event.value
            continue
        assert event.data_file is not None
        assert event.start_coordinate is not None
        assert event.chunk_shape is not None
        payload = np.load(
            payload_directory / event.data_file,
            mmap_mode="r",
            allow_pickle=False,
        )
        for row in rows:
            for column in columns:
                coordinate = [*prefix, row, column]
                if all(
                    start <= index < start + size
                    for index, start, size in zip(
                        coordinate,
                        event.start_coordinate,
                        event.chunk_shape,
                        strict=True,
                    )
                ):
                    local = tuple(
                        index - start
                        for index, start in zip(
                            coordinate, event.start_coordinate, strict=True
                        )
                    )
                    values[(row, column)] = payload[local].item()
    return values


def _scalar_comparison_stats(
    reference_path: str | Path, candidate_path: str | Path
) -> tuple[int, int] | None:
    reference_events = read_trace(reference_path).events
    candidate_events = read_trace(candidate_path).events
    all_events = reference_events + candidate_events
    if any(event.event_type != "scalar" for event in all_events):
        return None

    def values(events: list[TraceEvent]) -> dict[tuple[str, str, tuple[int, ...]], Any]:
        result: dict[tuple[str, str, tuple[int, ...]], Any] = {}
        for event in events:
            assert event.coordinate is not None
            result[
                (
                    event.source_operation_id,
                    event.tensor_name,
                    tuple(event.coordinate),
                )
            ] = event.value
        return result

    reference = values(reference_events)
    candidate = values(candidate_events)
    keys = reference.keys() | candidate.keys()
    mismatches = sum(reference.get(key) != candidate.get(key) for key in keys)
    return len(reference), mismatches


def _render(
    comparison: ComparisonReport,
    diagnosis: DiagnosisReport,
    protocol: ProtocolReport,
    minimization: dict[str, Any] | None,
    regression: dict[str, Any] | None,
    baseline: ComparisonReport | None,
    scenario_note: str | None,
    tensor_slice: str,
    arithmetic_evidence: dict[str, Any] | None,
    regression_source: tuple[str, str] | None,
    comparison_stats: tuple[int, int] | None,
) -> str:
    status = "MATCH" if comparison.matched else "DIVERGENCE"
    status_class = "pass" if comparison.matched else "fail"
    protocol_status = "PASS" if protocol.protocol_ok else "FAIL"
    protocol_class = "pass" if protocol.protocol_ok else "fail"
    matched_label = (
        "Matched values"
        if comparison.matched
        else "Values before first divergence"
    )
    alignment_label = (
        f"{comparison.matched_values} values agree"
        if comparison.matched
        else f"{comparison.matched_values} before divergence"
    )
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
        f"<td>{_escape(item.recommended_check)}</td>"
        "</tr>"
        for item in protocol.findings
    )
    if not findings:
        findings = '<tr><td colspan="6">No protocol findings.</td></tr>'
    optional_sections = ""
    if minimization is not None:
        optional_sections += _json_section("Minimization", minimization)
    if regression is not None:
        optional_sections += _json_section("Generated regression", regression)
    presentation = _presentation_panel(comparison, baseline, scenario_note)
    arithmetic_panel = _arithmetic_panel(arithmetic_evidence)
    regression_panel = _regression_panel(regression_source)
    raw_report = {
        "dashboard_format_version": DASHBOARD_FORMAT_VERSION,
        "comparison": comparison.to_dict(),
        "diagnosis": diagnosis.to_dict(),
        "protocol": protocol.to_dict(),
        "minimization": minimization,
        "regression": regression,
        "baseline": None if baseline is None else baseline.to_dict(),
        "scenario_note": scenario_note,
        "arithmetic_evidence": arithmetic_evidence,
        "generated_regression": (
            None if regression_source is None else regression_source[0]
        ),
    }
    first_number = (
        "—" if divergence is None else f"#{comparison.matched_values + 1}"
    )
    likely_cause = (
        "None" if numerical is None else numerical.title.removeprefix("Likely ")
    )
    total_compared = (
        min(comparison.reference_values, comparison.candidate_values)
        if comparison_stats is None
        else comparison_stats[0]
    )
    mismatch_count = "—" if comparison_stats is None else comparison_stats[1]
    numerical_status = "PASS" if comparison.matched else "FAIL"
    numerical_class = "pass" if comparison.matched else "fail"
    diagnostic_lanes = f"""
<section><h2>Independent diagnostic lanes</h2><div class="lanes">
  <div><span>Numerical correctness</span>
    <strong class="{numerical_class}">{numerical_status}</strong>
    <p>{_escape(likely_cause)}</p></div>
  <div><span>Streaming protocol</span>
    <strong class="{protocol_class}">{protocol_status}</strong>
    <p>{len(protocol.findings)} ready/valid, ordering, count, or completion
    findings</p></div>
</div></section>"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TensorWright Verify — {_escape(comparison.model_id)}</title>
<style>
:root {{
  color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  --accent: #7657ff; --panel: #7770; --line: #7774;
}}
* {{ box-sizing: border-box; }} html {{ scroll-behavior: smooth; }}
body {{ margin: 0; line-height: 1.45; background: #77708; }}
.shell {{
  display: grid; grid-template-columns: 220px minmax(0, 1100px);
  gap: 1.5rem; justify-content: center;
}}
.sidebar {{
  position: sticky; top: 0; height: 100vh; padding: 2rem 1rem;
  border-right: 1px solid var(--line);
}}
.sidebar strong {{ display: block; font-size: 1.1rem; margin-bottom: 1rem; }}
.sidebar a {{
  display: block; color: inherit; text-decoration: none; padding: .45rem .6rem;
  border-radius: .4rem; font-size: .9rem;
}}
.sidebar a:hover {{ background: #7657ff18; color: var(--accent); }}
main {{ min-width: 0; padding: 1rem 1.5rem 3rem; }}
header, section {{
  border: 1px solid var(--line); border-radius: .8rem;
  padding: 1.15rem 1.3rem; margin: 1rem 0; background: #77708;
}}
h1, h2 {{ margin-top: 0; }}
.summary {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: .8rem;
}}
.metric {{ background: #7771; border-radius: .5rem; padding: .8rem; }}
.metric strong {{ display: block; font-size: 1.35rem; }}
.secondary-metrics {{
  display: flex; flex-wrap: wrap; gap: 1.2rem; margin: 1rem 0; color: #777;
}}
.lanes {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
.lanes > div {{ background: #7771; border-radius: .55rem; padding: 1rem; }}
.lanes strong {{ display: block; font-size: 1.6rem; }}
.pass {{ color: #17833b; }} .fail {{ color: #c0392b; }}
.scenario {{
  border-color: #e59b2688; background: #e59b2618;
  display: flex; gap: .8rem; align-items: center;
}}
.scenario strong {{ color: #c97800; text-transform: uppercase; letter-spacing: .08em; }}
.reveal {{ display: grid; grid-template-columns: 1fr 1.4fr; gap: 1rem; }}
.baseline {{ border-left: .35rem solid #17833b; }}
.value-reveal {{
  display: flex; align-items: center; justify-content: center; gap: 1.2rem;
  font-size: clamp(1.8rem, 5vw, 3.5rem); font-weight: 800;
}}
.value-reveal .arrow {{ color: #c0392b; }}
.value-label {{ display: block; font-size: .75rem; font-weight: 500; color: #777; }}
.fault-facts {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .6rem; }}
.fact {{
  background: #7771; border-radius: .45rem; padding: .65rem; text-align: center;
}}
.fact strong {{ display: block; font-size: 1.1rem; }}
.flow {{
  display: grid; grid-template-columns: 1fr auto 1fr auto 1fr;
  align-items: center; gap: .75rem; margin: 1.5rem 0 .25rem;
}}
.flow-node {{
  min-height: 4.2rem; display: grid; place-content: center; text-align: center;
  background: #7771; border: 1px solid #7775; border-radius: .6rem; padding: .7rem;
}}
.flow-node strong {{ display: block; font-size: 1.05rem; }}
.flow-arrow {{ font-size: 1.8rem; color: #777; }}
@media (max-width: 700px) {{
  .flow {{ grid-template-columns: 1fr; }}
  .flow-arrow {{ transform: rotate(90deg); text-align: center; }}
}}
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
.tensor-grid {{
  border-collapse: separate; border-spacing: .35rem;
  width: auto; margin: 1rem auto;
}}
.tensor-grid td {{
  min-width: 5.5rem; border: 1px solid #7775; border-radius: .45rem;
  padding: .7rem; text-align: center; background: #7771;
}}
.tensor-grid td.mismatch {{ border: 2px solid #c0392b; background: #c0392b18; }}
.tensor-grid small, .tensor-grid b {{ display: block; color: #777; }}
.tensor-grid td.mismatch b {{ color: #c0392b; }}
.table-scroll {{ max-width: 100%; overflow-x: auto; }}
.arithmetic {{
  display: flex; flex-wrap: wrap; align-items: center; justify-content: center;
  gap: .7rem; margin: 1rem 0;
}}
.arithmetic div:not(.arrow) {{
  background: #7771; border-radius: .5rem; padding: .7rem 1rem; text-align: center;
}}
.arithmetic span, .rounding-compare span {{ display: block; color: #777; }}
.arithmetic strong {{ display: block; font-size: 1.25rem; }}
.rounding-compare, .before-after {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;
}}
.pass-box, .fail-box {{ border-radius: .55rem; padding: 1rem; }}
.pass-box {{ border: 1px solid #17833b88; background: #17833b14; }}
.fail-box {{ border: 1px solid #c0392b88; background: #c0392b14; }}
.rounding-compare code {{ display: block; margin: .7rem 0; }}
.rounding-compare b {{ font-size: 2rem; }}
.impact {{ font-size: 1.1rem; text-align: center; }}
.pipeline {{ display: flex; align-items: stretch; gap: .4rem; margin: 1rem 0; }}
.pipeline span {{ flex: 1; background: #7771; padding: .7rem; border-radius: .4rem; }}
.pipeline b {{ display: block; color: #17833b; font-size: .8rem; }}
.pipeline .pipeline-fail {{ border: 2px solid #c0392b; }}
.pipeline .pipeline-fail b {{ color: #c0392b; }}
.pipeline i {{ align-self: center; color: #777; }}
.before-after strong {{ display: block; font-size: 1.8rem; }}
.command {{ display: block; margin: .7rem 0; padding: .6rem; background: #7771; }}
.suggested-fix {{
  margin-top: 1rem; border-left: .35rem solid var(--accent);
  background: #7657ff14; border-radius: .45rem; padding: 1rem;
}}
.suggested-fix > strong {{ color: var(--accent); font-size: 1.1rem; }}
.suggested-fix code {{ display: block; overflow-wrap: anywhere; }}
button {{ cursor: pointer; border: 0; border-radius: .4rem; padding: .65rem 1rem; }}
@media (max-width: 700px) {{
  .shell {{ display: block; }} .sidebar {{ display: none; }} main {{ padding: .7rem; }}
  .rounding-compare, .before-after {{ grid-template-columns: 1fr; }}
  .lanes {{ grid-template-columns: 1fr; }}
  .pipeline {{ flex-direction: column; }} .pipeline i {{ transform: rotate(90deg); }}
}}
</style>
</head>
<body><div class="shell">
<aside class="sidebar"><strong>TensorWright</strong>
  <nav aria-label="Dashboard sections">
    <a href="#overview">Overview</a>
    <a href="#experiment">Experiment</a>
    <a href="#arithmetic-evidence">Arithmetic evidence</a>
    <a href="#tensor-window">Tensor window</a>
    <a href="#regression">Regression</a>
    <a href="#diagnosis">Diagnosis</a>
    <a href="#protocol">Protocol</a>
    <a href="#technical-details">Technical details</a>
  </nav>
</aside><main>
<header id="overview">
<h1>TensorWright Verify</h1>
<p>Offline debugging dashboard · format version {DASHBOARD_FORMAT_VERSION}</p>
<div class="summary">
<div class="metric"><span>Result</span>
<strong class="{status_class}">{status}</strong></div>
<div class="metric"><span>First divergence</span><strong>{first_number}</strong></div>
<div class="metric"><span>{matched_label}</span>
<strong>{comparison.matched_values}</strong></div>
<div class="metric"><span>Likely cause</span>
<strong>{_escape(likely_cause)}</strong></div>
</div>
<div class="secondary-metrics">
  <span>Compared: <b>{total_compared}/{comparison.reference_values}</b></span>
  <span>Total mismatches: <b>{mismatch_count}</b></span>
  <span>Protocol violations: <b>{len(protocol.findings)}</b></span>
</div>
<div class="flow" aria-label="Verification flow">
  <div class="flow-node"><span>Software truth</span>
    <strong>{_escape(comparison.reference_backend)}</strong></div>
  <div class="flow-arrow">→</div>
  <div class="flow-node"><span>Semantic alignment</span>
    <strong>{alignment_label}</strong></div>
  <div class="flow-arrow">→</div>
  <div class="flow-node"><span>Hardware candidate</span>
    <strong>{_escape(comparison.candidate_backend)}</strong></div>
</div>
</header>
{presentation}
{diagnostic_lanes}
{arithmetic_panel}
{regression_panel}
<section id="technical-details"><details>
<summary>Technical divergence identifiers</summary>
<h2>First divergence</h2>{divergence_panel}</details></section>
{tensor_slice}
<section id="diagnosis"><h2>Numerical diagnosis</h2>{diagnosis_panel}</section>
<section id="protocol"><h2>Protocol findings</h2>
<div class="table-scroll"><table><thead><tr><th>Severity</th><th>Rule</th>
<th>Event</th><th>Cycle</th><th>Evidence</th><th>Recommended fix</th></tr></thead>
<tbody>{findings}</tbody></table></div></section>
{optional_sections}
{_json_section("Complete machine-readable report", raw_report)}
</main></div></body>
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
