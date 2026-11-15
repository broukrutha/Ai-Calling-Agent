"""
Render a benchmark summary as a standalone HTML page or as Markdown.

    from evaluation.report_generator import generate_html_report
    generate_html_report(summary, "benchmark_report.html")

Everything interpolated into the page is HTML-escaped.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

_STYLE = """
:root { --bg:#0d1117; --card:#161b22; --border:#30363d; --green:#22c55e; --yellow:#f59e0b;
        --text:#e6edf3; --muted:#8b949e; }
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; padding:32px; }
h1 { font-size:1.8rem; margin-bottom:6px; }
h2 { font-size:1.1rem; margin-bottom:14px; color:#c9d1d9; }
.subtitle { color:var(--muted); margin-bottom:28px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin-bottom:28px; }
.card, .section { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px; }
.section { margin-bottom:22px; overflow-x:auto; }
.label { font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; margin-bottom:8px; }
.value { font-size:1.8rem; font-weight:700; }
.sub { font-size:.78rem; color:var(--muted); margin-top:4px; }
.green { color:var(--green); } .yellow { color:var(--yellow); }
.tag { display:inline-block; background:#21262d; border:1px solid var(--border); border-radius:6px;
       padding:4px 10px; font-size:.75rem; margin:0 6px 6px 0; }
table { width:100%; border-collapse:collapse; font-size:.85rem; }
th { background:#21262d; padding:9px 12px; text-align:left; color:var(--muted); }
td { padding:9px 12px; border-bottom:1px solid var(--border); }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
tr.warn td { color:var(--yellow); }
footer { color:var(--muted); font-size:.75rem; text-align:center; margin-top:32px; }
"""


def _tone(ok: bool) -> str:
    return "green" if ok else "yellow"


def _kpi(label: str, value: str, sub: str, tone: str = "") -> str:
    return (
        f'<div class="card"><div class="label">{escape(label)}</div>'
        f'<div class="value {tone}">{escape(value)}</div><div class="sub">{escape(sub)}</div></div>'
    )


def generate_html_report(summary: dict[str, Any], output_path: str = "benchmark_report.html") -> str:
    """Write the report and return its absolute path."""
    latency = summary.get("latency", {})
    cost = summary.get("cost", {})
    accuracy = summary.get("accuracy", {})
    stack = summary.get("api_stack", {})
    budget = summary.get("budget_ms", 0)
    within = summary.get("within_budget_pct", 0)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    cards = "".join([
        _kpi("Latency P50", f"{latency.get('p50_ms', 0):.0f} ms", f"Budget {budget} ms",
             _tone(latency.get("p50_ms", 0) <= budget)),
        _kpi("Latency P95", f"{latency.get('p95_ms', 0):.0f} ms", "95th percentile",
             _tone(latency.get("p95_ms", 0) <= budget)),
        _kpi("Within budget", f"{within}%", f"Queries ≤ {budget} ms", _tone(within >= 90)),
        _kpi("Answer accuracy", f"{accuracy.get('mean_score', 0) * 100:.0f}%", "Expected keywords found",
             _tone(accuracy.get("mean_score", 0) >= 0.6)),
        _kpi("Cost / query", f"₹{cost.get('per_query_avg_inr', 0):.5f}",
             f"₹{cost.get('per_100_queries_inr', 0):.3f} per 100"),
        _kpi("Successful", f"{summary.get('successful', 0)}/{summary.get('total_queries', 0)}",
             f"{summary.get('errors', 0)} errors"),
    ])
    tags = "".join(f'<span class="tag">{escape(key.upper())}: {escape(str(value))}</span>' for key, value in stack.items())
    rows = "".join(
        f'<tr class="{"" if row.get("within_budget", True) else "warn"}">'
        f'<td>{escape(str(row.get("query", ""))[:80])}</td>'
        f'<td class="num">{row.get("retrieval_ms", 0):.0f}</td>'
        f'<td class="num">{row.get("llm_ms", 0):.0f}</td>'
        f'<td class="num">{row.get("total_ms", 0):.0f}</td>'
        f'<td class="num">{row.get("accuracy", 0) * 100:.0f}%</td>'
        f'<td class="num">₹{row.get("cost_inr", 0):.5f}</td></tr>'
        for row in summary.get("individual_results", [])
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Calling Agent — Benchmark Report</title>
<style>{_STYLE}</style>
</head>
<body>
<h1>AI Calling Agent — Benchmark</h1>
<p class="subtitle">{escape(generated)} · {summary.get('total_queries', 0)} queries</p>
<div class="grid">{cards}</div>
<div class="section"><h2>Stack</h2>{tags}</div>
<div class="section">
  <h2>Query results</h2>
  <table>
    <thead><tr><th>Query</th><th class="num">Retrieval (ms)</th><th class="num">LLM (ms)</th>
    <th class="num">Total (ms)</th><th class="num">Accuracy</th><th class="num">Cost</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
<footer>Text pipeline only: speech-to-text and text-to-speech are not exercised by this benchmark.</footer>
</body>
</html>"""

    out = Path(output_path)
    out.write_text(html, encoding="utf-8")
    return str(out.resolve())


def generate_markdown_report(summary: dict[str, Any]) -> str:
    latency = summary.get("latency", {})
    cost = summary.get("cost", {})
    accuracy = summary.get("accuracy", {})
    return f"""# AI Calling Agent — Benchmark Summary

| Metric | Value |
|--------|-------|
| Queries | {summary.get('successful', 0)}/{summary.get('total_queries', 0)} successful |
| Latency P50 | {latency.get('p50_ms', 0):.0f} ms |
| Latency P95 | {latency.get('p95_ms', 0):.0f} ms |
| Within budget ({summary.get('budget_ms', 0)} ms) | {summary.get('within_budget_pct', 0)}% |
| Mean retrieval | {latency.get('retrieval_mean_ms', 0):.0f} ms |
| Mean LLM | {latency.get('llm_mean_ms', 0):.0f} ms |
| Answer accuracy | {accuracy.get('mean_score', 0) * 100:.0f}% |
| Cost / query | ₹{cost.get('per_query_avg_inr', 0):.5f} |
"""
