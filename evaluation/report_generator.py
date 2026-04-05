"""
report_generator.py – Generates HTML benchmark reports with trade-off analysis.

FIXES APPLIED:
  - #18 Extracted _color() helper: Python conditional expressions like
        `{'green' if ... else 'yellow'}` no longer embedded inside HTML
        attribute strings — much easier to read and maintain.

Usage:
    from evaluation.report_generator import generate_html_report
    generate_html_report(benchmark_summary, output_path="report.html")
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


# ──────────────────────────────────────────────
# FIX #18: helpers extracted from f-string soup
# ──────────────────────────────────────────────

def _color(condition: bool, true_cls: str = "green", false_cls: str = "yellow") -> str:
    """Return a CSS class name based on a boolean condition."""
    return true_cls if condition else false_cls


def _pct_bar(value_ms: float, budget_ms: float) -> int:
    """Convert a latency value to a 0–100 integer percentage of budget."""
    if budget_ms <= 0:
        return 0
    return min(int((value_ms / budget_ms) * 100), 100)


def generate_html_report(
    summary: Dict[str, Any],
    output_path: str = "benchmark_report.html",
) -> str:
    """
    Generate a rich HTML report from benchmark summary data.

    Args:
        summary: Output dict from voice_agent.benchmark.run_benchmark()
        output_path: Where to save the HTML file.

    Returns:
        Absolute path to the generated report.
    """
    lat        = summary.get("latency", {})
    cost       = summary.get("cost", {})
    individual = summary.get("individual_results", [])
    api_stack  = summary.get("api_stack", {})
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    budget_ms  = summary.get("budget_ms", 1000)
    within_pct = summary.get("within_budget_pct", 0)

    # FIX #2 (carried through): component_means is now always present in
    # benchmark output — these bars will render correctly.
    comp = lat.get("component_means", {})

    # ── Individual results table rows ──────────────────────────────────
    rows_html = ""
    for r in individual:
        ok        = r.get("within_budget", True)
        row_class = "ok" if ok else "warn"
        status    = "✅" if ok else "⚠️"
        rows_html += f"""
        <tr class="{row_class}">
            <td>{status} {r.get('query', '')[:50]}</td>
            <td>{r.get('retrieval_ms', 0):.0f}</td>
            <td>{r.get('llm_ms', 0):.0f}</td>
            <td>{r.get('total_ms', 0):.0f}</td>
            <td>{r.get('context_chunks', 0)}</td>
            <td>₹{r.get('cost_inr', 0):.5f}</td>
        </tr>"""

    # ── FIX #18: compute CSS classes via helper, not inline ternaries ──
    p50_cls    = _color(lat.get("p50_ms", 9999) <= budget_ms)
    p95_cls    = _color(lat.get("p95_ms", 9999) <= budget_ms * 1.3)
    within_cls = _color(within_pct >= 90)

    stt_bar = _pct_bar(comp.get("stt_ms", 0),        budget_ms)
    rag_bar = _pct_bar(comp.get("retrieval_ms", 0),   budget_ms)
    llm_bar = _pct_bar(comp.get("llm_ms", 0),         budget_ms)
    tts_bar = _pct_bar(comp.get("tts_ms", 0),         budget_ms)

    retrieval_mean = comp.get("retrieval_ms", 0)

    api_tags = "".join([
        f'<span class="api-tag">🎤 STT: {api_stack.get("stt", "")}</span>',
        f'<span class="api-tag">🧠 LLM: {api_stack.get("llm", "")}</span>',
        f'<span class="api-tag">🔊 TTS: {api_stack.get("tts", "")}</span>',
        f'<span class="api-tag">📦 Vector: {api_stack.get("vector_store", "")}</span>',
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sarvam AI Voice Agent — Benchmark Report</title>
<style>
  :root {{
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --accent: #7c3aed; --green: #22c55e; --yellow: #f59e0b;
    --red: #ef4444; --text: #e6edf3; --muted: #8b949e;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; padding: 32px; }}
  h1 {{ font-size: 2rem; background: linear-gradient(135deg, #7c3aed, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px; }}
  .subtitle {{ color: var(--muted); margin-bottom: 32px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 32px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }}
  .kpi-label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }}
  .kpi-value {{ font-size: 2rem; font-weight: 700; }}
  .kpi-sub {{ font-size: 0.8rem; color: var(--muted); margin-top: 4px; }}
  .green {{ color: var(--green); }}
  .yellow {{ color: var(--yellow); }}
  .red {{ color: var(--red); }}
  .purple {{ color: #a78bfa; }}
  h2 {{ font-size: 1.2rem; margin-bottom: 16px; color: #c9d1d9; }}
  .bar-section {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
  .bar-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }}
  .bar-label {{ width: 80px; font-size: 0.85rem; color: var(--muted); }}
  .bar-track {{ flex: 1; background: #21262d; border-radius: 6px; height: 24px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 6px; display: flex; align-items: center; padding-left: 8px; font-size: 0.75rem; font-weight: 600; transition: width 0.4s ease; }}
  .bar-stt {{ background: linear-gradient(90deg, #7c3aed, #9f67f5); }}
  .bar-rag {{ background: linear-gradient(90deg, #0891b2, #06b6d4); }}
  .bar-llm {{ background: linear-gradient(90deg, #059669, #10b981); }}
  .bar-tts {{ background: linear-gradient(90deg, #d97706, #f59e0b); }}
  .bar-val {{ width: 70px; text-align: right; font-size: 0.85rem; color: var(--muted); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ background: #21262d; padding: 10px 12px; text-align: left; color: var(--muted); font-weight: 600; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); }}
  tr.warn td {{ color: var(--yellow); }}
  tr:hover {{ background: #21262d; }}
  .api-tag {{ display: inline-block; background: #21262d; border: 1px solid var(--border); border-radius: 6px; padding: 4px 10px; font-size: 0.75rem; margin: 4px; }}
  .section {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 24px; overflow-x: auto; }}
  .trade-off {{ background: linear-gradient(135deg, #0d1117, #1c1f26); border-left: 4px solid var(--accent); padding: 16px 20px; border-radius: 8px; margin-top: 16px; line-height: 1.7; }}
  footer {{ color: var(--muted); font-size: 0.75rem; text-align: center; margin-top: 40px; }}
</style>
</head>
<body>

<h1>🎙️ Sarvam AI Voice Agent</h1>
<p class="subtitle">Benchmark Report &nbsp;|&nbsp; {timestamp} &nbsp;|&nbsp; {summary.get('total_queries', 0)} queries</p>

<!-- KPI Cards -->
<div class="grid">
  <div class="card">
    <div class="kpi-label">Latency P50</div>
    <div class="kpi-value {p50_cls}">{lat.get('p50_ms', 0):.0f}<span style="font-size:1rem">ms</span></div>
    <div class="kpi-sub">Budget: {budget_ms}ms</div>
  </div>
  <div class="card">
    <div class="kpi-label">Latency P95</div>
    <div class="kpi-value {p95_cls}">{lat.get('p95_ms', 0):.0f}<span style="font-size:1rem">ms</span></div>
    <div class="kpi-sub">Within 30% headroom</div>
  </div>
  <div class="card">
    <div class="kpi-label">Within Budget</div>
    <div class="kpi-value {within_cls}">{within_pct}<span style="font-size:1rem">%</span></div>
    <div class="kpi-sub">Queries ≤ {budget_ms}ms</div>
  </div>
  <div class="card">
    <div class="kpi-label">Cost / Query</div>
    <div class="kpi-value green">₹{cost.get('per_query_avg_inr', 0):.4f}</div>
    <div class="kpi-sub">₹{cost.get('per_100_queries_inr', 0):.2f} / 100 queries</div>
  </div>
  <div class="card">
    <div class="kpi-label">LLM Cost</div>
    <div class="kpi-value green">FREE</div>
    <div class="kpi-sub">Sarvam-M (₹0/token)</div>
  </div>
  <div class="card">
    <div class="kpi-label">Successful</div>
    <div class="kpi-value purple">{summary.get('successful', 0)}/{summary.get('total_queries', 0)}</div>
    <div class="kpi-sub">{summary.get('errors', 0)} errors</div>
  </div>
</div>

<!-- Latency Breakdown Bars -->
<div class="bar-section">
  <h2>⏱ Latency Breakdown (Mean per Component)</h2>
  <div class="bar-row">
    <div class="bar-label">STT</div>
    <div class="bar-track"><div class="bar-fill bar-stt" style="width:{stt_bar}%">{comp.get('stt_ms', 0):.0f}ms</div></div>
    <div class="bar-val">Saaras v3</div>
  </div>
  <div class="bar-row">
    <div class="bar-label">RAG</div>
    <div class="bar-track"><div class="bar-fill bar-rag" style="width:{rag_bar}%">{comp.get('retrieval_ms', 0):.0f}ms</div></div>
    <div class="bar-val">ChromaDB</div>
  </div>
  <div class="bar-row">
    <div class="bar-label">LLM</div>
    <div class="bar-track"><div class="bar-fill bar-llm" style="width:{llm_bar}%">{comp.get('llm_ms', 0):.0f}ms</div></div>
    <div class="bar-val">Sarvam-M</div>
  </div>
  <div class="bar-row">
    <div class="bar-label">TTS</div>
    <div class="bar-track"><div class="bar-fill bar-tts" style="width:{tts_bar}%">{comp.get('tts_ms', 0):.0f}ms</div></div>
    <div class="bar-val">Bulbul v3</div>
  </div>
</div>

<!-- API Stack -->
<div class="section">
  <h2>🧩 API Stack</h2>
  {api_tags}

  <div class="trade-off">
    <strong>Trade-off Analysis</strong><br><br>
    <strong>Latency vs Accuracy:</strong> The RAG approach adds ~{retrieval_mean:.0f}ms for retrieval
    but significantly improves accuracy by grounding answers in your documents.
    Without RAG, the LLM would hallucinate.<br><br>
    <strong>Cost vs Performance:</strong> Sarvam-M's current free pricing makes this system
    extremely economical. The primary costs are STT (₹30/hr) and TTS (₹30/10K chars).
    Using local embeddings eliminates embedding API costs entirely.<br><br>
    <strong>Scalability:</strong> ChromaDB can handle millions of vectors locally.
    For production, consider migrating to a hosted vector DB (Pinecone, Weaviate) and
    adding Redis caching for common queries to cut latency further.
  </div>
</div>

<!-- Individual Results Table -->
<div class="section">
  <h2>📋 Individual Query Results</h2>
  <table>
    <thead>
      <tr>
        <th>Query</th>
        <th>RAG (ms)</th>
        <th>LLM (ms)</th>
        <th>Total (ms)</th>
        <th>Chunks</th>
        <th>Cost (₹)</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>

<footer>Generated by Sarvam AI Voice Agent Benchmark Tool &nbsp;|&nbsp; {timestamp}</footer>
</body>
</html>"""

    out = Path(output_path)
    out.write_text(html, encoding="utf-8")
    return str(out.absolute())


def generate_markdown_report(summary: Dict[str, Any]) -> str:
    """Generate a concise Markdown summary of benchmark results."""
    lat        = summary.get("latency", {})
    cost       = summary.get("cost", {})
    within_pct = summary.get("within_budget_pct", 0)
    comp       = lat.get("component_means", {})

    return f"""# Sarvam AI Voice Agent — Benchmark Summary

| Metric | Value |
|--------|-------|
| Total Queries | {summary.get('total_queries', 0)} |
| Successful | {summary.get('successful', 0)} |
| **Latency P50** | **{lat.get('p50_ms', 0):.0f} ms** |
| **Latency P95** | **{lat.get('p95_ms', 0):.0f} ms** |
| Within Budget ({summary.get('budget_ms', 1000)}ms) | {within_pct}% |
| Cost / Query | ₹{cost.get('per_query_avg_inr', 0):.5f} |
| Cost / 100 Queries | ₹{cost.get('per_100_queries_inr', 0):.3f} |
| LLM Cost | FREE (Sarvam-M) |

## Component Latencies (Mean)

| Component | Latency |
|-----------|---------|
| STT (Saaras v3) | {comp.get('stt_ms', 0):.0f} ms |
| RAG (ChromaDB) | {comp.get('retrieval_ms', 0):.0f} ms |
| LLM (Sarvam-M) | {comp.get('llm_ms', 0):.0f} ms |
| TTS (Bulbul v3) | {comp.get('tts_ms', 0):.0f} ms |
"""
