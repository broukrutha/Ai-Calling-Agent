"""
Render evaluation results as a standalone HTML page.

    generate_html_report([retrieval_summary, dialog_summary, answers_summary], "report.html")

Everything interpolated into the page is HTML-escaped.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

_STYLE = """
:root { --bg:#f6f7f9; --card:#fff; --border:#e2e5ea; --text:#151922; --muted:#667085;
        --good:#0b7d47; --good-soft:#e1f4ea; --bad:#c22b27; --bad-soft:#fce6e5; }
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font:14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; padding:32px; }
h1 { font-size:22px; margin-bottom:4px; } h2 { font-size:16px; margin:28px 0 12px; }
.sub { color:var(--muted); margin-bottom:8px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }
.label { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }
.value { font-size:24px; font-weight:650; margin-top:4px; font-variant-numeric:tabular-nums; }
table { width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--border); border-radius:10px; overflow:hidden; margin-top:12px; }
th, td { padding:8px 12px; text-align:left; border-bottom:1px solid var(--border); vertical-align:top; }
th { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; background:#fafbfc; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.pill { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px; font-weight:600; }
.pass { background:var(--good-soft); color:var(--good); } .fail { background:var(--bad-soft); color:var(--bad); }
.muted { color:var(--muted); }
"""


def _pct(value: Any) -> str:
    return "–" if value is None else f"{value * 100:.1f}%"


def _card(label: str, value: str) -> str:
    return f'<div class="card"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div></div>'


def _pill(passed: bool) -> str:
    return f'<span class="pill {"pass" if passed else "fail"}">{"pass" if passed else "fail"}</span>'


def _gates(summary: dict[str, Any]) -> str:
    if not summary.get("gates"):
        return ""
    rows = "".join(
        f"<tr><td>{escape(gate['metric'])}</td><td class='num'>{escape(str(gate['value']))}</td>"
        f"<td class='num'>{gate['minimum']}</td><td>{_pill(gate['passed'])}</td></tr>"
        for gate in summary["gates"]
    )
    return f"<table><thead><tr><th>Gate</th><th>Value</th><th>Minimum</th><th></th></tr></thead><tbody>{rows}</tbody></table>"


def _retrieval(summary: dict[str, Any]) -> str:
    a = summary["answerable"]
    cards = "".join([
        _card("Answer in context", _pct(a["context_hit"])), _card("Ranked first", _pct(a["hit_at_1"])),
        _card("MRR", f"{a['mrr']:.3f}"), _card("Context precision", _pct(a["context_precision"])),
        _card("Avg passages", f"{summary['avg_passages']:.2f}"), _card("Latency p50", f"{summary['latency_ms']['p50']:.0f} ms"),
    ])
    rows = "".join(
        f"<tr><td>{escape(row['question'])}<div class='muted'>{escape(row['type'])}</div></td>"
        f"<td>{_pill(bool(row['passed']))}</td><td>{escape(', '.join(row['sources']))}</td>"
        f"<td class='num'>{escape(', '.join(f'{score:.2f}' for score in row['scores']))}</td></tr>"
        for row in summary["rows"] if row["answerable"]
    )
    return (f"<div class='grid'>{cards}</div>{_gates(summary)}"
            f"<table><thead><tr><th>Question</th><th></th><th>Retrieved from</th><th>Scores</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")


def _dialog(summary: dict[str, Any]) -> str:
    cards = _card("Task success", _pct(summary["success_rate"])) + _card(
        "Avg turns to complete", str(summary["avg_turns_to_complete"] or "–"))
    rows = "".join(
        f"<tr><td>{escape(row['id'])}<div class='muted'>{escape(row['description'])}</div></td>"
        f"<td>{_pill(row['passed'])}</td><td>{escape('; '.join(row['problems']))}</td></tr>"
        for row in summary["rows"]
    )
    return (f"<div class='grid'>{cards}</div>{_gates(summary)}"
            f"<table><thead><tr><th>Scenario</th><th></th><th>Problems</th></tr></thead><tbody>{rows}</tbody></table>")


def _answers(summary: dict[str, Any]) -> str:
    cards = "".join([
        _card("Pass rate", _pct(summary["pass_rate"])), _card("Correctness", _pct(summary["correctness"])),
        _card("Faithfulness", _pct(summary["faithfulness"])), _card("Refusal accuracy", _pct(summary["refusal_accuracy"])),
        _card("False refusals", _pct(summary["false_refusal_rate"])), _card("Speakable", _pct(summary["voice_format_rate"])),
    ])
    rows = "".join(
        f"<tr><td>{escape(row['question'])}</td><td>{_pill(row['passed'])}</td>"
        f"<td>{escape(row.get('answer', row.get('error', '')))}"
        f"<div class='muted'>{escape(row.get('judge', {}).get('reason', ''))}</div></td></tr>"
        for row in summary["rows"]
    )
    judged = "" if summary["judged"] else "<p class='sub'>No LLM judge: scored by keywords and refusal detection only.</p>"
    return (f"{judged}<div class='grid'>{cards}</div>{_gates(summary)}"
            f"<table><thead><tr><th>Question</th><th></th><th>Answer</th></tr></thead><tbody>{rows}</tbody></table>")


_RENDERERS = {"retrieval": ("Retrieval", _retrieval), "dialog": ("Booking dialogs", _dialog), "answers": ("Answers", _answers)}


def render_html(results: list[dict[str, Any]]) -> str:
    sections = "".join(
        f"<h2>{escape(_RENDERERS[summary['suite']][0])} — {'all gates passed' if summary.get('gates_passed', True) else 'below threshold'}</h2>"
        f"{_RENDERERS[summary['suite']][1](summary)}"
        for summary in results if summary.get("suite") in _RENDERERS
    )
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Agent evaluation report</title><style>{_STYLE}</style></head><body>"
        f"<h1>Agent evaluation report</h1><p class='sub'>{escape(generated)}</p>{sections}</body></html>"
    )


def generate_html_report(results: list[dict[str, Any]], output_path: str = "evaluation_report.html") -> str:
    out = Path(output_path)
    out.write_text(render_html(results), encoding="utf-8")
    return str(out.resolve())
