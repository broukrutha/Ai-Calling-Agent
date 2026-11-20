"""Evaluation framework: datasets, metrics, the three suites, gates and the report."""

import json

import pytest

from evaluation import metrics
from evaluation.answer_eval import detect_refusal, evaluate_answers, parse_judgement, voice_format_ok
from evaluation.dataset import DEFAULT_DATASET, EvalCase, load_cases
from evaluation.dialog_eval import evaluate_dialogs, load_scenarios
from evaluation.report_generator import render_html
from evaluation.retrieval_eval import evaluate_retrieval
from evaluation.runner import SAMPLE_DOCS, check_thresholds, load_thresholds, run_suite
from tests.fakes import HashingEmbedder
from voice_agent.booking import store
from voice_agent.sarvam_client import LLMResult

# ── Datasets ─────────────────────────────────────────────────────────────────


def test_golden_dataset_is_valid_and_grounded():
    cases = load_cases()
    assert len(cases) >= 60 and len({case.id for case in cases}) == len(cases)
    documents = {path.name: metrics.normalize(path.read_text()) for path in SAMPLE_DOCS.iterdir()}
    for case in cases:
        for evidence in case.evidence:
            assert metrics.normalize(evidence) in documents[case.source], (case.id, evidence)
    kinds = {case.type for case in cases}
    assert {"factoid", "paraphrase", "followup", "topic_switch", "code_mixed", "unanswerable"} <= kinds


@pytest.mark.parametrize("bad,message", [
    ({"id": "x", "question": "q", "answerable": True}, "evidence"),
    ({"id": "x", "question": "q", "answerable": False, "evidence": ["a"]}, "must not"),
    ({"id": "x", "question": "q", "type": "riddle", "evidence": ["a"]}, "unknown type"),
    ({"id": "x", "question": "q", "evidence": ["a"], "history": [{"role": "system", "content": "c"}]}, "roles"),
])
def test_invalid_cases_are_rejected(tmp_path, bad, message):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match=message):
        load_cases(path)


def test_duplicate_ids_are_rejected(tmp_path):
    row = json.dumps({"id": "x", "question": "q", "evidence": ["a"]})
    path = tmp_path / "dup.jsonl"
    path.write_text(f"{row}\n{row}\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)


def test_dialog_scenarios_load():
    assert len(load_scenarios()) >= 10


# ── Metrics ──────────────────────────────────────────────────────────────────

def test_ranking_metrics():
    assert metrics.hit_at_k([0, 1, 0], 1) == 0.0 and metrics.hit_at_k([0, 1, 0], 2) == 1.0
    assert metrics.reciprocal_rank([0, 0, 1]) == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank([0, 0]) == 0.0
    assert metrics.ndcg_at_k([1, 0, 0], 3) == 1.0
    assert metrics.ndcg_at_k([0, 1], 2) == pytest.approx(1 / 1.5849625, rel=1e-4)
    assert metrics.context_precision([1, 0, 0, 1]) == 0.5
    assert metrics.evidence_recall(["Cleaning: 1,200 rupees", "x"], ["1,200 rupees", "missing"]) == 0.5


def test_keyword_accuracy_ignores_digit_separators():
    assert metrics.keyword_accuracy("It costs 1,20,000 rupees", ["120000"]) == 1.0
    assert metrics.keyword_accuracy("It costs 1 200 rupees", ["1200"]) == 1.0
    assert metrics.keyword_accuracy("No idea", ["1200"]) == 0.0


# ── Retrieval suite ──────────────────────────────────────────────────────────

class _P:
    def __init__(self, text, source="faq.md", score=0.5):
        self.text, self.source, self.score = text, source, score


async def test_retrieval_suite_scores_what_reaches_the_llm():
    cases = [
        EvalCase(id="a", question="price?", evidence=("1,200 rupees",)),
        EvalCase(id="b", question="hours?", evidence=("9 to 5",)),
        EvalCase(id="c", question="wifi?", type="unanswerable", answerable=False),
    ]
    answers = {"price?": [_P("noise"), _P("Cleaning 1,200 rupees")], "hours?": [_P("noise")], "wifi?": []}

    async def retrieve(question, history):
        return answers[question]

    summary = await evaluate_retrieval(cases, retrieve)
    assert summary["answerable"]["context_hit"] == 0.5 and summary["answerable"]["hit_at_1"] == 0.0
    assert summary["answerable"]["mrr"] == 0.25 and summary["passed"] == 1 and summary["failed"] == ["b"]
    assert summary["abstention"]["rate"] == 1.0


async def test_retrieval_suite_on_the_sample_kb_runs_end_to_end():
    summary = await run_suite("retrieval", kb="sample", embedding_function=HashingEmbedder())
    assert summary["cases"] >= 60 and summary["kb"] == "sample"
    assert {gate["metric"] for gate in summary["gates"]} == set(load_thresholds()["retrieval"])


# ── Answers suite ────────────────────────────────────────────────────────────

class _Agent:
    def __init__(self, answers):
        self.answers = answers
        self.histories = []

    async def query_text(self, text, session=None, language="en-IN", channel=None):
        self.histories.append(list(session.history))
        return {"answer": self.answers[text], "latency": {"total_ms": 900}, "cost_inr": 0.001,
                "tokens": {"input": 10, "output": 5}, "sources": ["faq.md"], "context": "[1] Cleaning: 1,200 rupees."}


class _Judge:
    def __init__(self, verdicts):
        self.verdicts = verdicts

    async def complete(self, messages, max_tokens=None, temperature=None):
        question = messages[1]["content"].split("\n", 1)[0].removeprefix("QUESTION: ")
        return LLMResult(answer=self.verdicts[question], input_tokens=0, output_tokens=0, latency_ms=1)


ANSWER_CASES = [
    EvalCase(id="ok", question="Cleaning price?", evidence=("1,200",), reference="1,200 rupees", answer_keywords=("1200",)),
    EvalCase(id="wrong", question="Braces price?", evidence=("35,000",), reference="35,000 rupees", answer_keywords=("35000",)),
    EvalCase(id="none", question="Wifi password?", type="unanswerable", answerable=False),
    EvalCase(id="made-up", question="CEO?", type="unanswerable", answerable=False),
    EvalCase(id="follow", question="And kids?", type="followup", evidence=("700",), answer_keywords=("700",),
             history=({"role": "user", "content": "Price?"}, {"role": "assistant", "content": "500."})),
]
ANSWERS = {
    "Cleaning price?": "A cleaning costs 1,200 rupees.",
    "Braces price?": "**Braces** cost 20,000 rupees.",
    "Wifi password?": "Sorry, I don't have that information.",
    "CEO?": "The CEO is Jane Doe.",
    "And kids?": "A child's first visit is 700 rupees.",
}


async def test_answer_suite_without_a_judge_uses_keywords_and_refusals():
    agent = _Agent(ANSWERS)
    summary = await evaluate_answers(ANSWER_CASES, agent)
    assert summary["judged"] is False
    rows = {row["id"]: row for row in summary["rows"]}
    assert rows["ok"]["passed"] and rows["follow"]["passed"] and rows["none"]["passed"]
    assert not rows["wrong"]["passed"] and not rows["made-up"]["passed"]
    assert rows["wrong"]["voice_ok"] is False                       # Markdown is not speakable
    assert summary["refusal_accuracy"] == 0.5 and summary["false_refusal_rate"] == 0.0
    assert agent.histories[-1] == list(ANSWER_CASES[-1].history)    # follow-ups get their history


async def test_answer_suite_with_a_judge():
    verdicts = {
        "Cleaning price?": '{"correctness": 1, "faithful": true, "refused": false, "reason": "matches"}',
        "Braces price?": 'Sure! ```json\n{"correctness": 0, "faithful": false, "refused": false, "reason": "wrong price"}\n```',
        "Wifi password?": '{"correctness": 1, "faithful": true, "refused": true, "reason": "declined"}',
        "CEO?": '{"correctness": 0, "faithful": false, "refused": false, "reason": "invented"}',
        "And kids?": "not json at all",
    }
    summary = await evaluate_answers(ANSWER_CASES, _Agent(ANSWERS), judge_client=_Judge(verdicts))
    rows = {row["id"]: row for row in summary["rows"]}
    assert rows["wrong"]["judge"]["reason"] == "wrong price" and not rows["wrong"]["passed"]
    assert "judge" not in rows["follow"] and rows["follow"]["passed"]       # unparseable verdict → keywords
    assert summary["correctness"] == 0.5 and summary["faithfulness"] == 0.5
    assert summary["failed"] == ["wrong", "made-up"]


async def test_agent_errors_are_reported_as_failures():
    class Broken:
        async def query_text(self, *args, **kwargs):
            raise RuntimeError("provider down")

    summary = await evaluate_answers(ANSWER_CASES[:1], Broken())
    assert summary["errors"] == 1 and summary["rows"][0]["error"].startswith("RuntimeError")


@pytest.mark.parametrize("answer,refused", [
    ("Sorry, I don't have that information.", True),
    ("That isn't mentioned in my documents.", True),
    ("I'm not sure about that.", True),
    ("A cleaning costs 1,200 rupees.", False),
    ("We don't offer home visits at the moment.", False),
])
def test_detect_refusal(answer, refused):
    assert detect_refusal(answer) is refused


def test_voice_format():
    assert voice_format_ok("It costs 1,200 rupees. Anything else?")
    assert not voice_format_ok("- item one\n- item two")
    assert not voice_format_ok("See https://example.com")
    assert not voice_format_ok("word " * 90)


def test_parse_judgement_clamps_and_rejects():
    assert parse_judgement('{"correctness": 3, "faithful": true}')["correctness"] == 1.0
    assert parse_judgement('{"correctness": "high"}') is None
    assert parse_judgement("no braces here") is None


# ── Dialog suite, gates, report ──────────────────────────────────────────────

async def test_dialog_suite_passes_and_leaves_real_bookings_alone():
    store.save_booking(cal_booking_uid="real-1", cal_booking_id=0, event_type_id=0, booking_date="2026-04-06",
                       time_ist="10:00", start_utc="", attendee_name="Real", attendee_email="real@example.com")
    summary = await evaluate_dialogs(load_scenarios())
    assert summary["success_rate"] == 1.0, [row["problems"] for row in summary["rows"] if not row["passed"]]
    assert [row["cal_booking_uid"] for row in store.get_bookings()] == ["real-1"]


def test_threshold_checks():
    checks = check_thresholds({"answerable": {"mrr": 0.95}, "success_rate": None}, {"answerable.mrr": 0.9, "success_rate": 1.0})
    assert [check["passed"] for check in checks] == [True, False]


async def test_report_escapes_everything():
    summary = await evaluate_answers(
        [EvalCase(id="x", question="<script>alert(1)</script>", evidence=("a",), answer_keywords=("a",))],
        _Agent({"<script>alert(1)</script>": "<img src=x onerror=alert(1)>"}),
    )
    html = render_html([summary])
    assert "<script>alert" not in html and "<img src=x" not in html and "&lt;script&gt;" in html


def test_dataset_file_is_the_default():
    assert DEFAULT_DATASET.name == "golden.jsonl" and DEFAULT_DATASET.exists()
