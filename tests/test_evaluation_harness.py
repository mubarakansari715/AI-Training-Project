import json
from types import SimpleNamespace

import pytest

from app.rag.retriever import RetrievedChunk
from app.services.rag_service import AskResult
from evaluation.harness import (
    JUDGE_PROMPT,
    Throttle,
    evaluate_answers,
    first_hit_rank,
    load_questions,
    parse_verdict,
    summarize_answers,
    summarize_retrieval,
)


def _chunk(page):
    return RetrievedChunk(text="t", source="doc.pdf", page=page, score=0.5, chunk_id=f"c{page}")


def test_first_hit_rank_finds_first_expected_page():
    chunks = [_chunk(9), _chunk(5), _chunk(21)]
    assert first_hit_rank(chunks, [5, 21]) == 2
    assert first_hit_rank(chunks, [40]) is None


@pytest.mark.parametrize(
    ("output", "correct"),
    [
        ('{"verdict": "correct", "reason": "States 166."}', True),
        ('```json\n{"verdict": "Incorrect", "reason": "Wrong number."}\n```', False),
        ("Sure! Here is my grade: {\"verdict\": \"correct\", \"reason\": \"ok\"}", True),
    ],
)
def test_parse_verdict(output, correct):
    assert parse_verdict(output)["correct"] is correct


@pytest.mark.parametrize("output", ["", "looks right to me", '{"verdict": "maybe"}', "{not json"])
def test_unparseable_judge_output_counts_as_incorrect(output):
    verdict = parse_verdict(output)
    assert verdict["correct"] is False
    assert "Unparseable" in verdict["reason"]


def test_throttle_spaces_calls(monkeypatch):
    clock = {"now": 100.0}
    slept = []
    monkeypatch.setattr("evaluation.harness.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr("evaluation.harness.time.sleep", lambda s: (slept.append(s), clock.update(now=clock["now"] + s)))
    throttle = Throttle(calls_per_minute=12)  # one call per 5s

    throttle.wait(calls=2)  # first: no wait, books 10s
    throttle.wait()         # must wait the 10s
    assert slept == [10.0]
    assert Throttle(None).interval == 0.0


def test_evaluate_answers_uses_pipeline_ask_and_judge():
    asked, judged = [], []

    def ask(question, strict_mode, model_name, temperature, groq_model_name, auto_mode):
        asked.append((question, strict_mode, model_name, temperature))
        return AskResult(answer="GPO visited 166 libraries (p. 5).", sources=[_chunk(5)], search_query=question)

    def complete(prompt, temperature, model_name, groq_model_name):
        judged.append((prompt, model_name or groq_model_name, temperature))
        return '{"verdict": "correct", "reason": "Matches."}'

    pipeline = SimpleNamespace(ask=ask, generator=SimpleNamespace(complete=complete))
    questions = [
        {"id": "q1", "question": "How many libraries?", "reference_answer": "166.", "expected_pages": [5, 21]},
        {"id": "trap", "question": "FY 2025 budget?", "reference_answer": "Not in the document.", "expected_pages": []},
    ]

    rows = evaluate_answers(pipeline, questions, model_name="gemini-x", judge_model="judge-y")

    assert asked[0] == ("How many libraries?", True, "gemini-x", 0.0)
    assert judged[0][0] == JUDGE_PROMPT.format(question="How many libraries?", reference="166.",
                                               answer="GPO visited 166 libraries (p. 5).")
    assert judged[0][1:] == ("judge-y", 0.0)
    assert rows[0]["correct"] is True and rows[0]["citation_ok"] is True
    assert rows[1]["citation_ok"] is None  # no expected pages -> citation not scored
    assert summarize_answers(rows) == {"accuracy": 1.0, "citation_accuracy": 1.0}


def test_failed_answer_is_retried_instead_of_graded(monkeypatch):
    """A provider failure (e.g. a per-minute rate limit) isn't a wrong answer:
    the question is asked again after a pause, and only then graded."""
    from app.rag.generator import FALLBACK_MESSAGE

    monkeypatch.setattr("evaluation.harness.time.sleep", lambda s: None)
    answers = iter([FALLBACK_MESSAGE, "GPO visited 166 libraries (p. 5)."])

    def ask(question, **kwargs):
        return AskResult(answer=next(answers), sources=[_chunk(5)], search_query=question)

    graded = []
    pipeline = SimpleNamespace(ask=ask, generator=SimpleNamespace(
        complete=lambda prompt, **kwargs: graded.append(prompt) or '{"verdict": "correct", "reason": "ok"}'))

    rows = evaluate_answers(pipeline, [{"id": "q1", "question": "How many?", "reference_answer": "166.",
                                        "expected_pages": [5]}], judge_model="j")

    assert rows[0]["answer"] == "GPO visited 166 libraries (p. 5)." and len(graded) == 1


def test_summarize_retrieval():
    rows = [{"hit": True, "rank": 1, "retrieved_pages": [1, 2]}, {"hit": True, "rank": 2, "retrieved_pages": [3]},
            {"hit": False, "rank": None, "retrieved_pages": []}]
    assert summarize_retrieval(rows) == {"hit_rate": 0.667, "mrr": 0.5, "avg_chunks": 1.0}


def test_load_questions_requires_ground_truth_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluation.harness.PROJECT_ROOT", tmp_path)
    (tmp_path / "q.json").write_text(json.dumps({"document": "d.pdf", "questions": [{"id": "a", "question": "?"}]}))
    with pytest.raises(ValueError, match="reference_answer"):
        load_questions("q.json")


def test_gpo_questions_file_is_ground_truth_only():
    data = load_questions("evaluation/questions/gpo.json")
    allowed = {"id", "question", "reference_answer", "expected_pages", "sample"}
    for q in data["questions"]:
        assert set(q) <= allowed, f"{q['id']} has non-ground-truth keys: {set(q) - allowed}"
    assert sum(q.get("sample", False) for q in data["questions"]) == 3


def test_judge_retries_empty_replies(monkeypatch):
    from evaluation.harness import grade_answer

    monkeypatch.setattr("evaluation.harness.time.sleep", lambda s: None)
    replies = ["", "not json", '{"verdict": "correct", "reason": "Matches."}']
    gen = SimpleNamespace(complete=lambda prompt, temperature, model_name, groq_model_name: replies.pop(0))

    assert grade_answer(gen, "q", "ref", "ans") == {"correct": True, "reason": "Matches."}
    assert replies == []


def test_judge_gives_up_after_repeated_empty_replies(monkeypatch):
    from evaluation.harness import grade_answer

    monkeypatch.setattr("evaluation.harness.time.sleep", lambda s: None)
    gen = SimpleNamespace(complete=lambda prompt, temperature, model_name, groq_model_name: "")

    verdict = grade_answer(gen, "q", "ref", "ans")
    assert verdict["correct"] is False and verdict["reason"].startswith("Unparseable")


def test_judge_model_is_routed_to_its_provider():
    from evaluation.harness import grade_answer

    calls = []
    gen = SimpleNamespace(complete=lambda prompt, temperature, model_name, groq_model_name:
                          calls.append((model_name, groq_model_name)) or '{"verdict": "correct", "reason": "ok"}')
    grade_answer(gen, "q", "r", "a", "qwen/qwen3.8-27b")
    grade_answer(gen, "q", "r", "a", "gemini-3.5-flash-lite")
    assert calls == [(None, "qwen/qwen3.8-27b"), ("gemini-3.5-flash-lite", None)]

