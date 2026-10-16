from types import SimpleNamespace

import pytest

from app.rag.generator import FALLBACK_MESSAGE
from app.rag.retriever import RetrievedChunk
from app.services.rag_service import AskResult
from evaluation.compare import PipelineFailed, describe_steps, question_options, resumable_rows, run_side, side_config
from evaluation.harness import Throttle

JUDGE = '{"verdict": "correct", "reason": "Matches."}'
GROUNDING = '{"claims": 2, "supported": 2, "unsupported": [], "relevance": 5, "facts": 1, "covered": 1}'


def _chunk(page, text="t", section=None):
    return RetrievedChunk(text=text, source="r.pdf", page=page, score=0.5, chunk_id=f"c{page}{text[:3]}",
                          section=section)


class FakeRetriever:
    """Returns `by_query[query]`, or a different order once the reranker is off."""

    def __init__(self, by_query, without_rerank=None, reranker=None, named=(), keyword_index=None):
        self.by_query, self.without_rerank, self.keyword_index = by_query, without_rerank, keyword_index
        self.reranker, self.rerank_candidates, self.top_k = reranker, 20, 5
        self.section_aware, self._named = bool(named), [list(g) for g in named]
        self.last_stats = {}

    def _section_filter(self, query):
        return self._named, None

    def retrieve_many(self, queries):
        chunks = [c for q in queries for c in self.retrieve(q)]
        return list({c.chunk_id: c for c in chunks}.values())

    def retrieve(self, query):
        chunks = self.by_query[query] if self.reranker or self.without_rerank is None else self.without_rerank
        self.last_stats = {"searches": 1, "candidates": 7, "keyword_only": 2, "kept": len(chunks), "footnotes": 0,
                           "whole_pages": False, "passages": len(chunks)}
        return chunks


def _pipeline(retriever, answers=("South Asia: 6.3 percent (r.pdf, page 96).",), search_query=None,
              rewrite=True, sub_queries=()):
    answers, replies = list(answers), [JUDGE, GROUNDING]
    generator = SimpleNamespace(usage=[], complete=lambda prompt, **kwargs: replies.pop(0))

    def ask(question, **kwargs):
        generator.usage.append({"model": "m", "prompt_tokens": 1500, "completion_tokens": 100})
        return AskResult(answer=answers.pop(0), search_query=search_query or question, sub_queries=list(sub_queries))

    return SimpleNamespace(retriever=retriever, vector_store=SimpleNamespace(count=lambda: 531),
                           ask=ask, generator=generator, rewrite_queries=rewrite, decompose_queries=bool(sub_queries))


QUESTION = {"id": "q", "question": "What is the outlook for South Asia?", "reference_answer": "6.3 percent.",
            "expected_pages": [96, 97]}


def test_question_options_put_main_questions_first_and_label_groups():
    options = question_options({"questions": [{"id": "x"}, {"id": "s", "sample": True}, {"id": "y"}]})
    assert [(o["id"], o["group"]) for o in options] == [("s", "main"), ("x", "related"), ("y", "related")]


def test_naive_side_switches_every_advanced_technique_off():
    naive, advanced = side_config("before"), side_config("after")
    assert (naive.reranker_model, naive.section_aware, naive.read_visuals, naive.rewrite_queries,
            naive.hybrid_search, naive.decompose_queries) == ("", False, False, False, False, False)
    assert advanced.rewrite_queries and advanced.reranker_model


def test_side_measures_retrieval_grounding_tokens_and_technique_effects():
    rewritten = "growth outlook South Asia 2026"
    retriever = FakeRetriever(
        {rewritten: [_chunk(96), _chunk(189)], QUESTION["question"]: [_chunk(189), _chunk(5)]},
        without_rerank=[_chunk(189), _chunk(96)], reranker=object())

    side = run_side(_pipeline(retriever, search_query=rewritten), QUESTION, "after", side_config("after"),
                    "m", "j", Throttle(None))

    assert side["passed"] and side["rank"] == 1 and side["citation_ok"] is True
    assert side["context_precision"] == 0.5 and side["irrelevant_passages"] == 1
    assert side["faithfulness"] == 1.0 and side["answer_relevance"] == 1.0 and side["context_recall"] == 1.0
    assert (side["prompt_tokens"], side["completion_tokens"]) == (1500, 100)
    assert side["cost_usd"] is None  # the fake model "m" has no price
    assert side["rewritten"] and side["search_query"] == rewritten
    # Without the rewrite the answer page was not found; without reranking it was 2nd.
    assert side["rewrite_effect"]["rank_without"] is None
    assert side["rerank_effect"]["rank_without"] == 2 and side["rerank_effect"]["rank_with"] == 1
    assert [s["kind"] for s in side["steps"]] == ["Query", "Retrieve", "Post-process", "Generate"]


def test_fail_when_the_answer_page_was_not_retrieved_even_if_judged_correct():
    retriever = FakeRetriever({QUESTION["question"]: [_chunk(189)]})
    side = run_side(_pipeline(retriever, rewrite=False), QUESTION, "before", side_config("before"),
                    "m", "j", Throttle(None))
    assert side["correct"] and not side["passed"] and side["hit"] is False
    assert side["rewrite_effect"] is None and side["rerank_effect"] is None


def test_pipeline_that_never_answers_raises_a_clear_error(monkeypatch):
    monkeypatch.setattr("evaluation.compare.time.sleep", lambda s: None)
    retriever = FakeRetriever({QUESTION["question"]: [_chunk(96)]})
    pipeline = _pipeline(retriever, answers=[FALLBACK_MESSAGE] * 3, rewrite=False)
    with pytest.raises(PipelineFailed, match="Naive RAG"):
        run_side(pipeline, QUESTION, "before", side_config("before"), "m", "j", Throttle(None))


def test_steps_name_the_sections_searched():
    chunks = [_chunk(96)]
    retriever = FakeRetriever({"q": chunks}, named=[["Ch. 2 › South Asia"]])
    steps = describe_steps(_pipeline(retriever), "q", "q", chunks, {"searches": 1, "kept": 1})
    assert "Ch. 2 › South Asia" in steps[0]["text"]


def test_steps_report_what_keyword_search_added():
    pipeline = SimpleNamespace(retriever=FakeRetriever({}, reranker=object(), keyword_index=object()),
                               vector_store=SimpleNamespace(count=lambda: 40))
    stats = {"searches": 1, "candidates": 9, "keyword_only": 2, "kept": 5}

    steps = [s["text"] for s in describe_steps(pipeline, "q", "q", [], stats)]

    assert any("Keyword search (BM25)" in s and "2 candidate(s) found only by exact words" in s for s in steps)
    assert any(s.startswith("Hybrid search over 40 chunks") for s in steps)


def test_decomposed_question_is_retrieved_per_sub_question_and_its_effect_measured():
    q = dict(QUESTION, question="Compare South Asia and Europe growth", expected_pages=[96, 40])
    parts = ["What is South Asia growth?", "What is Europe growth?"]
    retriever = FakeRetriever({parts[0]: [_chunk(96)], parts[1]: [_chunk(40)], q["question"]: [_chunk(96), _chunk(5)]},
                              reranker=object())
    retriever.without_rerank = None

    side = run_side(_pipeline(retriever, sub_queries=parts), q, "after", side_config("after"), "m", "j",
                    Throttle(None))

    assert side["sub_queries"] == parts and side["queries_generated"] == 2
    assert side["context_precision"] == 1.0  # both parts found their page
    assert side["decompose_effect"]["precision_without"] == 0.5  # one search: one part's page only
    assert any("split the question into 2 sub-questions" in s["text"] for s in side["steps"])


def test_resume_reuses_rows_only_from_the_same_models_and_pipeline_configs(tmp_path):
    import json

    path = tmp_path / "x_compare.json"
    configs = {"before": "naive", "after": "advanced | hybrid BM25"}
    path.write_text(json.dumps({"answer_model": "m", "judge_model": "j", "configs": configs, "rows": [{"id": "q"}]}))

    assert resumable_rows(path, "m", "j", configs) == {"q": {"id": "q"}}
    assert resumable_rows(path, "m", "j", {"before": "naive", "after": "advanced"}) == {}  # technique changed
    assert resumable_rows(path, "other", "j", configs) == {}
    assert resumable_rows(tmp_path / "missing.json", "m", "j", configs) == {}
