from types import SimpleNamespace

from app.rag.retriever import RetrievedChunk
from app.services.rag_service import AskResult
from evaluation.compare import ask_side, describe_steps
from evaluation.harness import EvalConfig, Throttle


def _chunk(page, text="t", section=None):
    return RetrievedChunk(text=text, source="r.pdf", page=page, score=0.5, chunk_id=f"c{page}{text[:3]}",
                          section=section)


def _pipeline(chunks, named=(), reranker=None, answer="South Asia: 6.3 percent (r.pdf, page 96)."):
    retriever = SimpleNamespace(
        section_aware=bool(named), reranker=reranker, rerank_candidates=20, top_k=5,
        _section_filter=lambda q: ([list(g) for g in named], None),
        retrieve=lambda q: chunks,
    )
    return SimpleNamespace(
        retriever=retriever,
        vector_store=SimpleNamespace(count=lambda: 531),
        ask=lambda question, **kwargs: AskResult(answer=answer, search_query="growth outlook South Asia 2026"),
        generator=SimpleNamespace(complete=lambda prompt, **kwargs: '{"verdict": "correct", "reason": "Matches."}'),
    )


def test_steps_describe_what_the_pipeline_did():
    chunks = [_chunk(96), _chunk(4, "[Footnote 1, referenced on page 4] Mass of recycled material."),
              _chunk(10, "[Chart on page 10: Footprint (read from the page image)]\nProduction 50%")]
    steps = describe_steps(_pipeline(chunks, named=[["Ch. 2 › South Asia"]], reranker=object()),
                           "What is the outlook for South Asia?", "growth outlook South Asia 2026", chunks)

    assert [s["kind"] for s in steps] == ["Query", "Retrieve", "Retrieve", "Post-process", "Post-process",
                                          "Post-process", "Generate"]
    assert "Ch. 2 › South Asia" in steps[1]["text"]
    assert "cross-encoder" in steps[2]["text"] and "531 chunks" in steps[2]["text"]
    assert steps[-1]["text"].startswith("Grounded prompt with 3 numbered passages")


def test_plain_search_has_no_section_or_rerank_steps():
    steps = describe_steps(_pipeline([_chunk(5)]), "How many?", "How many?", [_chunk(5)])
    assert [s["kind"] for s in steps] == ["Retrieve", "Generate"]
    assert steps[0]["text"] == "Similarity search over 531 chunks: top 5"


def test_side_keeps_answer_verdict_rank_and_every_passage():
    chunks = [_chunk(189, "SAR table"), _chunk(96, "Growth in SAR is expected to soften to 6.3 percent")]
    question = {"id": "q", "question": "What is the outlook for South Asia?", "reference_answer": "6.3 percent.",
                "expected_pages": [96, 97]}

    side = ask_side(_pipeline(chunks), question, "After", EvalConfig(), "m", "j", Throttle(None))

    assert side["correct"] is True and side["judge_reason"] == "Matches."
    assert side["rank"] == 2  # the answering page was the second passage
    assert [(p["rank"], p["page"], p["expected"]) for p in side["passages"]] == [(1, 189, False), (2, 96, True)]
