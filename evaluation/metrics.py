"""Scores for comparing two RAG pipelines on one question.

Three kinds of numbers, never mixed up:

- Measured: retrieval hit and rank, citation check, passage counts,
  latency and token usage, read from what the pipeline actually did.
- Reference-based: context precision, from the pages a human marked as
  answering the question (`expected_pages` in the questions file).
- Model estimates (LLM judge): faithfulness, answer relevance and context
  recall, RAGAS-style. The judge counts claims and facts, and the score is
  the ratio, but the judgement itself is a model's, not verified truth.
"""

from __future__ import annotations

import json
import re
import time

from app.config.settings import settings

_GROUNDING_ATTEMPTS = 3
_GROUNDING_RETRY_SECONDS = 5.0
# Bounds the judge prompt; passages beyond this are not shown to the judge.
_GROUNDING_CONTEXT_CHARS = 12_000

GROUNDING_PROMPT = """You are checking how well a document question-answering system's answer is grounded.

Question: {question}

Reference answer (written by a person from the document): {reference}

Passages the system was given (numbered, with their page):
{passages}

System answer: {answer}

Do three counts, using only the passages above:
1. Split the system answer into its factual claims about the document (ignore citations,
   greetings and statements that something was not found). "claims" = how many there are;
   "supported" = how many the passages state or directly imply; "unsupported" = the claims
   that are not supported, each in a few words.
2. "relevance": how directly the system answer addresses the question, whether or not it is
   correct: 1 = off-topic or a refusal, 3 = partly, 5 = directly and fully.
3. Split the reference answer into its key facts. "facts" = how many there are; "covered" =
   how many of them the passages contain. If the reference says the document does not contain
   the answer, facts = 0 and covered = 0.

Reply with JSON only:
{{"claims": <int>, "supported": <int>, "unsupported": [<str>, ...], "relevance": <1-5>,
"relevance_reason": "<one sentence>", "facts": <int>, "covered": <int>}}"""


def context_precision(passages: list[dict], expected_pages: list[int]) -> float | None:
    """Share of the passages given to the model that come from a page that
    answers the question; None when the question has no answering pages."""
    if not expected_pages:
        return None
    if not passages:
        return 0.0
    return sum(p.get("expected", p["page"] in expected_pages) for p in passages) / len(passages)


def parse_grounding(output: str) -> dict:
    """Turn the judge's counts into 0-1 scores (None where nothing was counted)."""
    match = re.search(r"\{.*\}", output or "", re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        return {"faithfulness": None, "answer_relevance": None, "context_recall": None, "unsupported": [],
                "relevance_reason": "", "error": f"Unreadable judge reply: {(output or '')[:120]!r}"}

    def count(key: str) -> int:
        try:
            return max(0, int(data.get(key) or 0))
        except (TypeError, ValueError):
            return 0

    claims, facts = count("claims"), count("facts")
    supported, covered = min(count("supported"), claims), min(count("covered"), facts)
    try:
        relevance = min(5, max(1, int(data.get("relevance"))))
    except (TypeError, ValueError):
        relevance = None
    return {
        "faithfulness": supported / claims if claims else None,
        "answer_relevance": (relevance - 1) / 4 if relevance is not None else None,
        "context_recall": covered / facts if facts else None,
        "claims": claims, "supported": supported, "facts": facts, "covered": covered,
        "unsupported": [str(u) for u in (data.get("unsupported") or [])][:5],
        "relevance_reason": str(data.get("relevance_reason") or ""),
        "error": None,
    }


def grade_grounding(generator, question: str, reference: str, answer: str, passages: list[dict],
                    model_name: str) -> dict:
    """One judge call for faithfulness, answer relevance and context recall.
    An empty or unreadable reply is retried, like the correctness judge."""
    context, used = [], 0
    for p in passages:
        block = f"[{p['rank']}] page {p['page']}: {p['text']}"
        if used + len(block) > _GROUNDING_CONTEXT_CHARS:
            break
        context.append(block)
        used += len(block)
    prompt = GROUNDING_PROMPT.format(question=question, reference=reference, answer=answer,
                                     passages="\n\n".join(context) or "(none)")
    gemini = model_name.startswith("gemini")
    result = parse_grounding("")
    for attempt in range(1, _GROUNDING_ATTEMPTS + 1):
        result = parse_grounding(generator.complete(
            prompt, temperature=0.0,
            model_name=model_name if gemini else None, groq_model_name=None if gemini else model_name,
        ))
        if not result["error"]:
            return result
        if attempt < _GROUNDING_ATTEMPTS:
            time.sleep(_GROUNDING_RETRY_SECONDS)
    return result


# ------------------------------------------------------------- comparison


def change(before: float | None, after: float | None) -> tuple[float, float | None] | None:
    """(percentage points, relative %) for two 0-1 scores; relative is None
    when the baseline is 0. None when either score is missing."""
    if before is None or after is None:
        return None
    points = round((after - before) * 100, 1)
    relative = round((after - before) / before * 100, 1) if before else None
    return points, relative


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.0%}"


def _change_text(before: float | None, after: float | None) -> str:
    delta = change(before, after)
    if delta is None:
        return "N/A"
    points, relative = delta
    if points == 0:
        return "no change"
    text = f"{points:+.1f} pp"
    return text + (f" ({relative:+.1f}% relative)" if relative is not None else "")


def _number_change(before: float | None, after: float | None, unit: str = "", digits: int = 0) -> str:
    if before is None or after is None:
        return "N/A"
    diff = after - before
    if round(diff, digits) == 0:
        return "no change"
    text = f"{diff:+,.{digits}f}{unit}"
    return text + (f" ({diff / before * 100:+.1f}% relative)" if before and not unit else "")


def _hit_text(side: dict) -> str:
    if side["hit"] is None:
        return "N/A (no answering page)"
    return f"Yes, rank {side['rank']}" if side["hit"] else "No"


def _yes_no(value: bool | None) -> str:
    return "N/A" if value is None else ("Yes" if value else "No")


def _count(value: int | None) -> str:
    return "N/A" if value is None else str(value)


def _tokens(side: dict) -> int | None:
    if side["prompt_tokens"] is None:
        return None
    return side["prompt_tokens"] + side["completion_tokens"]


def estimate_cost(usage: list[dict], prices: dict[str, tuple[float, float]] | None = None) -> float | None:
    """USD for the LLM calls in `usage` (generator.usage entries) at paid-tier
    list prices (settings.model_prices). None when there were no calls or a
    model has no price, rather than a partial total that looks complete."""
    prices = settings.model_prices if prices is None else prices
    if not usage or any(u["model"] not in prices for u in usage):
        return None
    return sum(u["prompt_tokens"] * prices[u["model"]][0] + u["completion_tokens"] * prices[u["model"]][1]
               for u in usage) / 1_000_000


def _money(value: float | None) -> str:
    return "N/A" if value is None else f"${value:.5f}"


def _money_change(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "N/A"
    return f"{'+' if after >= before else '-'}${abs(after - before):.5f}"


def metric_rows(before: dict, after: dict) -> list[dict]:
    """The before-vs-after table: one row per metric, with how it is measured."""
    judge = "Model estimate (LLM judge)"

    def row(metric, naive, advanced, delta, how):
        return {"Metric": metric, "Naive RAG": naive, "Advanced RAG": advanced, "Change": delta,
                "How it is measured": how}

    tokens_before, tokens_after = _tokens(before), _tokens(after)
    return [
        row("Correct and complete", _yes_no(before["correct"]), _yes_no(after["correct"]),
            "same" if before["correct"] == after["correct"] else ("fixed" if after["correct"] else "broken"),
            f"{judge}: compared with the reference answer; any missing key fact fails it."),
        row("Answer relevance", _pct(before["answer_relevance"]), _pct(after["answer_relevance"]),
            _change_text(before["answer_relevance"], after["answer_relevance"]),
            f"{judge}: how directly the answer addresses the question (1-5, shown as 0-100%), "
            "whether or not it is correct."),
        row("Faithfulness to the document", _pct(before["faithfulness"]), _pct(after["faithfulness"]),
            _change_text(before["faithfulness"], after["faithfulness"]),
            f"{judge}: supported claims / all factual claims in the answer, checked against the "
            "passages it was given. N/A when the answer makes no factual claim."),
        row("Context precision", _pct(before["context_precision"]), _pct(after["context_precision"]),
            _change_text(before["context_precision"], after["context_precision"]),
            "Reference pages: passages from a page that answers the question / all passages given to "
            "the model. N/A when the question has no answering page."),
        row("Context recall", _pct(before["context_recall"]), _pct(after["context_recall"]),
            _change_text(before["context_recall"], after["context_recall"]),
            f"{judge}: key facts of the reference answer found in the passages / all its key facts. "
            "Needs the reference answer; N/A when the document has no answer."),
        row("Retrieval hit", _hit_text(before), _hit_text(after),
            "same" if before["hit"] == after["hit"] else ("found" if after["hit"] else "lost"),
            "Measured: was a passage from a page that answers the question given to the model, and at "
            "what rank."),
        row("Citation correctness", _yes_no(before["citation_ok"]), _yes_no(after["citation_ok"]),
            "same" if before["citation_ok"] == after["citation_ok"] else ("fixed" if after["citation_ok"] else "broken"),
            "Measured: does the answer cite at least one page that answers the question."),
        row("Query optimization", "Original question" if not before["rewritten"] else f"\"{before['search_query']}\"",
            "Original question" if not after["rewritten"] else f"\"{after['search_query']}\"",
            "rewritten" if after["rewritten"] and not before["rewritten"] else "same",
            "Measured: the query actually sent to search (Advanced RAG rewrites it with the LLM)."),
        row("Retrieved passages", str(before["passages_count"]), str(after["passages_count"]),
            _number_change(before["passages_count"], after["passages_count"]),
            "Measured: passages handed to the model. More is not better by itself; see the next row."),
        row("Passages not from an answering page", _count(before["irrelevant_passages"]),
            _count(after["irrelevant_passages"]),
            _number_change(before["irrelevant_passages"], after["irrelevant_passages"]),
            "Reference pages: passages from pages that do not answer the question (lower is better)."),
        row("Response latency", f"{before['latency_s']:.2f} s", f"{after['latency_s']:.2f} s",
            _number_change(before["latency_s"], after["latency_s"], " s", 2),
            "Measured: wall-clock time to answer (query rewriting, search, answer generation)."),
        row("Retrieval latency", f"{before['retrieval_latency_s']:.2f} s", f"{after['retrieval_latency_s']:.2f} s",
            _number_change(before["retrieval_latency_s"], after["retrieval_latency_s"], " s", 2),
            "Measured: time of the search alone (embedding, filtering, reranking, attachments)."),
        row("Token usage", "N/A" if tokens_before is None else f"{tokens_before:,}",
            "N/A" if tokens_after is None else f"{tokens_after:,}", _number_change(tokens_before, tokens_after),
            "Measured: prompt + completion tokens the provider reported for answering (query "
            "rewriting and answer); the judge's tokens are not included."),
        row("Estimated cost", _money(before.get("cost_usd")), _money(after.get("cost_usd")),
            _money_change(before.get("cost_usd"), after.get("cost_usd")),
            "Estimate: the token usage above at the model's paid-tier list price (MODEL_PRICES); the app "
            "runs on free tiers, so this is what the same calls would cost on a paid plan. N/A when the "
            "model has no price."),
    ]


def _status(before: float | None, after: float | None, higher_is_better: bool = True) -> str:
    if before is None or after is None:
        return "n/a"
    if round(after - before, 6) == 0:
        return "same"
    return "improved" if (after > before) == higher_is_better else "regressed"


def _rank_status(rank_without: int | None, rank_with: int | None) -> str:
    """Lower rank is better; not retrieved is worse than any rank."""
    worst = 10 ** 6
    a, b = rank_without or worst, rank_with or worst
    return "same" if a == b else ("improved" if b < a else "regressed")


def _rank_text(rank: int | None) -> str:
    return f"rank {rank}" if rank else "not retrieved"


def _effect(effect: dict | None, what: str) -> dict:
    if not effect:
        return {"text": f"N/A: Advanced RAG did not use {what} for this question.", "status": "n/a"}
    status = _rank_status(effect["rank_without"], effect["rank_with"])
    if status == "same":
        status = _status(effect["precision_without"], effect["precision_with"])
    text = (f"Answer page {_rank_text(effect['rank_without'])} without {what} → {_rank_text(effect['rank_with'])} "
            f"with it; context precision {_pct(effect['precision_without'])} → {_pct(effect['precision_with'])}.")
    return {"text": text, "status": status}


def summary_points(before: dict, after: dict) -> list[dict]:
    """Plain answers to the before/after questions, with the actual numbers."""
    points = []

    def add(question: str, text: str, status: str) -> None:
        points.append({"question": question, "text": text, "status": status})

    retrieval = _rank_status(before["rank"], after["rank"]) if before["hit"] is not None else "n/a"
    if retrieval in ("same", "n/a"):
        retrieval = _status(before["context_precision"], after["context_precision"])
    add("Did Advanced RAG retrieve more relevant information?",
        f"Answer page {_rank_text(before['rank'])} → {_rank_text(after['rank'])}; context precision "
        f"{_pct(before['context_precision'])} → {_pct(after['context_precision'])} "
        f"({_change_text(before['context_precision'], after['context_precision'])}); passages not from an "
        f"answering page {_count(before['irrelevant_passages'])} → {_count(after['irrelevant_passages'])}.", retrieval)
    add("Did the answer become more faithful to the document?",
        f"Faithfulness {_pct(before['faithfulness'])} → {_pct(after['faithfulness'])} "
        f"({_change_text(before['faithfulness'], after['faithfulness'])}).",
        _status(before["faithfulness"], after["faithfulness"]))
    add("Did answer relevance improve?",
        f"Answer relevance {_pct(before['answer_relevance'])} → {_pct(after['answer_relevance'])} "
        f"({_change_text(before['answer_relevance'], after['answer_relevance'])}).",
        _status(before["answer_relevance"], after["answer_relevance"]))
    rewrite = _effect(after["rewrite_effect"], "query rewriting")
    if after["rewritten"]:
        rewrite["text"] = f"Rewritten to \"{after['search_query']}\". " + rewrite["text"]
    add("Did query rewriting help?", rewrite["text"], rewrite["status"])
    if after.get("sub_queries"):
        decompose = _effect(after.get("decompose_effect"), "query decomposition")
        decompose["text"] = f"Split into {len(after['sub_queries'])} sub-questions. " + decompose["text"]
    else:
        decompose = _effect(None, "query decomposition")
    add("Did query decomposition help?", decompose["text"], decompose["status"])
    keyword = _effect(after.get("keyword_effect"), "keyword search (BM25)")
    add("Did hybrid keyword search help?", keyword["text"], keyword["status"])
    rerank = _effect(after["rerank_effect"], "reranking")
    add("Did reranking improve the final context?", rerank["text"], rerank["status"])
    # Latency and tokens are what the extra steps cost, not answer quality:
    # more of them is "cost", never "regressed".
    latency_change = _number_change(before["latency_s"], after["latency_s"], " s", 2)
    add("Did latency increase or decrease?",
        f"{before['latency_s']:.2f} s → {after['latency_s']:.2f} s ({latency_change}).",
        _cost_status(before["latency_s"], after["latency_s"]))
    tokens_before, tokens_after = _tokens(before), _tokens(after)
    token_change = _number_change(tokens_before, tokens_after)
    cost_before, cost_after = before.get("cost_usd"), after.get("cost_usd")
    cost_text = (f"estimated cost {_money(cost_before)} → {_money(cost_after)} "
                 f"({_money_change(cost_before, cost_after)}) at paid-tier prices"
                 if cost_before is not None and cost_after is not None
                 else "cost not available (no price for this model in MODEL_PRICES)")
    add("Did token usage or estimated cost increase or decrease?",
        f"Tokens {tokens_before if tokens_before is not None else 'N/A'} → "
        f"{tokens_after if tokens_after is not None else 'N/A'} "
        f"({token_change}); {cost_text}.",
        _cost_status(tokens_before, tokens_after))

    # The overall verdict is about the answer. Retrieval, rewriting and
    # reranking are how it was produced (reranking is part of the retrieval
    # change, so counting both would count one passage swap twice).
    quality = [_correct_status(before["correct"], after["correct"]),
               _status(before["faithfulness"], after["faithfulness"]),
               _status(before["answer_relevance"], after["answer_relevance"])]
    counts = {s: quality.count(s) for s in ("improved", "regressed", "same")}
    overall = "mixed" if counts["improved"] and counts["regressed"] else (
        "improved" if counts["improved"] else ("regressed" if counts["regressed"] else "same"))
    verdict = f"Judge verdict {'correct' if before['correct'] else 'incorrect'} → {'correct' if after['correct'] else 'incorrect'}."
    add("Overall, compared with Naive RAG?",
        f"{verdict} Answer quality (correctness, faithfulness, relevance): {counts['improved']} improved, "
        f"{counts['regressed']} regressed, {counts['same']} the same. Retrieval {_STATUS_WORDS[retrieval]}. "
        f"Cost of the extra steps: {latency_change} latency, {token_change} tokens"
        f"{', ' + _money_change(cost_before, cost_after) if cost_before is not None and cost_after is not None else ''}. "
        "One question, so this is an example, not an average.", overall)
    return points


_STATUS_WORDS = {"improved": "improved", "regressed": "got worse", "same": "unchanged", "n/a": "not measurable",
                 "mixed": "mixed"}


def _cost_status(before: float | None, after: float | None) -> str:
    """Latency or tokens: "cost" when Advanced RAG spends more, "improved" when less."""
    status = _status(before, after, higher_is_better=False)
    return "cost" if status == "regressed" else status


def _correct_status(before: bool | None, after: bool | None) -> str:
    if before is None or after is None:
        return "n/a"
    return "same" if before == after else ("improved" if after else "regressed")
