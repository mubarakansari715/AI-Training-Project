# Naive RAG vs Advanced RAG: What This System Needs, and Why

This page lists which advanced RAG techniques this system uses and why.
Every number was **measured on our three documents** (reports in
`evaluation/reports/`), not taken from general claims. For the full story
of each problem and fix, see
[RAG_USE_CASES_ISSUES_AND_FIXES.md](RAG_USE_CASES_ISSUES_AND_FIXES.md).

- **Naive RAG:** load → fixed-size chunks (800 characters, 150 overlap) →
  embeddings (MiniLM) → ChromaDB similarity search, top 5 → grounded answer
  with citations.
- **Advanced RAG:** the same base, plus the techniques below.

---

## 1. Naive RAG vs Advanced RAG: what each failure needed

| Use case | Problem with Naive RAG | Advanced technique that fixed it | Naive → Advanced (measured) | Evidence |
|---|---|---|---|---|
| **1. GPO** (82 pages, plain text) | The right paragraph was sometimes missed: similarity search finds the topic, not the exact fact | **Reranking** (cross-encoder) | Right page retrieved: **92% → 100%**; ranking score (MRR): **0.84 → 1.0** | `gpo_generated_baseline_retrieval` vs `GPO-Anual-report_generated` (same 25 questions) |
| **2. GEP** (200 pages, chapters) | Answers came from the wrong section: "executive summary risks" retrieved pp. 52–80, not p. 19; South Asia gave 6.6% instead of 6.3% | **Section-aware search + section labels on chunks** | Right section found: **93% → 100%**; right page cited: **71% → 100%** | `gep_baseline_standard` vs `gep_final` |
| 2. GEP | Comparison questions got evidence for only one region | **One search per named section** | South Asia vs Sub-Saharan Africa: wrong → correct | `gep_baseline_standard` vs `gep_final` |
| 2. GEP | Right section, but the headline number was in a lower-ranked chunk | **Small-to-big** (send the whole page of the best hits) | Correct answers: **80% → 93%** | `gep_sections_standard` vs `gep_final` |
| 2. GEP | Passages from unrelated pages (statistical tables, other chapters) | Section filter | Passages from the answering pages: **60% → 100%** (South Asia question) | Retrieval check, 14 Oct 2026 |
| **3. iMac** (12 pages, charts) | Chart numbers invisible to text extraction: "the split is not in the document" (it is 50% / 5% / 45%) | **Multimodal RAG** (a vision model reads charts and tables from page images) | Correct answers: **86% → 100%** | `imac_baseline` vs `imac_final` |
| 3. iMac | A claim's footnote was never retrieved: "15% recycled content, measured how?" → "not available" | **Footnote linking** (a passage marked `[1]` brings its note along) | Wrong → correct | `imac_baseline` vs `imac_final` |

**Sample questions from the task spec (9 in total), latest run (15 Oct 2026):**

| | Naive RAG | Advanced RAG |
|---|---|---|
| Passed (judge: correct and complete, answer page retrieved) | **7 / 9** | **9 / 9** |
| Context precision (passages from an answering page) | 60% | 86% |
| Faithfulness (claims supported by the passages) | 99% | 100% |
| Average tokens per answer | 1,981 | 2,324 (+17%) |
| Average latency | 1.2 s | 2.5 s |
| Estimated cost, all 9 answers (paid-tier prices) | $0.00221 | $0.00237 (+7%) |

Advanced RAG fixed the two Naive failures: the GEP "main risks" answer
(Naive retrieved no passage from the executive summary) and the iMac carbon
footprint split (a chart value). Answer model `openai/gpt-oss-20b` (Groq),
judge `gemini-3.5-flash-lite`, the same on both sides; reports in
`evaluation/reports/{gpo,gep,imac}_compare.json`.

**Answers on the full hand-checked sets (42 questions, the field guide's main measure):**
`python -m evaluation.compare --full`, answers `openai/gpt-oss-20b`, judge `gemini-3.5-flash-lite`,
the same on both sides; reports `evaluation/reports/{gpo,gep,imac}_full_compare.json`.

| Document | Naive RAG passed | Advanced RAG passed | Context precision (Naive → Advanced) |
|---|---|---|---|
| GPO (13 questions) | 13 | 13 | 55% → 69% |
| GEP (15 questions) | 8 | **15** | 56% → 91% |
| iMac (14 questions) | 12 | **14** | 44% → 72% |
| **All 42** | **33 (79%)** | **42 (100%)** | 52% → 78% |

Advanced RAG fixed 9 questions and broke none: the GEP executive-summary
risks, four regional growth forecasts, the South Asia vs Sub-Saharan Africa
comparison and the low-income-countries box (Naive mixed in another
region's figures or missed the section), and the iMac carbon split and
transport/end-of-life shares (chart values). Cost of the extra steps:
latency 1.3 s → 2.9 s per answer; tokens about the same (2,253 → 2,238,
since passages are compressed); estimated cost for all 42 answers $0.0123 →
$0.0101 at paid-tier prices (all runs here used free tiers). Per the field
guide, differences under about 0.1 (10 points) on sets this size are noise:
GPO's 13 = 13 is a tie; GEP's 8 → 15 and iMac's 12 → 14 are not.

**Retrieval on the full hand-checked sets (42 questions, no LLM calls):**
`python -m evaluation.run --questions evaluation/questions/<doc>_full.json --retrieval-only`

| Document | Naive MRR | Advanced MRR | Answer page found (Naive → Advanced) |
|---|---|---|---|
| GPO (13 questions) | 0.84 | **0.94** | 100% → 100% |
| GEP (15 questions) | 0.69 | **1.00** | 93% → 100% |
| iMac (14 questions) | 1.00 | 1.00 | 100% → 100% |

iMac's gap is not retrieval (the answer page is found) but reading it: the
chart values are only in the page image, so Naive answers "not in the document".

Checked against the field guide's techniques (15 Oct 2026):

| Technique | Result on these documents | Kept? |
|---|---|---|
| Ligature repair ("ﬁscal" → "fiscal") | GEP had ~1,560 ligature characters; Advanced MRR 0.96 → **1.00** | ✅ Always on |
| Sentence chunking (whole sentences, sentence overlap) | MRR unchanged on all three (GPO 0.94, GEP 1.00, iMac 1.00) | Option (`CHUNK_STRATEGY=sentence`), off by default |
| Reranker on the visual report | iMac MRR 1.00 with and without it, so no harm here (tables reach it as Markdown) | ✅ On |
| Two-column reading order | GPO has 16 two-column pages, but pypdf already reads them a column at a time | Not needed |
| Chunk size 300 vs 800 characters | MRR not better at 300 on any document (GPO 0.94 vs 0.94, GEP 1.00 vs 1.00, iMac 0.89 vs 1.00) | 800 by default; 300/500/custom on the results page |
| Sentence packing across pages, sized in tokens or characters (the guide's rule) | No gain (GPO 0.92 at 800, 0.90 at 200 tokens; GEP and iMac 1.00) | Option (`CHUNK_STRATEGY=sentence`, `CHUNK_UNIT=tokens`) |
| Structure from font sizes | GEP MRR 0.96 vs 1.00 with bookmarks | Fallback when a PDF has no bookmarks (`STRUCTURE_SOURCE=auto`) |
| Tables from header bars + word positions | All 3 iMac tables rebuilt exactly, no vision model; same key numbers in context as with vision | Fallback when the vision model doesn't read a PDF (`LAYOUT_TABLES=auto`) |
| LLM routing to sections | GEP MRR 1.00, tied with rules (the questions name their regions); fewer passages (4.1 vs 4.8) | Option (`QUERY_ROUTING=llm`); rules by default (no LLM call) |
| bge-small embeddings (the guide's model) | GPO 0.94 and GEP 1.00 (tie), iMac 0.96 vs 1.00 | MiniLM kept |

---

## 2. Which advanced techniques this system needs

| Technique | Needed? | Justification |
|---|---|---|
| Reranking | ✅ **Yes** | Measured: retrieval 92% → 100%. No extra LLM cost (local model) |
| Section-aware search + section labels | ✅ **Yes** | Measured: the core fix for long reports (93% → 100% retrieval, 71% → 100% citations) |
| Small-to-big (whole pages) | ✅ **Yes** | Measured: 80% → 93% correct on GEP |
| Multimodal (charts and tables) | ✅ **Yes** | Measured: the only way to answer chart questions (86% → 100%) |
| Footnote linking | ✅ **Yes** | Measured: fixed the footnote question; no LLM cost |
| Query rewriting | ⚠️ **Keep, not proven** | Useful for typos and follow-ups in chat; no clean before/after measurement on these documents yet. The comparison page measures it per question |
| Agentic RAG (auto mode) | ⚠️ **Optional** | More LLM calls per question; on GPO it raised correct citations 83% → 100% at the same accuracy (`gpo_groq_standard` vs `gpo_groq_auto_mode`). Not needed for single-fact questions |
| Model routing (fast vs strong model) | ⚠️ **Cost saver only** | Saves tokens on easy questions; no accuracy claim |
| **Hybrid search** (BM25 keyword + vector) | ✅ **Yes** | Measured on 25 GPO questions without the reranker: right page found 88% → 96%, MRR 0.77 → 0.91. With the reranker on, hit rate and MRR are unchanged but more passages come from answering pages. No LLM cost |
| **Context compression** (relative reranker cutoff, margin 5) | ✅ **Yes** | ~15% fewer passages on GPO (4.9 → 4.0 per question) with no lost answer page on GPO or GEP, and no failed answer in the 9-question run |
| **Query decomposition** | ✅ **Yes** | Splits comparison/multi-part questions only (3 of 9). GPO FDsys question: context precision 60% → 100%; iMac carbon split: fail → pass. One extra short LLM call when it triggers |
| **Cost estimates** | ✅ **Yes** | Token usage priced at paid-tier list prices: Advanced costs ~7% more than Naive on the 9 questions |
| Multi-query / RAG-Fusion, HyDE | ❌ Not now | Retrieval already finds the right section 100% of the time; these would add LLM cost for no measured gain |
| Semantic chunking | ❌ Not now | Section labels plus whole pages already solve the chunk-boundary problem |
| GraphRAG | ❌ No | These are single reports, not a web of linked entities |

---

## 3. Summary

We started from plain RAG. On our three documents it failed on exactly
three things: fine-grained facts, long structured reports, and charts and
footnotes. Each advanced technique we added fixed one measured failure,
and a second round added hybrid search, context compression and query
decomposition, each kept only after a before/after measurement. On all 42
hand-checked questions Advanced RAG passes 42 (Naive: 33), fixing 9 and
breaking none, with 78% of its passages from answering pages (Naive: 52%),
for about the same tokens and 1.6 s more latency per answer. Every
technique in the Advanced RAG field guide is built and measured; the ones
that did not beat the default on these documents (sentence chunking, 300
character chunks, font-based structure, layout tables, LLM routing,
bge-small) are kept as switches or automatic fallbacks. We did not add techniques that showed no measured
need (multi-query, query expansion, OCR, semantic chunking).

## Caveats

- **Question sets:** 13–15 hand-checked questions per document (42 in all,
  the field guide used 37), plus a 25-question generated GPO set, graded by an LLM judge (pinned model, temperature 0), so they
  are model-based estimates, not verified ground truth.
- **Model for the latest run:** the 9-question run used `openai/gpt-oss-20b`
  (Groq's daily token limit for `gpt-oss-120b` was used up) with a Gemini
  judge. Earlier reports used `gpt-oss-120b`, so compare pass rates within a
  run, not across runs.
- **The GEP "main risks" answer** now passes, but earlier runs showed it can
  leave out a point that is in its context; one run is not proof that this is fixed for good.
- **Query rewriting and model routing** have no isolated measurement yet.
  The Naive vs Advanced page (Use case results → Apply) measures rewriting
  per question.
