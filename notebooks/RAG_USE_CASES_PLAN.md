# RAG Hands-On Use Cases — Plan

Three tasks, one document each, ordered from simplest to hardest. Each document
forces a different RAG skill:

| # | Task | Document (in `notebooks/`) | Size | Core skill |
|---|---|---|---|---|
| 1 | Baseline RAG | `GPO-Anual-report.pdf` (GPO 2018 Annual Report) | 82 pages | Clean answers from plain narrative text |
| 2 | Advanced document RAG | `GEP-Jun-2026.pdf` (World Bank Global Economic Prospects) | 200 pages | Long documents with chapters and sections |
| 3 | Multimodal / complex RAG | `iMac_PER_Oct2024.pdf` (Apple iMac Environmental Report) | 12 pages | Facts inside charts, tables, icons and footnotes |

Spec: `RAG-Hands-On-Use-Cases.docx`. Each use case has 3 sample questions and a
pass/fail bar.

---

## Where we start (measured)

ALLY AI already has a full RAG pipeline:

- **Parsing and cleaning:** `pypdf` text extraction, then text cleanup.
- **Chunking:** about 800 characters per chunk, 150-character overlap.
- **Embeddings and storage:** local `all-MiniLM-L6-v2` embeddings in ChromaDB.
- **Search:** top-5 results with a 0.2 similarity threshold, plus query rewriting.
- **Answers:** grounded answers with page citations from OpenRouter, with Gemini as the fallback.

I ran all 9 sample questions through it as-is, with strict mode on and each
document in its own index, then checked every answer against the PDF text:

| Task | Question | Result today |
|---|---|---|
| 1 GPO | Depository libraries visited in FY 2018? | ✅ 166 (p. 5, p. 21) |
| 1 GPO | System that replaced FDsys, when fully deployed? | ⚠️ govinfo, "scheduled to replace FDsys in December 2018" (p. 39). Said "deployment date not specified"; needs a check for other wording |
| 1 GPO | Printing procured for agencies in FY 2018? | ✅ $375.7 million (p. 14; p. 5 says "more than $375 million") |
| 2 GEP | Growth outlook for South Asia? | ⚠️ Numbers came mainly from the statistical tables (p. 189), not the South Asia regional section |
| 2 GEP | What does Chapter 3 say about sovereign debt? | ✅ Plausible, cites pp. 109–141; must confirm those pages are Chapter 3 |
| 2 GEP | Executive summary: main risks? | ❌ Cited pp. 52, 54, 200, not the Executive Summary. **Fails the "right section" bar** |
| 3 iMac | Energy use vs ENERGY STAR? | ✅ 58% less (p. 7) |
| 3 iMac | Total carbon footprint; use vs manufacturing? | ❌ Gave totals, said the breakdown wasn't there. It is: *Production 50% · Transportation 5% · Product use 45%* (p. 10), and the chart on p. 2 extracts as jumbled labels |
| 3 iMac | Stand material and % recycled? | ✅ 100% recycled aluminum (p. 1, p. 4) |

**Score: 5 pass, 2 partial, 2 fail.** The failures are exactly where the
documents are meant to push us: section awareness (task 2) and visual content
(task 3).

### Document facts that shape the plan
- **GPO:** 10 of 82 pages have almost no text (cover, photo pages). Financial
  statements with bookmarks sit at the back.
- **GEP:** has built-in **PDF bookmarks** for each part: Executive Summary,
  Ch. 1–4, Boxes 1.1/1.2/2.1, and each region (East Asia and Pacific, South
  Asia, Sub-Saharan Africa…). That gives the section structure for free. 23
  pages are mostly tables or figures.
- **iMac:** no bookmarks, 14 embedded images (charts and icons), and numbered
  footnotes on p. 11.

---

## Shared setup (do once, before task 1)

1. **One isolated index per document.** Use a separate Chroma collection per
   use case, so documents never mix and each experiment is reproducible.
2. **One notebook per task**, which imports the existing `app/` modules
   (`document_loader`, `chunker`, `embeddings`, `vector_store`, `retriever`,
   `generator`):
   - `01_gpo_baseline.ipynb`
   - `02_gep_structured.ipynb`
   - `03_imac_multimodal.ipynb`

   Improvements that work go back into `app/`, so ALLY AI gets them too.
3. **Evaluation harness.** This is the most important piece; every task is
   "validated with the sample questions".
   - A questions file per document with the question, expected answer
     (key facts or numbers), and expected page(s) or section.
   - A script that runs every question and reports:
     - **Retrieval hit@k:** did the right page or section appear in the
       retrieved chunks?
     - **Answer correctness:** does the answer contain the expected facts
       (keyword check, optionally an LLM-as-judge)?
     - **Citation accuracy:** are the cited pages or sections the right ones?
   - Add **5–10 extra questions per document** beyond the 3 samples
     (especially "confusable" ones), so results aren't judged on 3 data points.
4. **Results log.** Keep a table per task of each configuration tried versus
   its score. This is what you present to the team.

---

## Task 1 — GPO Annual Report: Baseline RAG

> **Status: ✅ done.**
> - **Results:** 13/13 answers correct, 100% page citations, 100% retrieval hit rate with the app's default
>   configuration.
> - **Where:** `notebooks/01_gpo_baseline.ipynb`, with ground truth in `evaluation/questions/gpo.json`. Reproduce
>   with `python -m evaluation.run --questions evaluation/questions/gpo.json`.
> - **Grading:** an AI judge with a pinned model, checked against deliberately wrong answers; no keyword matching.

**Goal:** a solid, well-measured baseline over plain narrative text.
**Pass bar:** the exact fact from the right paragraph, nothing invented.

**Expected answers:**
- **Q1:** 166 depository libraries in 25 states (p. 5, p. 21).
- **Q2:** govinfo replaced FDsys. Search the whole report for "govinfo",
  because p. 39 says "scheduled to replace FDsys in December 2018"; check
  whether the Director's message states full deployment.
- **Q3:** $375.7 million (p. 14), described on p. 5 as "more than $375 million".

**Steps:**
1. **Ingestion:** confirm the 10 low-text pages lose nothing important; if
   they hold infographics, note it as a known gap. Check for repeated
   headers and footers.
2. **Chunking experiment:** compare 800/150 (current) with smaller (500/100)
   and larger (1200/200) chunks on hit@k. Plain text is usually best around
   300–600 words per chunk.
3. **Embeddings:** keep MiniLM as the baseline. Optionally compare one
   stronger small model (e.g. `bge-small-en-v1.5`) on the same questions.
4. **Retrieval:** tune `top_k` (3/5/8) and the similarity threshold. Record
   how each change moves the score.
5. **Generation:** strict mode on, answers must cite pages, and the model may
   not add facts that aren't in the retrieved text.
6. **Deliverable:** notebook + results table + the chosen baseline config.
   This becomes the "before" number for tasks 2 and 3.

---

## Task 2 — World Bank GEP (200 pages): Advanced Document RAG

**Goal:** pull the *right section*, and never mix regions or chapters.
**Pass bar:** right chapter or section, no facts borrowed from another
region or chapter.
**Today's failure:** the "executive summary" question answered from Ch. 1
and appendix pages.

**Steps:**
1. **Read the structure.** Turn the PDF bookmarks into page ranges:
   `Executive Summary → pp. a–b`, `Ch. 2 → South Asia → pp. c–d`, and so on.
   Fall back to heading detection (font size or bold, via PyMuPDF) for
   anything the bookmarks miss.
2. **Section-aware chunking.** Split *within* sections and never across a
   section boundary. Give every chunk metadata: `chapter`, `section`,
   `region`, `is_box`, `is_statistical_table`, `page`.
3. **Contextual chunk headers.** Prefix each chunk's text with its path,
   e.g. `[Ch. 2 Regional Outlooks › South Asia]`. The embedding then "knows"
   the region, which is what keeps South Asia and Sub-Saharan Africa apart.
4. **Metadata-aware retrieval.**
   - When the question names a section ("Chapter 3", "executive summary",
     "South Asia"), filter the search to that section with a ChromaDB
     `where` filter. Otherwise search everything.
   - Treat the statistical appendix as its own section, so tables don't
     crowd out narrative answers unless the question asks for figures.
5. **Small-to-big (parent–child) retrieval.** Search with small chunks for
   precision, then send the surrounding larger passage from the same section
   to the LLM for full context.
6. **Better ranking.**
   - **Hybrid search:** BM25 keyword search (`rank_bm25`) plus vector search,
     which helps with exact terms like "Chapter 3" and region names.
   - **Re-ranking:** re-rank the top ~20 with a cross-encoder
     (`ms-marco-MiniLM-L-6-v2`; works with this Mac's PyTorch 2.2.2).
7. **Context management.** Keep a token budget: deduplicate overlapping
   chunks, order them by page, label each with its section, and cite
   "Ch. 2 › South Asia, p. 83" instead of only page numbers.
8. **Evaluate.** Compare against the task-1-style baseline on the 3 samples
   plus confusable extras (e.g. "growth outlook for East Asia and Pacific" vs
   "for Sub-Saharan Africa", "what does Box 1.1 say about AI?").
9. **Deliverable:** notebook + a before/after table showing which technique
   fixed which question.

---

## Task 3 — Apple iMac Environmental Report: Multimodal / Complex RAG

**Goal:** make facts in charts, tables, icon call-outs and footnotes
searchable.
**Pass bar:** the right number from the chart or table, not a nearby
general sentence.
**Today's failure:** the carbon-footprint breakdown chart (p. 2) extracts as
jumbled labels, and the percentage split on p. 10 was missed.

**Steps:**
1. **Layout-aware parsing.** Replace plain `pypdf` text with **PyMuPDF**
   (`fitz`) and/or **pdfplumber**, so text blocks come out in reading order
   with their positions.
2. **Tables.** Extract with `page.find_tables()` (PyMuPDF) or pdfplumber.
   Store each table as **one chunk in Markdown table form**, with its title
   or caption and page, and never split a table across chunks.
3. **Charts and images.** Render each chart or image region, then ask a
   **vision LLM** to turn it into text with the actual numbers, e.g.
   "Bar chart: carbon footprint 346 kg CO2e: production 50%, transport 5%,
   use 45%…". Our existing Gemini key supports images. Store the result as
   a `chart` chunk with its page.
4. **Icon call-outs.** Keep short call-out blocks ("100% recycled aluminum
   in the stand") as their own small chunks, so they aren't diluted inside
   long text.
5. **Footnotes.** Detect numbered notes (p. 11) and the markers that point
   to them ("Recycled and low-carbon materials⁸"). Attach the footnote text
   to the chunk that references it, so claims keep their conditions.
6. **Metadata.** Tag each chunk with `content_type` = text / table / chart /
   callout / footnote, and show it in citations ("p. 10, chart").
7. **Avoid tools that clash with this Mac.** PyTorch is capped at 2.2.2 on
   Intel macOS, so avoid local layout models that need newer PyTorch
   (`docling`, `unstructured` hi-res). PyMuPDF + pdfplumber + Gemini vision
   avoid that. LlamaParse (cloud API) is an option to compare against.
8. **Evaluate.** Use the 3 samples plus extras taken directly from charts
   and tables (e.g. a specific value from the power-consumption table, a
   footnote condition).
9. **Deliverable:** notebook + before/after table + short notes on which
   parser handled which content type.

---

## Suggested order and what to present

1. **Shared setup** (isolated indexes, questions files, eval script).
2. **Task 1:** establish and document the baseline numbers.
3. **Task 2:** structure-aware improvements, measured against the baseline.
4. **Task 3:** parsing improvements, measured against the baseline.
5. **Final summary** for the team: one table, 3 documents × (baseline vs
   improved), plus the 2–3 techniques that mattered most for each document
   shape.

## Housekeeping

- `notebooks/` is currently **untracked** in git (16 MB of PDFs). Decide
  whether to commit the PDFs (public reports) or list them in `.gitignore`
  and keep only the notebooks, questions files and results in git.
- New dependencies to expect: `pymupdf`, `pdfplumber`, `rank_bm25`, and
  possibly a cross-encoder model, all compatible with the current setup.
