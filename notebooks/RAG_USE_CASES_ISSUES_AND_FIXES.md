# RAG Hands-On Use Cases — Issues Found and How We Fixed Them

The three use cases go from easiest to hardest. Each section below has the
same parts: the **problem** we saw (with the numbers from the evaluation
reports), **why** it happened, **what we changed**, and the **result**.

Every number comes from a report in `evaluation/reports/`. An AI judge grades
each answer against a reference answer written from the PDF. "Correct" means
the judge found every key fact. "Cites right page" means the answer cites a
page where the fact is.

| Use case | Document | Before | After |
|---|---|---|---|
| 1. Plain text accuracy | GPO Annual Report, 82 pages | Retrieval hit rate 92% (no reranker) | 100% hit rate, 13/13 correct |
| 2. Length and structure | World Bank GEP, 200 pages | 80% correct, 71% right page cited | 93% correct, 100% right page cited |
| 3. Visual / scattered facts | Apple iMac Environmental Report, 12 pages | 86% correct | 100% correct, 100% right page cited |

---

## Use case 1 — GPO Annual Report (plain text baseline)

**Pass bar:** the exact fact from the right paragraph, with nothing invented.

### Issue: the right paragraph sometimes wasn't retrieved
- **Problem:** with plain similarity search, 2 of the 25 generated test
  questions (`gen-15`, `gen-18`) did not get their answering page in the top 5.
  The hit rate was 92%, with a ranking score (MRR) of 0.84, where 1.0 means
  the right page always comes first.
- **Why:** the embedding model (MiniLM) ranks by overall topic similarity. A
  paragraph on the same topic but without the specific fact can outrank the
  one that has it.
- **Fix:** a cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`). It fetches 20
  candidates and re-scores each one against the question itself, then keeps
  the best 5.
- **Result:** on the same 25 questions, a 100% hit rate and an MRR of 1.0
  (the answering page ranked first every time). On the final question set,
  13/13 answers are correct.

### Remaining small issue
- One answer (the NARA renovation question) cited with bracket markers like
  `【1】` instead of "(page N)". The fact is right, but the page check can't
  read that citation, so the cited-page score is 92%. The prompt already
  forbids those markers; `gpt-oss-120b` sometimes ignores that.

---

## Use case 2 — World Bank GEP (long document with chapters and sections)

**Pass bar:** the right chapter and section is found, and no facts are mixed
in from another region or chapter.

### Issue 1: answers came from the wrong section
- **Problem:** for *"According to the executive summary, what are the main
  risks?"*, retrieval returned pages 76, 54, 74, 80 and 52, none of them the
  Executive Summary (page 19). For *"What is the growth outlook for South
  Asia?"*, the numbers came from the statistical tables and the wrong
  figures were given (6.6% instead of 6.3%).
- **Why:** a 200-page report repeats the same vocabulary everywhere. "Growth
  is projected to…" appears in every regional section, so similarity search
  can't tell South Asia's paragraph from Sub-Saharan Africa's. The search
  also had no idea which section a page belongs to.
- **Fix (section-aware retrieval, `app/rag/structure.py`):**
  1. **Find the sections.** Each page gets its chapter/section from the PDF
     bookmarks, the running headers ("CHAPTER 3 … | JUNE 2026"), and
     uppercase "BOX n.n" labels. Some bookmarks point to the wrong page (for
     example "South Asia" → p. 95, which is a table), so each title is
     re-located to the page where it really appears as a heading.
  2. **Label every chunk** with its section, e.g.
     `[Ch. 2 Regional Outlooks › South Asia]`. Both the search and the LLM
     can now see where a passage comes from.
  3. **Search only the named section.** "According to the executive
     summary…", "Chapter 3", "Box 1.1" and "South Asia" restrict the search
     to that section. Front matter (contents, abbreviations) is left out of
     general searches.
  4. **The prompt tells the model** to use only the sections the question
     asks about and to never mix two regions' facts.
- **Result:** retrieval found the right section for every question (hit rate
  93% → 100%).

### Issue 2: comparison questions only got evidence from one side
- **Problem:** *"Compare South Asia and Sub-Saharan Africa"* returned country
  figures and missed both regional totals (6.3% and 4%).
- **Why:** one shared top 5 filled up with chunks from whichever region
  matched better.
- **Fix:** when a question names several sections, each one gets its own
  share of the search results.
- **Result:** the comparison answer is now correct.

### Issue 3: the headline number was in the section but not retrieved
- **Problem:** even inside the right section, answers left out key facts:
  India's 6.6%, Chapter 3's debt figures (under 40% → over 70% of GDP; +110
  and +30 basis points), and Europe and Central Asia's 2.1% (answered 2.8%).
  Accuracy stayed at 80%.
- **Why:** a section's headline facts sit on its opening page, but often in
  a chunk that scored a bit lower than its neighbours, so it missed the top 5.
- **Fix (small-to-big retrieval):** for a question about one named section,
  the whole page(s) of the best hits go to the LLM, capped at 2 pages. The
  prompt also asks the model to be complete: every point that answers the
  question, with the qualifiers on a figure.
- **Result:** 93% correct (14/15), 100% right page cited.

### Issue 4: the risks answer leaves out points (not fixed yet)
- **Problem:** the executive-summary risks answer listed the downside risks
  but left out *"On the upside, broader investment in and adoption of AI
  could lift activity."*
- **Why:** that sentence *was* in the context; the model read "risks" as
  only the bad things.
- **Tried:** for questions about risks, the prompt now asks for a separate
  **"Upside risks:"** line whenever the context has one.
- **Result:** that did not fix it. Asked again with the new prompt, the
  answer still had no upside line, and this time it also left out the
  severe scenario ("global growth could fall to just 1.3 percent in 2026").
  Both points sit in the same retrieved passage. So the problem is the model
  skipping parts of its context, which changes from run to run even at
  temperature 0, not a missing instruction.
- **Status:** the pass bar is still met (right section, nothing mixed in);
  only the strict completeness check fails.

---

## Use case 3 — Apple iMac Environmental Report (charts, tables, footnotes)

**Pass bar:** the right number is pulled from a chart or table, not just a
general sentence nearby.

### Issue 1: facts inside charts were invisible
- **Problem:** *"What is the total carbon footprint, and how much comes from
  use vs manufacturing?"* gave the total but said the split wasn't in the
  document. It is: Production 50% · Transportation 5% · Product use 45%
  (pages 2 and 10).
- **Why:** the charts are drawn as vector shapes. Text extraction gets
  jumbled labels or nothing, and there is no image to OCR.
- **Fix (`app/rag/visual.py`):** each page of a short PDF (up to 30 pages)
  is rendered as an image. A vision model transcribes its charts, tables and
  call-outs into text: tables as Markdown, charts with their printed values.
  Each element becomes its own searchable passage, so a table is never split
  from its title. Replies are cached, so each page is read only once.
- **Result:** the carbon-footprint split is now answered correctly, along
  with every other chart and table question.

### Issue 2: footnotes were cut off from the claims they qualify
- **Problem:** *"What exactly does the 'more than 15 percent recycled
  content' claim measure, according to its footnote?"* was answered "not
  available".
- **Why:** the claim (pages 1 and 4) only carries a marker ("15%¹"). The
  definition is on page 11 under "Endnotes", which never mentions "15
  percent", so the search never connected the two.
- **Fix:** superscript markers on each page are matched to the document's
  endnotes. Each referenced note becomes its own passage on that page:
  `[Footnote 1, referenced on page 4] Product recycled or renewable content
  is the mass of certified recycled material relative to the overall mass of
  the device…`
- **Result:** 14/14 correct, 100% right page cited.

### Follow-up: footnotes no longer depend on their search rank
- **Problem:** the footnote passage was only the 5th of the 5 retrieved for
  the question, so a small retrieval change could push it out again.
- **Fix:** when a retrieved passage carries a footnote marker such as `[1]`,
  the retriever adds that note's passage from the same page (once per
  document, at most 3 notes per question), wherever the note itself ranked.
- **Result:** the definition always reaches the model when the claim does.

---

## Issues in the evaluation itself

Measuring the use cases turned up problems in the measuring tool too. These
matter because a wrong score hides a real problem, or invents one.

| Problem | Why | Fix |
|---|---|---|
| A GEP run scored 0% because 13 of 15 grades were "Unparseable judge output: ''" | The judge model was rate-limited and returned nothing; an empty grade counted as a wrong answer | Empty or unreadable judge replies are retried; failed answers are asked again after the per-minute limit resets |
| A report was out of date and showed a fixed question as failing | The report was written a minute before the footnote fix was saved | Re-run the evaluation after every change to retrieval or prompts |
| Scores labelled `gpt-oss-120b` could partly come from Gemini | When Groq was rate-limited, the app quietly switched to Gemini, and the report kept the Groq label | Evaluation runs now stay on the pinned model, and a run stops with a clear error instead of writing a misleading report |

---

## What a learner takes away

1. **Plain text:** a reranker is the cheapest big win. Similarity search
   finds the topic; the reranker finds the fact.
2. **Long, structured documents:** retrieval has to know *where* a passage
   sits. Label chunks with their section, search the section the question
   names, and send whole pages for headline facts.
3. **Visual documents:** text extraction misses charts and cuts footnotes off
   from their claims. Read the page image, and attach each footnote to the
   page that references it.
4. **Always:** trust the evaluation only after checking it. A rate limit, a
   stale report or a silent model switch can each produce the wrong score.
