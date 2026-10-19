"""RAG pipeline orchestrator.

Wires together indexing (parse -> clean -> chunk -> embed -> store)
and asking (question -> retrieve -> build context -> generate)
behind one service, so the UI never talks to the individual RAG
components directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config.settings import settings
from app.rag.agent import AgentError, run_agent
from app.rag.chart_reader import answer_from_charts, chart_pages, is_chart_question
from app.rag.chunker import chunk_pages
from app.rag.citations import cited_chunks, resolve_bracket_refs
from app.rag.embeddings import EmbeddingService
from app.rag.generator import AnswerGenerator
from app.rag.model_router import resolve_groq_model
from app.rag.page_store import PageStore
from app.rag.progress import Progress, report, scaled
from app.rag.query_decomposer import decompose_query, needs_decomposition
from app.rag.query_rewriter import needs_rewrite, rewrite_query, words
from app.rag.query_router import llm_route
from app.rag.reranker import Reranker
from app.rag.retriever import RetrievedChunk, Retriever
from app.rag.small_talk import is_small_talk
from app.rag.text_cleaner import clean_text
from app.rag.vector_store import VectorStore, VectorStoreError
from app.services.document_service import UploadedDocument, parse_uploaded_files

logger = logging.getLogger(__name__)

_EMBED_BATCH = 64


def run_inline(fn, *args, progress: Progress | None = None, **kwargs):
    """Run a reading step here. index_text/index_visuals take another
    runner with this signature to read in a worker process instead
    (app.services.ingest_jobs.ProcessRunner); `fn` reports to `progress`."""
    return fn(*args, progress=progress, **kwargs)

NOT_FOUND_MESSAGE = (
    "I couldn't find this information in the uploaded documents, "
    "so I don't want to guess."
)

# Where an answer's content came from — used by the UI to label each
# answer so it's clear whether it's grounded in the user's documents.
SOURCE_DOCUMENT = "document"
SOURCE_GENERAL_KNOWLEDGE = "general_knowledge"
SOURCE_SMALL_TALK = "small_talk"


@dataclass
class IndexResult:
    """One file's text indexing (RAGPipeline.index_text)."""

    added: int = 0
    error: str | None = None
    # {"pages", "passages", "charts"} (charts measured from the drawing).
    stats: dict[str, int] = field(default_factory=dict)
    # The document's text was already indexed exactly like this.
    unchanged: bool = False


@dataclass
class AskResult:
    """The outcome of asking a question: an answer plus its sources."""

    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    found_context: bool = False
    # One of the SOURCE_* constants above, or None when no real answer
    # was generated (empty question, strict-mode refusal, error).
    source: str | None = None
    # What retrieval actually searched for (the rewritten query, if any).
    search_query: str | None = None
    # Auto mode: every search the agent decided to run, in order.
    searches: list[str] = field(default_factory=list)
    # The sub-questions searched for a decomposed multi-part question
    # (empty when it was searched as one question).
    sub_queries: list[str] = field(default_factory=list)
    # The Groq model used (after resolving "auto"), for display.
    model: str | None = None
    # Pages whose chart images the answer was read from (app.rag.chart_reader);
    # empty for a normal text answer.
    chart_pages: list[str] = field(default_factory=list)


class RAGPipeline:
    """End-to-end RAG orchestrator used by the Streamlit UI.

    Every tuning knob defaults to the app's settings (`.env`); passing one
    explicitly overrides it for this pipeline only — used by the evaluation
    harness to compare configurations through exactly the code users hit.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        embedding_model: str | None = None,
        embedding_query_prefix: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        reranker_model: str | None = None,
        rerank_candidates: int | None = None,
        section_aware: bool | None = None,
        read_visuals: bool | None = None,
        hybrid_search: bool | None = None,
        chunk_strategy: str | None = None,
        chunk_unit: str | None = None,
        structure_source: str | None = None,
        layout_tables: str | None = None,
        query_routing: str | None = None,
        page_store_dir: str | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
        self.chunk_strategy = chunk_strategy or settings.chunk_strategy
        self.chunk_unit = chunk_unit or settings.chunk_unit
        self.structure_source = structure_source or settings.structure_source
        self.layout_tables = layout_tables or settings.layout_tables
        self.section_aware = settings.section_aware if section_aware is None else section_aware
        # Charts/tables/call-outs from page images + footnotes, for short PDFs (app/rag/visual.py).
        self.read_visuals = settings.visual_reading_max_pages > 0 if read_visuals is None else read_visuals
        self.embedding_service = EmbeddingService(embedding_model, embedding_query_prefix)
        self.vector_store = VectorStore(
            persist_dir=persist_dir, collection_name=collection_name
        )
        reranker_model = settings.reranker_model if reranker_model is None else reranker_model
        self.retriever = Retriever(
            self.vector_store, self.embedding_service, top_k=top_k, score_threshold=score_threshold,
            reranker=Reranker(reranker_model) if reranker_model else None,
            rerank_candidates=rerank_candidates, section_aware=self.section_aware, hybrid_search=hybrid_search,
        )
        self.generator = AnswerGenerator()
        if (query_routing or settings.query_routing) == "llm":
            self.retriever.router = lambda question, labels: llm_route(
                self.generator, question, labels, groq_model_name=settings.groq_fast_model)
        self.indexed_documents: set[str] = set()
        self.rewrite_queries = settings.query_rewriting
        self.decompose_queries = settings.query_decomposition
        self.chart_reading = settings.chart_reading
        self.page_store = PageStore(page_store_dir)
        # What the last index_files call read, per document (IndexResult.stats).
        self.last_indexed: dict[str, dict[str, int]] = {}
        # Words of the indexed documents, for skipping needless rewrites;
        # rebuilt when the chunk count changes (see _document_vocabulary).
        self._vocabulary: tuple[int, set[str]] = (-1, set())

    def _document_vocabulary(self) -> set[str]:
        """Every word in the indexed chunks, cached until the index changes."""
        count = self.vector_store.count()
        if self._vocabulary[0] != count:
            vocabulary = set()
            for chunk in self.vector_store.all_chunks():
                vocabulary.update(words(chunk["text"]))
            self._vocabulary = (count, vocabulary)
        return self._vocabulary[1]

    def index_files(self, files, progress: Progress | None = None) -> tuple[int, dict[str, str]]:
        """Parse, clean, chunk, embed, and store uploaded files: each file's
        text first (index_text), then what its page images show
        (index_visuals).

        Returns (num_chunks_added, errors) where `errors` maps
        filename to a friendly message for anything that failed.
        `progress` hears about each stage (app.rag.progress).
        """
        files = list(files)
        self.last_indexed = {}
        added, errors = 0, {}
        for number, file in enumerate(files):
            part = scaled(progress, number / len(files), (number + 1) / len(files))
            text = self.index_text(file, scaled(part, 0.0, 0.6))
            if text.error:
                errors[file.name] = text.error
                continue
            visuals = self.index_visuals(file, scaled(part, 0.6, 1.0), text)
            self.last_indexed[file.name] = {**text.stats, "passages": text.stats["passages"] + visuals.stats["passages"]}
            added += text.added + visuals.added
        report(progress, "Done", 1.0)
        return added, errors

    def index_text(self, file, progress: Progress | None = None, run=run_inline) -> IndexResult:
        """Index one uploaded file's text (plus charts measured from its
        drawing and tables rebuilt from its layout), so it can be searched
        while its page images are still being read (index_visuals).
        Reading takes the first half of `progress`, embedding most of the
        rest. Re-uploading a document replaces it; unchanged text is a
        cheap no-op. `run` runs the reading (see run_inline)."""
        report(progress, "Reading the document", 0.0)
        pages, errors = run(parse_uploaded_files, [UploadedDocument(file.name, file.getvalue())],
                            detect_structure=self.section_aware, read_visuals=self.read_visuals,
                            structure_source=self.structure_source, layout_tables=self.layout_tables,
                            progress=scaled(progress, 0.0, 0.5), defer_visuals=True)
        report(progress, "Splitting into passages", 0.5)
        chunks = self._chunk(pages)
        if not chunks:
            return IndexResult(error=errors.get(file.name) or "No extractable text was found in this document.")
        stats = {
            "pages": max((p.page for p in pages if p.page is not None), default=0),
            "passages": len(chunks),
            "charts": sum(1 for p in pages if p.text.startswith("[Chart on page")
                          and "measured from the drawing" in p.text),
        }
        try:
            # Unchanged content with the same settings yields the same chunk
            # IDs. Anything else (edited file, new chunking/section settings)
            # replaces the document, page-image chunks included, which would
            # otherwise sit next to the new ones.
            unchanged = self.vector_store.chunk_ids_for_source(file.name, visual=False) == {
                c.chunk_id for c in chunks}
            added = 0
            if not unchanged:
                self.vector_store.delete_ids(self.vector_store.chunk_ids_for_source(file.name))
                embeddings = self._embed_with_progress([c.text for c in chunks], scaled(progress, 0.52, 0.95))
                report(progress, "Saving", 0.96)
                added = self.vector_store.add_chunks(chunks, embeddings)
        except VectorStoreError as exc:
            logger.exception("Indexing %s failed", file.name)
            return IndexResult(error=str(exc))

        self.indexed_documents.add(file.name)
        self.page_store.save(file.name, file.getvalue())
        self._vocabulary = (-1, set())
        report(progress, "Text indexed", 1.0)
        return IndexResult(added=added, stats=stats, unchanged=unchanged)

    def index_visuals(self, file, progress: Progress | None = None, text: IndexResult | None = None,
                      run=run_inline) -> IndexResult:
        """Add what a PDF's page images show (charts, tables, call-outs read
        by a vision model, and footnotes; app.rag.visual) to its indexed
        text; `stats["passages"]` counts them. Skipped when the text was
        unchanged (`text`, from index_text) and its page images were read
        already. A failure is logged and leaves the text indexed. `run` runs
        the reading (see run_inline)."""
        stored = self.vector_store.chunk_ids_for_source(file.name, visual=True)
        if not self.visuals_pending(file, text):
            return IndexResult(stats={"passages": len(stored)}, unchanged=bool(stored))
        from app.rag.visual import read_visuals

        try:
            pages = run(read_visuals, file.getvalue(), file.name, progress=scaled(progress, 0.0, 0.85))
            for page in pages:
                page.text = clean_text(page.text)
            chunks = self._chunk([page for page in pages if page.text])
            for chunk in chunks:
                chunk.visual = True
            if stored == {c.chunk_id for c in chunks}:
                return IndexResult(stats={"passages": len(chunks)}, unchanged=True)
            self.vector_store.delete_ids(stored)
            embeddings = self._embed_with_progress([c.text for c in chunks], scaled(progress, 0.85, 0.98))
            added = self.vector_store.add_chunks(chunks, embeddings)
        except Exception:
            logger.exception("Adding the page images of %s failed; its text stays indexed", file.name)
            return IndexResult(error="Charts and tables could not be read; the text is searchable.",
                               stats={"passages": 0})
        self._vocabulary = (-1, set())
        report(progress, "Charts and tables added", 1.0)
        return IndexResult(added=added, stats={"passages": len(chunks)})

    def visuals_pending(self, file, text: IndexResult | None = None) -> bool:
        """Whether index_visuals will read this file's page images: a PDF,
        page-image reading on, and not an unchanged re-upload already read."""
        if not self.read_visuals or not file.name.lower().endswith(".pdf"):
            return False
        return not (text is not None and text.unchanged
                    and self.vector_store.chunk_ids_for_source(file.name, visual=True))

    def _chunk(self, pages: list) -> list:
        length = self.embedding_service.count_tokens if self.chunk_unit == "tokens" else len
        return chunk_pages(pages, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
                           strategy=self.chunk_strategy, length_function=length) if pages else []

    def _embed_with_progress(self, texts: list[str], progress: Progress | None) -> list[list[float]]:
        """Embed in batches, so a long document's progress keeps moving."""
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            report(progress, f"Embedding passages ({start:,} of {len(texts):,})", start / len(texts))
            embeddings += self.embedding_service.embed_texts(texts[start:start + _EMBED_BATCH])
        return embeddings

    def ask(
        self,
        question: str,
        strict_mode: bool | None = None,
        history: list[dict] | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        groq_model_name: str | None = None,
        auto_mode: bool | None = None,
    ) -> AskResult:
        """Answer a question using retrieved document context.

        `history` (prior chat turns, oldest first) is passed through to
        the LLM for conversational continuity — e.g. resolving "it"/
        "that" or remembering something said earlier in the session.
        It does not affect retrieval or the strict-mode "not found"
        check, which are always based on the current question alone.

        `model_name`/`groq_model_name`/`temperature` override the
        generator's configured defaults for this call only, so the UI
        can switch models or adjust temperature without rebuilding the
        pipeline.

        Greetings/pleasantries ("hi", "thanks", ...) always get a
        normal reply, bypassing the strict "not found" refusal — that
        guard is meant to stop fabricated *document* facts, not to
        block small talk. Small talk also skips retrieval, so it never
        shows unrelated document sources.

        Documents are always checked first. When strict mode is off
        (the default) and nothing relevant is found there, the model
        answers from its own general knowledge instead of refusing —
        `AskResult.source` tells the two apart so the UI can label
        which one produced the answer. Strict mode keeps the old
        behavior of refusing outright in that case.

        Auto mode (`auto_mode`, default AGENT_MODE) hands the question to
        the agent (app.rag.agent), which decides whether and how often to
        search. It needs Groq and indexed documents, is skipped when a
        Gemini model was picked explicitly, and falls back to the standard
        path below on any agent failure. A Groq model of "auto" is resolved
        per question to the fast or strong model (app.rag.model_router).
        """
        if not question or not question.strip():
            return AskResult(answer="Please enter a question.")

        strict = settings.strict_document_mode if strict_mode is None else strict_mode
        small_talk = is_small_talk(question)
        gemini_picked = model_name is not None and groq_model_name is None
        groq_model = resolve_groq_model(groq_model_name or self.generator.groq_model, question)

        # A question about a chart is answered by looking at the chart page
        # (app.rag.chart_reader), in auto mode too: no text-only model can
        # see a value that exists only as a bar height.
        if self.chart_reading and not small_talk and self.vector_store.count() > 0 and is_chart_question(question):
            charted = self._answer_from_charts(question, gemini_picked, model_name, groq_model)
            if charted is not None:
                return charted

        agent_on = settings.agent_mode if auto_mode is None else auto_mode
        if agent_on and not gemini_picked and self.generator.groq_api_key and self.vector_store.count() > 0:
            try:
                result = run_agent(
                    question, self.generator, self.retriever, history,
                    strict=strict, must_search=strict and not small_talk,
                    model_name=groq_model, temperature=temperature,
                    max_searches=settings.agent_max_searches,
                )
            except AgentError:
                logger.warning("Auto mode failed; falling back to the standard pipeline", exc_info=True)
            else:
                if strict and not result.sources and not small_talk:
                    return AskResult(answer=NOT_FOUND_MESSAGE, searches=result.searches, model=groq_model)
                # Show only what the answer actually cites: when the agent
                # searched but answered "not found" or from general knowledge,
                # its unrelated search results aren't sources.
                result.answer = resolve_bracket_refs(result.answer, result.sources)
                chunks = cited_chunks(result.answer, result.sources)
                if chunks:
                    source = SOURCE_DOCUMENT
                elif small_talk:
                    source = SOURCE_SMALL_TALK
                elif strict:
                    source = None  # strict mode: the agent declined; nothing to label
                else:
                    source = SOURCE_GENERAL_KNOWLEDGE
                return AskResult(
                    answer=result.answer, sources=chunks, found_context=bool(chunks), source=source,
                    search_query=result.searches[0] if result.searches else None,
                    searches=result.searches, model=groq_model,
                )

        # Fix typos and resolve follow-ups before searching (see
        # app.rag.query_rewriter). Skipped when there's nothing to search,
        # or nothing to fix (no unknown words, not a follow-up).
        search_query = question
        if (self.rewrite_queries and not small_talk and self.vector_store.count() > 0
                and needs_rewrite(question, history, self._document_vocabulary())):
            search_query = rewrite_query(
                self.generator, question, history,
                model_name=model_name, groq_model_name=groq_model,
            )
            small_talk = is_small_talk(search_query)

        # Split comparison/multi-part questions into sub-questions, each
        # searched on its own (app.rag.query_decomposer). Questions naming
        # sections are already searched per section by the retriever.
        sub_queries = []
        if (self.decompose_queries and not small_talk and self.vector_store.count() > 0
                and needs_decomposition(search_query) and not self.retriever.named_sections(search_query)):
            sub_queries = decompose_query(self.generator, search_query, model_name=model_name,
                                          groq_model_name=groq_model)
            sub_queries = sub_queries if len(sub_queries) > 1 else []

        # Small talk ("hi", "who are you?") never needs document
        # context — searching anyway attaches unrelated chunks as sources.
        try:
            if small_talk:
                chunks = []
            elif sub_queries:
                chunks = self.retriever.retrieve_many(sub_queries)
            else:
                chunks = self.retriever.retrieve(search_query)
        except Exception:
            logger.exception("Retrieval failed")
            return AskResult(answer="Something went wrong while searching your documents.")

        if strict and not chunks and not small_talk:
            return AskResult(answer=NOT_FOUND_MESSAGE, sources=[], search_query=search_query, sub_queries=sub_queries)

        answer = self.generator.generate(
            question,
            chunks,
            history=history,
            model_name=model_name,
            temperature=temperature,
            groq_model_name=groq_model,
        )

        answer = resolve_bracket_refs(answer, chunks)
        # As in auto mode: sources are what the answer cites — a "not found"
        # reply shouldn't be labelled as coming from the retrieved chunks.
        if chunks:
            chunks = cited_chunks(answer, chunks)
        if small_talk:
            source = SOURCE_SMALL_TALK
        elif chunks:
            source = SOURCE_DOCUMENT
        elif strict:
            source = None
        else:
            source = SOURCE_GENERAL_KNOWLEDGE

        return AskResult(
            answer=answer, sources=chunks, found_context=bool(chunks),
            source=source, search_query=search_query, sub_queries=sub_queries,
            model=None if gemini_picked else groq_model,
        )

    def _answer_from_charts(self, question: str, gemini_picked: bool, model_name: str | None,
                            groq_model: str | None) -> AskResult | None:
        """The answer read from the retrieved chart pages (their measurements
        and images), or None (no chart page among the results, or neither
        a measurement nor the vision model gave an answer)."""
        try:
            chunks = self.retriever.retrieve(question)
            pages = chart_pages(self.page_store, question, chunks)
        except Exception:
            logger.exception("Finding chart pages failed")
            return None
        if not pages:
            return None
        answer, saw_images = answer_from_charts(self.generator, self.page_store, question, chunks, pages,
                                                model_name=model_name if gemini_picked else None,
                                                groq_model_name=None if gemini_picked else groq_model)
        if not answer:
            return None
        answer = resolve_bracket_refs(answer, chunks)
        shown = {page for _, page in pages}
        sources = [c for c in chunks if c.page in shown] or cited_chunks(answer, chunks)
        return AskResult(
            answer=answer, sources=sources, found_context=True, source=SOURCE_DOCUMENT, search_query=question,
            model=model_name if gemini_picked else settings.chart_model if saw_images else groq_model,
            chart_pages=[f"{source}, page {page}" for source, page in pages],
        )

    def clear_index(self) -> None:
        """Delete all indexed content (used by the 'clear index' button)."""
        self.vector_store.clear()
        self.indexed_documents.clear()

    def document_count(self) -> int:
        return len(self.indexed_documents)
