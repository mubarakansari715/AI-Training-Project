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
from app.rag.chunker import chunk_pages
from app.rag.citations import cited_chunks
from app.rag.embeddings import EmbeddingService
from app.rag.generator import AnswerGenerator
from app.rag.model_router import resolve_groq_model
from app.rag.query_rewriter import rewrite_query
from app.rag.reranker import Reranker
from app.rag.retriever import RetrievedChunk, Retriever
from app.rag.small_talk import is_small_talk
from app.rag.vector_store import VectorStore, VectorStoreError
from app.services.document_service import parse_uploaded_files

logger = logging.getLogger(__name__)

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
    # The Groq model used (after resolving "auto"), for display.
    model: str | None = None


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
        chunk_strategy: str | None = None,
        chunk_unit: str | None = None,
        structure_source: str | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
        self.chunk_strategy = chunk_strategy or settings.chunk_strategy
        self.chunk_unit = chunk_unit or settings.chunk_unit
        self.structure_source = structure_source or settings.structure_source
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
            rerank_candidates=rerank_candidates, section_aware=self.section_aware,
        )
        self.generator = AnswerGenerator()
        self.indexed_documents: set[str] = set()
        self.rewrite_queries = settings.query_rewriting

    def index_files(self, files) -> tuple[int, dict[str, str]]:
        """Parse, clean, chunk, embed, and store uploaded files.

        Returns (num_chunks_added, errors) where `errors` maps
        filename to a friendly message for anything that failed.
        """
        pages, errors = parse_uploaded_files(files, detect_structure=self.section_aware, read_visuals=self.read_visuals,
                                             structure_source=self.structure_source)
        if not pages:
            return 0, errors

        length = self.embedding_service.count_tokens if self.chunk_unit == "tokens" else len
        chunks = chunk_pages(pages, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
                             strategy=self.chunk_strategy, length_function=length)
        if not chunks:
            return 0, errors

        try:
            # Re-uploading a document replaces it. Unchanged content with the
            # same settings yields the same chunk IDs (a cheap no-op below);
            # anything else — edited file, new chunking/section settings —
            # would otherwise leave the old chunks next to the new ones.
            chunks = self._replace_changed_sources(chunks)
            embeddings = self.embedding_service.embed_texts([c.text for c in chunks])
            num_added = self.vector_store.add_chunks(chunks, embeddings)
        except VectorStoreError as exc:
            logger.exception("Indexing failed")
            errors["_index"] = str(exc)
            return 0, errors

        self.indexed_documents.update(page.source for page in pages)
        return num_added, errors

    def _replace_changed_sources(self, chunks: list) -> list:
        """Drop chunks already stored unchanged; delete a document's old
        chunks when its new chunk set differs. Returns the chunks to add."""
        by_source: dict[str, set[str]] = {}
        for chunk in chunks:
            by_source.setdefault(chunk.source, set()).add(chunk.chunk_id)
        unchanged: set[str] = set()
        for source, new_ids in by_source.items():
            old_ids = self.vector_store.chunk_ids_for_source(source)
            if old_ids == new_ids:
                unchanged.add(source)
            elif old_ids:
                logger.info("Re-indexing %s: replacing %d old chunks", source, len(old_ids))
                self.vector_store.delete_ids(old_ids)
        return [c for c in chunks if c.source not in unchanged]

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
        # app.rag.query_rewriter). Skipped when there's nothing to search.
        search_query = question
        if self.rewrite_queries and not small_talk and self.vector_store.count() > 0:
            search_query = rewrite_query(
                self.generator, question, history,
                model_name=model_name, groq_model_name=groq_model,
            )
            small_talk = is_small_talk(search_query)

        # Small talk ("hi", "who are you?") never needs document
        # context — searching anyway attaches unrelated chunks as sources.
        try:
            chunks = [] if small_talk else self.retriever.retrieve(search_query)
        except Exception:
            logger.exception("Retrieval failed")
            return AskResult(answer="Something went wrong while searching your documents.")

        if strict and not chunks and not small_talk:
            return AskResult(answer=NOT_FOUND_MESSAGE, sources=[], search_query=search_query)

        answer = self.generator.generate(
            question,
            chunks,
            history=history,
            model_name=model_name,
            temperature=temperature,
            groq_model_name=groq_model,
        )

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
            source=source, search_query=search_query,
            model=None if gemini_picked else groq_model,
        )

    def clear_index(self) -> None:
        """Delete all indexed content (used by the 'clear index' button)."""
        self.vector_store.clear()
        self.indexed_documents.clear()

    def document_count(self) -> int:
        return len(self.indexed_documents)
