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
from app.rag.chunker import chunk_pages
from app.rag.embeddings import EmbeddingService
from app.rag.generator import AnswerGenerator
from app.rag.retriever import Retriever, RetrievedChunk
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


class RAGPipeline:
    """End-to-end RAG orchestrator used by the Streamlit UI."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore(
            persist_dir=persist_dir, collection_name=collection_name
        )
        self.retriever = Retriever(self.vector_store, self.embedding_service)
        self.generator = AnswerGenerator()
        self.indexed_documents: set[str] = set()

    def index_files(self, files) -> tuple[int, dict[str, str]]:
        """Parse, clean, chunk, embed, and store uploaded files.

        Returns (num_chunks_added, errors) where `errors` maps
        filename to a friendly message for anything that failed.
        """
        pages, errors = parse_uploaded_files(files)
        if not pages:
            return 0, errors

        chunks = chunk_pages(pages)
        if not chunks:
            return 0, errors

        try:
            embeddings = self.embedding_service.embed_texts([c.text for c in chunks])
            num_added = self.vector_store.add_chunks(chunks, embeddings)
        except VectorStoreError as exc:
            logger.exception("Indexing failed")
            errors["_index"] = str(exc)
            return 0, errors

        self.indexed_documents.update(page.source for page in pages)
        return num_added, errors

    def ask(
        self,
        question: str,
        strict_mode: bool | None = None,
        history: list[dict] | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        openrouter_model_name: str | None = None,
    ) -> AskResult:
        """Answer a question using retrieved document context.

        `history` (prior chat turns, oldest first) is passed through to
        the LLM for conversational continuity — e.g. resolving "it"/
        "that" or remembering something said earlier in the session.
        It does not affect retrieval or the strict-mode "not found"
        check, which are always based on the current question alone.

        `model_name`/`openrouter_model_name`/`temperature` override the
        generator's configured defaults for this call only, so the UI
        can switch models or adjust temperature without rebuilding the
        pipeline.

        Greetings/pleasantries ("hi", "thanks", ...) always get a
        normal reply, bypassing the strict "not found" refusal — that
        guard is meant to stop fabricated *document* facts, not to
        block small talk.

        Documents are always checked first. When strict mode is off
        (the default) and nothing relevant is found there, the model
        answers from its own general knowledge instead of refusing —
        `AskResult.source` tells the two apart so the UI can label
        which one produced the answer. Strict mode keeps the old
        behavior of refusing outright in that case.
        """
        if not question or not question.strip():
            return AskResult(answer="Please enter a question.")

        strict = settings.strict_document_mode if strict_mode is None else strict_mode
        small_talk = is_small_talk(question)

        try:
            chunks = self.retriever.retrieve(question)
        except Exception:
            logger.exception("Retrieval failed")
            return AskResult(answer="Something went wrong while searching your documents.")

        if strict and not chunks and not small_talk:
            return AskResult(answer=NOT_FOUND_MESSAGE, sources=[])

        answer = self.generator.generate(
            question,
            chunks,
            history=history,
            model_name=model_name,
            temperature=temperature,
            openrouter_model_name=openrouter_model_name,
        )

        if small_talk:
            source = SOURCE_SMALL_TALK
        elif chunks:
            source = SOURCE_DOCUMENT
        else:
            source = SOURCE_GENERAL_KNOWLEDGE

        return AskResult(answer=answer, sources=chunks, found_context=bool(chunks), source=source)

    def clear_index(self) -> None:
        """Delete all indexed content (used by the 'clear index' button)."""
        self.vector_store.clear()
        self.indexed_documents.clear()

    def document_count(self) -> int:
        return len(self.indexed_documents)
