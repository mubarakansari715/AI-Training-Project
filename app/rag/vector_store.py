"""Persistent ChromaDB vector store.

Stores chunk embeddings with their metadata (source, page, chunk_id)
in a local, persistent collection, so uploaded documents don't need
to be re-processed every time the app restarts.
"""

from __future__ import annotations

import logging

import chromadb
import numpy as np

from app.config.settings import settings
from app.rag.chunker import Chunk

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised when the vector store cannot complete an operation."""


class VectorStore:
    """Thin wrapper around a persistent ChromaDB collection."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name

        try:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.exception("Failed to initialize ChromaDB")
            raise VectorStoreError("Could not initialize the document index.") from exc

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Add chunks and their embeddings to the collection.

        Chunks whose `chunk_id` already exists are skipped, so
        re-uploading the same document is a cheap no-op rather than a
        duplicate entry. Returns the number of chunks actually added.
        """
        if not chunks:
            return 0

        existing_ids = set(
            self._collection.get(ids=[c.chunk_id for c in chunks])["ids"]
        )
        new_items = [
            (chunk, embedding)
            for chunk, embedding in zip(chunks, embeddings)
            if chunk.chunk_id not in existing_ids
        ]
        if not new_items:
            return 0

        try:
            self._collection.add(
                ids=[c.chunk_id for c, _ in new_items],
                embeddings=[e for _, e in new_items],
                documents=[c.text for c, _ in new_items],
                metadatas=[
                    {
                        "source": c.source,
                        # Chroma metadata can't store None, so use a sentinel.
                        "page": c.page if c.page is not None else -1,
                        "section": c.section or "",
                        "pages": ",".join(map(str, c.pages)),
                    }
                    for c, _ in new_items
                ],
            )
        except Exception as exc:
            logger.exception("Failed to add chunks to ChromaDB")
            raise VectorStoreError("Could not save documents to the index.") from exc

        return len(new_items)

    def search(
        self, query_embedding: list[float], top_k: int | None = None, where: dict | None = None
    ) -> list[dict]:
        """Return the top_k most similar chunks to `query_embedding`,
        optionally restricted by a ChromaDB metadata filter (`where`)."""
        top_k = top_k or settings.top_k
        if self._collection.count() == 0:
            return []

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._collection.count()),
                where=where,
            )
        except Exception as exc:
            logger.exception("Failed to query ChromaDB")
            raise VectorStoreError("Could not search the document index.") from exc

        matches = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            page = metadata.get("page")
            matches.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "source": metadata.get("source"),
                    "page": None if page == -1 else page,
                    "section": metadata.get("section") or None,
                    "pages": _pages(metadata),
                    "distance": results["distances"][0][i],
                }
            )
        return matches

    def clear(self) -> None:
        """Delete and recreate the collection (used by 'clear index')."""
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.exception("Failed to clear ChromaDB collection")
            raise VectorStoreError("Could not clear the document index.") from exc

    def count(self) -> int:
        return self._collection.count()

    def all_chunks(self) -> list[dict]:
        """Every stored chunk as {chunk_id, text, source, page} — for
        whole-corpus work like keyword (BM25) scoring or building test sets."""
        stored = self._collection.get(include=["documents", "metadatas"])
        return [
            {
                "chunk_id": chunk_id,
                "text": text,
                "source": meta.get("source"),
                "page": None if meta.get("page") == -1 else meta.get("page"),
                "section": meta.get("section") or None,
                "pages": _pages(meta),
            }
            for chunk_id, text, meta in zip(stored["ids"], stored["documents"], stored["metadatas"])
        ]

    def similarities(self, query_embedding: list[float], ids: list[str]) -> dict[str, float]:
        """Cosine similarity of `query_embedding` to each stored chunk in `ids`
        (the same score `search` gives), for chunks found another way."""
        if not ids:
            return {}
        stored = self._collection.get(ids=list(ids), include=["embeddings"])
        query = np.asarray(query_embedding, dtype=float)
        query_norm = np.linalg.norm(query) or 1.0
        result = {}
        for chunk_id, vector in zip(stored["ids"], stored["embeddings"]):
            vector = np.asarray(vector, dtype=float)
            result[chunk_id] = float(vector @ query / ((np.linalg.norm(vector) or 1.0) * query_norm))
        return result

    def chunk_ids_for_source(self, source: str) -> set[str]:
        """IDs of every stored chunk from one document."""
        return set(self._collection.get(where={"source": source}, include=[])["ids"])

    def delete_ids(self, ids: set[str]) -> None:
        if not ids:
            return
        try:
            self._collection.delete(ids=sorted(ids))
        except Exception as exc:
            logger.exception("Failed to delete chunks from ChromaDB")
            raise VectorStoreError("Could not update the document index.") from exc

    def sections(self) -> list[str]:
        """Distinct section labels in the index (empty for unstructured documents)."""
        stored = self._collection.get(include=["metadatas"])
        return sorted({meta.get("section") for meta in stored["metadatas"] if meta.get("section")})

    def page_chunks(self, source: str, page: int) -> list[dict]:
        """Every stored chunk of one page, in reading order (the chunk index
        in the ID, see app.rag.chunker._make_chunk_id)."""
        stored = self._collection.get(
            where={"$and": [{"source": source}, {"page": page}]}, include=["documents", "metadatas"]
        )
        chunks = [
            {"chunk_id": chunk_id, "text": text, "source": source, "page": page,
             "section": meta.get("section") or None}
            for chunk_id, text, meta in zip(stored["ids"], stored["documents"], stored["metadatas"])
        ]
        return sorted(chunks, key=lambda c: _chunk_index(c["chunk_id"]))


def _chunk_index(chunk_id: str) -> int:
    parts = chunk_id.split(":")
    return int(parts[-2]) if len(parts) >= 4 and parts[-2].isdigit() else 0


def _pages(metadata: dict) -> list[int]:
    """The pages a chunk covers; empty for chunks indexed before pages were stored."""
    return [int(page) for page in (metadata.get("pages") or "").split(",") if page]
