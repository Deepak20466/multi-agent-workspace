"""Chroma-backed vector store wrapper.

Owns embedding + persistence; all retrieval strategies (vector, BM25,
hybrid/RRF) sit on top of this in hybrid_retrieval.py.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from src.utils.retry_handler import with_retry, RetryableError
from src.utils.schemas import Chunk, RetrievedChunk

logger = logging.getLogger("vectorstore")

DEFAULT_COLLECTION = "documents"
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")


class VectorStore:
    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        persist_dir: str = PERSIST_DIR,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @with_retry(exceptions=(RetryableError, ConnectionError, TimeoutError))
    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        try:
            self.collection.upsert(
                ids=[c.chunk_id for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[self._flatten_metadata(c) for c in chunks],
            )
        except Exception as exc:
            raise RetryableError(str(exc)) from exc
        logger.info("upserted %d chunks into '%s'", len(chunks), self.collection.name)
        return len(chunks)

    @with_retry(exceptions=(RetryableError, ConnectionError, TimeoutError))
    def similarity_search(self, query: str, k: int = 10, where: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        try:
            result = self.collection.query(query_texts=[query], n_results=k, where=where)
        except Exception as exc:
            raise RetryableError(str(exc)) from exc

        retrieved: list[RetrievedChunk] = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata or {}
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=metadata.get("doc_id", ""),
                text=text,
                chunk_index=int(metadata.get("chunk_index", 0)),
                page_number=metadata.get("page_number"),
                metadata=metadata,
            )
            # cosine distance -> similarity score in [0, 1]
            score = max(0.0, 1.0 - distance)
            retrieved.append(RetrievedChunk(chunk=chunk, vector_score=score))

        return retrieved

    def delete_by_doc_id(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self.collection.count()

    @staticmethod
    def _flatten_metadata(chunk: Chunk) -> dict[str, Any]:
        flat = {
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.chunk_index,
        }
        if chunk.page_number is not None:
            flat["page_number"] = chunk.page_number
        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                flat[key] = value
        return flat
