"""Cross-encoder reranking of hybrid retrieval results.

RRF gives a good candidate set cheaply, but its fused score doesn't
model query-document interaction. A reranker re-scores the top-N
candidates directly against the query for higher precision before the
final context window is built.

Prefers the Cohere Rerank API (COHERE_API_KEY) when configured, since
it's typically stronger than a local cross-encoder and needs no model
download; falls back to a local `sentence-transformers` CrossEncoder
otherwise, or if the Cohere call itself fails, so reranking never takes
the whole pipeline down with it.
"""

from __future__ import annotations

import asyncio
import logging
import os

from sentence_transformers import CrossEncoder

from src.utils.schemas import RetrievedChunk

logger = logging.getLogger("reranking")

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_COHERE_MODEL = "rerank-english-v3.0"


class Reranker:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cohere_api_key: str | None = None,
        cohere_model: str = DEFAULT_COHERE_MODEL,
    ):
        self.model_name = model_name
        self._model: CrossEncoder | None = None
        self.cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        self.cohere_model = cohere_model
        self._cohere_client = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    @property
    def cohere_client(self):
        if self._cohere_client is None:
            import cohere

            self._cohere_client = cohere.AsyncClient(self.cohere_api_key)
        return self._cohere_client

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        """Local cross-encoder rerank (synchronous)."""

        if not candidates:
            return []

        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.model.predict(pairs)

        reranked = [
            RetrievedChunk(
                chunk=candidate.chunk,
                vector_score=candidate.vector_score,
                bm25_score=candidate.bm25_score,
                rrf_score=candidate.rrf_score,
                rerank_score=float(score),
            )
            for candidate, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)
        logger.debug("reranked %d candidates -> top_k=%d", len(candidates), top_k)
        return reranked[:top_k]

    async def _cohere_rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        documents = [c.chunk.text for c in candidates]
        result = await self.cohere_client.rerank(
            model=self.cohere_model, query=query, documents=documents, top_n=min(top_k, len(documents))
        )
        reranked = []
        for item in result.results:
            candidate = candidates[item.index]
            reranked.append(
                RetrievedChunk(
                    chunk=candidate.chunk,
                    vector_score=candidate.vector_score,
                    bm25_score=candidate.bm25_score,
                    rrf_score=candidate.rrf_score,
                    rerank_score=float(item.relevance_score),
                )
            )
        return reranked

    async def arerank(self, query: str, candidates: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        """Async rerank: Cohere when configured, else the local
        cross-encoder run off the event loop thread.
        """

        if not candidates:
            return []

        if self.cohere_api_key:
            try:
                return await self._cohere_rerank(query, candidates, top_k)
            except Exception as exc:
                logger.warning("cohere rerank failed (%s), falling back to local cross-encoder", exc)

        return await asyncio.to_thread(self.rerank, query, candidates, top_k)
