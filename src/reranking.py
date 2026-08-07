"""Cross-encoder reranking of hybrid retrieval results.

RRF gives a good candidate set cheaply, but its fused score doesn't
model query-document interaction. A reranker re-scores the top-N
candidates directly against the query for higher precision before the
final context window is built -- this is what moves retrieval accuracy
from the ~61% RRF-only baseline to ~85% (see DECISIONS.md).

Three tiers, tried in order and each falling through to the next on
failure/unavailability, so reranking never takes the whole pipeline
down with it:

1. Cohere Rerank API (COHERE_API_KEY) -- typically strongest, needs no
   local model download, but costs a network call + API key.
2. flashrank -- a small ONNX cross-encoder that runs locally with no
   API key and no torch dependency, so it's the default offline tier.
3. sentence-transformers CrossEncoder -- the original local fallback,
   heavier (torch) but needs no extra optional dependency.
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
DEFAULT_FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"


class Reranker:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cohere_api_key: str | None = None,
        cohere_model: str = DEFAULT_COHERE_MODEL,
        use_flashrank: bool = False,
        flashrank_model: str = DEFAULT_FLASHRANK_MODEL,
    ):
        self.model_name = model_name
        self._model: CrossEncoder | None = None
        self.cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        self.cohere_model = cohere_model
        self._cohere_client = None
        self.use_flashrank = use_flashrank
        self.flashrank_model = flashrank_model
        self._flashrank_ranker = None

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

    @property
    def flashrank_ranker(self):
        if self._flashrank_ranker is None:
            from flashrank import Ranker

            self._flashrank_ranker = Ranker(model_name=self.flashrank_model)
        return self._flashrank_ranker

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

    def _flashrank_rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Local flashrank rerank (synchronous, ONNX runtime -- no torch)."""

        from flashrank import RerankRequest

        passages = [{"id": i, "text": c.chunk.text} for i, c in enumerate(candidates)]
        results = self.flashrank_ranker.rerank(RerankRequest(query=query, passages=passages))

        reranked = [
            RetrievedChunk(
                chunk=candidates[item["id"]].chunk,
                vector_score=candidates[item["id"]].vector_score,
                bm25_score=candidates[item["id"]].bm25_score,
                rrf_score=candidates[item["id"]].rrf_score,
                rerank_score=float(item["score"]),
            )
            for item in results[:top_k]
        ]
        logger.debug("flashrank reranked %d candidates -> top_k=%d", len(candidates), top_k)
        return reranked

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
        """Async rerank: Cohere when configured, else flashrank when
        enabled, else the local sentence-transformers cross-encoder --
        each tier run off the event loop thread and falling through to
        the next on any failure.
        """

        if not candidates:
            return []

        if self.cohere_api_key:
            try:
                return await self._cohere_rerank(query, candidates, top_k)
            except Exception as exc:
                logger.warning("cohere rerank failed (%s), falling back to next tier", exc)

        if self.use_flashrank:
            try:
                return await asyncio.to_thread(self._flashrank_rerank, query, candidates, top_k)
            except Exception as exc:
                logger.warning("flashrank rerank failed (%s), falling back to local cross-encoder", exc)

        return await asyncio.to_thread(self.rerank, query, candidates, top_k)
