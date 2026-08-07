"""Cross-encoder reranking of hybrid retrieval results.

RRF gives a good candidate set cheaply, but its fused score doesn't
model query-document interaction. A cross-encoder re-scores the
top-N candidates directly against the query for higher precision
before the final context window is built.
"""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from src.utils.schemas import RetrievedChunk

logger = logging.getLogger("reranking")

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
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
