"""Hybrid retrieval: dense vector search + BM25 sparse search fused via
alpha-weighted Reciprocal Rank Fusion (RRF), run concurrently across
every query variant produced by query_expansion.QueryExpander
(RAG-Fusion).

RRF is used instead of a plain weighted score blend because vector
cosine similarity and BM25 scores live on incomparable scales; RRF
only needs each list's *rank*, which sidesteps that normalization
problem. `alpha` tunes how much weight dense vs. sparse results get
in the fused ranking (alpha=1.0 -> vector-only, alpha=0.0 -> BM25-only).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from src.utils.retry_handler import async_api_rate_limit_retry
from src.utils.schemas import Chunk, RetrievedChunk
from src.vectorstore import VectorStore

logger = logging.getLogger("hybrid_retrieval")

DEFAULT_RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


@dataclass
class BM25Index:
    """In-memory BM25 index over the corpus of chunks currently in the store."""

    chunks: list[Chunk] = field(default_factory=list)
    _bm25: BM25Okapi | None = field(default=None, init=False, repr=False)

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        if self._bm25 is None or not self.chunks:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True)[:k]
        return [RetrievedChunk(chunk=chunk, bm25_score=float(score)) for chunk, score in ranked if score > 0]


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    k: int = DEFAULT_RRF_K,
) -> list[RetrievedChunk]:
    """Fuse multiple ranked result lists into one, scored by unweighted RRF.

    score(chunk) = sum over lists of 1 / (k + rank_in_that_list)
    k=60 is the standard damping constant from the original RRF paper.
    """

    fused: dict[str, RetrievedChunk] = {}
    rrf_scores: dict[str, float] = {}

    for result_list in result_lists:
        for rank, retrieved in enumerate(result_list, start=1):
            chunk_id = retrieved.chunk.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

            if chunk_id not in fused:
                fused[chunk_id] = retrieved
            else:
                existing = fused[chunk_id]
                fused[chunk_id] = RetrievedChunk(
                    chunk=existing.chunk,
                    vector_score=existing.vector_score if existing.vector_score is not None else retrieved.vector_score,
                    bm25_score=existing.bm25_score if existing.bm25_score is not None else retrieved.bm25_score,
                )

    for chunk_id, retrieved in fused.items():
        fused[chunk_id] = RetrievedChunk(
            chunk=retrieved.chunk,
            vector_score=retrieved.vector_score,
            bm25_score=retrieved.bm25_score,
            rrf_score=rrf_scores[chunk_id],
        )

    return sorted(fused.values(), key=lambda r: r.rrf_score or 0.0, reverse=True)


class HybridRetriever:
    """Combines a VectorStore (dense) with an in-memory BM25Index (sparse)
    via alpha-weighted RRF, querying both concurrently for every query
    variant passed to `retrieve`.
    """

    def __init__(self, vectorstore: VectorStore, bm25: Optional[BM25Index] = None, alpha: float = 0.5):
        self.vectorstore = vectorstore
        self.bm25 = bm25 or BM25Index()
        self.alpha = alpha

    def index_corpus(self, chunks: list[Chunk]) -> None:
        """Refresh the BM25 side-index. Call after adding chunks to the vector store."""
        self.bm25.build(chunks)

    async def _vector_search(self, query: str, k: int) -> List[RetrievedChunk]:
        return await asyncio.to_thread(self.vectorstore.similarity_search, query, k)

    async def _bm25_search(self, query: str, k: int) -> List[RetrievedChunk]:
        return await asyncio.to_thread(self.bm25.search, query, k)

    def _weighted_rrf(
        self,
        vector_lists: List[List[RetrievedChunk]],
        bm25_lists: List[List[RetrievedChunk]],
        k: int = DEFAULT_RRF_K,
    ) -> List[RetrievedChunk]:
        fused: Dict[str, RetrievedChunk] = {}
        scores: Dict[str, float] = {}

        def _accumulate(result_lists: List[List[RetrievedChunk]], weight: float) -> None:
            if weight <= 0:
                return
            for result_list in result_lists:
                for rank, retrieved in enumerate(result_list, start=1):
                    chunk_id = retrieved.chunk.chunk_id
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
                    if chunk_id not in fused:
                        fused[chunk_id] = retrieved
                    else:
                        existing = fused[chunk_id]
                        fused[chunk_id] = RetrievedChunk(
                            chunk=existing.chunk,
                            vector_score=existing.vector_score
                            if existing.vector_score is not None
                            else retrieved.vector_score,
                            bm25_score=existing.bm25_score if existing.bm25_score is not None else retrieved.bm25_score,
                        )

        _accumulate(vector_lists, self.alpha)
        _accumulate(bm25_lists, 1.0 - self.alpha)

        for chunk_id, retrieved in fused.items():
            fused[chunk_id] = RetrievedChunk(
                chunk=retrieved.chunk,
                vector_score=retrieved.vector_score,
                bm25_score=retrieved.bm25_score,
                rrf_score=scores[chunk_id],
            )

        return sorted(fused.values(), key=lambda r: r.rrf_score or 0.0, reverse=True)

    @async_api_rate_limit_retry
    async def retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
        vector_k: int = 20,
        bm25_k: int = 20,
    ) -> List[RetrievedChunk]:
        """Run dense + sparse retrieval for every query variant
        concurrently, then fuse all result lists via alpha-weighted RRF.

        `queries` is typically [original_query, *expansions] from
        QueryExpander.multi_query, so a single ambiguous phrasing
        doesn't limit recall (RAG-Fusion).
        """

        tasks = []
        for query in queries:
            tasks.append(self._vector_search(query, vector_k))
            tasks.append(self._bm25_search(query, bm25_k))

        results = await asyncio.gather(*tasks)
        vector_lists = results[0::2]
        bm25_lists = results[1::2]

        if not self.bm25.is_built:
            logger.debug("bm25 index not built yet, falling back to vector-only results")
            fused = reciprocal_rank_fusion(vector_lists)
        else:
            fused = self._weighted_rrf(vector_lists, bm25_lists)

        return fused[:top_k]
