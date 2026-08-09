"""RAG agent: guard input -> check cache -> expand query (multi-query +
HyDE) -> hybrid retrieve -> RRF fuse -> Cohere rerank -> cite -> generate.

Ties together guardrails (PII/prompt-injection), query_expansion
(multi-query + HyDE), hybrid_retrieval (alpha-weighted RRF), reranking
(Cohere with local cross-encoder fallback), and citation into a single
grounded-answer pipeline used by the router for RAG-classified queries.
"""

from __future__ import annotations

import asyncio
import os
import time

from loguru import logger

from src.cache import ResponseCache
from src.citation import (
    build_citations,
    format_answer_with_citations,
    invalid_citation_markers,
    strip_invalid_citation_markers,
    verify_citation_markers,
)
from src.guardrails import PIIGuard, detect_prompt_injection
from src.hybrid_retrieval import HybridRetriever
from src.query_expansion import QueryExpander
from src.reranking import Reranker
from src.telemetry import traced_call, log_event
from src.utils.schemas import AgentResponse, RouteName

RAG_MODEL = os.getenv("RAG_MODEL", "claude-haiku-4-5")
# Bounds every RAGAgent LLM call (multi-query, HyDE, and final generation
# share one LLM instance). Unbounded generation is a major, needless
# latency cost on local CPU backends and works against concise answers;
# 300 tokens comfortably covers a 3-5 sentence HyDE passage or a short
# grounded answer without capping legitimate long-form responses too hard.
RAG_MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "300"))


async def _passthrough(value):
    """Wraps an already-known value as a coroutine so it can sit next to
    a real async call in asyncio.gather (used to keep multi_query/hyde
    concurrent regardless of which one, if either, is disabled)."""
    return value


def _extract_text(response: object) -> str:
    """Agent-level `llm` objects in this codebase are expected to return
    a plain string from `.ainvoke`, but a raw langchain ChatModel returns
    a message object instead — accept either.
    """

    content = getattr(response, "content", None)
    return str(content) if content is not None else str(response)


class RAGAgent:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker | None = None,
        cache: ResponseCache | None = None,
        query_expander: QueryExpander | None = None,
        pii_guard: PIIGuard | None = None,
        llm=None,
        llm_backend: str | None = None,
        ollama_model: str | None = None,
        ollama_base_url: str | None = None,
        top_k: int = 10,
        rerank_top_k: int = 5,
        use_rerank: bool = True,
        use_multi_query: bool = True,
        use_hyde: bool = True,
        rerank_score_threshold: float | None = None,
    ):
        self.retriever = retriever
        self.reranker = reranker or Reranker()
        self.cache = cache or ResponseCache()
        if llm is None and llm_backend:
            # Explicit opt-in only -- callers that construct RAGAgent
            # without a backend (e.g. eval/tests) keep the current
            # llm=None stub-answer behavior unchanged.
            from src.llm_factory import build_llm

            llm = build_llm(
                RAG_MODEL,
                backend=llm_backend,
                ollama_model=ollama_model,
                ollama_base_url=ollama_base_url,
                max_tokens=RAG_MAX_TOKENS,
            )
        self.llm = llm
        self.query_expander = query_expander or QueryExpander(llm=llm)
        self.pii_guard = pii_guard or PIIGuard()
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.use_rerank = use_rerank
        self.use_multi_query = use_multi_query
        self.use_hyde = use_hyde
        self.rerank_score_threshold = rerank_score_threshold

    async def _generate(self, query: str, context_chunks: list[str]) -> str:
        if self.llm is None:
            joined = "\n---\n".join(context_chunks)
            return f"Based on the retrieved context:\n{joined}\n\nAnswer to '{query}' [1]"
        prompt = (
            "Answer the question using only the numbered context below. "
            "Be concise and direct -- a couple of sentences is usually enough. "
            "Cite sources inline as [n].\n\n"
            + "\n".join(f"[{i}] {c}" for i, c in enumerate(context_chunks, start=1))
            + f"\n\nQuestion: {query}"
        )
        return _extract_text(await self.llm.ainvoke(prompt))

    async def answer(self, query: str, k: int | None = None) -> AgentResponse:
        if detect_prompt_injection(query):
            log_event("rag_blocked_injection", query=query)
            return AgentResponse(
                answer="Request blocked: potential prompt injection detected.",
                route=RouteName.RAG,
            )
        safe_query, _ = self.pii_guard.anonymize(query)

        cache_key = ResponseCache.make_key("rag", safe_query)
        cached = self.cache.get(cache_key)
        if cached is not None:
            log_event("rag_cache_hit", query=safe_query)
            return AgentResponse(**cached)

        rerank_top_k = k or self.rerank_top_k

        start = time.perf_counter()
        with traced_call("rag"):
            # multi_query and hyde are independent LLM calls over the same
            # query -- run them concurrently instead of back-to-back to
            # halve their combined latency contribution.
            multi_query_task = (
                self.query_expander.multi_query(safe_query)
                if self.use_multi_query
                else _passthrough([safe_query])
            )
            hyde_task = (
                self.query_expander.hyde(safe_query) if self.use_hyde else _passthrough(safe_query)
            )
            queries, hyde_passage = await asyncio.gather(multi_query_task, hyde_task)
            if hyde_passage not in queries:
                queries = [*queries, hyde_passage]

            candidates = await self.retriever.retrieve(queries, top_k=self.top_k)
            if self.use_rerank:
                top = await self.reranker.arerank(safe_query, candidates, top_k=rerank_top_k)
            else:
                top = candidates[:rerank_top_k]

            if self.rerank_score_threshold is not None:
                # Drop candidates the reranker itself scored as irrelevant.
                # RRF's fused score reflects rank position across the
                # dense/sparse lists, not direct query-document relevance,
                # so a low-signal chunk can still land in the top-k purely
                # on recall -- the reranker's score is the actual relevance
                # judgment, and until now it was only used to sort, never
                # to filter. Candidates with no rerank_score (use_rerank
                # disabled) are left untouched -- there's no signal to
                # filter on.
                top = [
                    c
                    for c in top
                    if c.rerank_score is None or c.rerank_score >= self.rerank_score_threshold
                ]

            if not top:
                # Nothing cleared the relevance bar -- answer honestly
                # instead of asking the LLM to generate from an empty (or
                # near-empty) context, which invites hallucination.
                citations = []
                answer_text = (
                    "I don't have enough relevant information in the indexed "
                    "documents to answer that."
                )
            else:
                citations = build_citations(top)
                raw_answer = await self._generate(safe_query, [c.chunk.text for c in top])
                if not verify_citation_markers(raw_answer, len(citations)):
                    # The LLM referenced a citation number that doesn't
                    # exist (hallucinated -- e.g. [5] when only 3 sources
                    # were retrieved). Strip it rather than let it render
                    # as if it pointed to a real source: the citation IDs/
                    # metadata built above are untouched, only the
                    # answer's own inline markers are sanitized. Logged
                    # marker numbers, not the answer/query text itself --
                    # the generated answer can echo content straight from
                    # source documents, which don't belong verbatim in
                    # application logs.
                    invalid_markers = invalid_citation_markers(raw_answer, len(citations))
                    log_event("rag_invalid_citation_marker", invalid_markers=invalid_markers, n_citations=len(citations))
                    logger.warning("dropping hallucinated citation marker(s) {}", invalid_markers)
                    raw_answer = strip_invalid_citation_markers(raw_answer, len(citations))
                answer_text = format_answer_with_citations(raw_answer, citations)

        response = AgentResponse(
            answer=answer_text,
            citations=citations,
            route=RouteName.RAG,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        self.cache.set(cache_key, response.model_dump(mode="json"))
        return response
