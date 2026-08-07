"""RAG agent: guard input -> check cache -> expand query (multi-query +
HyDE) -> hybrid retrieve -> RRF fuse -> Cohere rerank -> cite -> generate.

Ties together guardrails (PII/prompt-injection), query_expansion
(multi-query + HyDE), hybrid_retrieval (alpha-weighted RRF), reranking
(Cohere with local cross-encoder fallback), and citation into a single
grounded-answer pipeline used by the router for RAG-classified queries.
"""

from __future__ import annotations

import time

from src.cache import ResponseCache
from src.citation import build_citations, format_answer_with_citations
from src.guardrails import PIIGuard, detect_prompt_injection
from src.hybrid_retrieval import HybridRetriever
from src.query_expansion import QueryExpander
from src.reranking import Reranker
from src.telemetry import traced_call, log_event
from src.utils.schemas import AgentResponse, RouteName


class RAGAgent:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker | None = None,
        cache: ResponseCache | None = None,
        query_expander: QueryExpander | None = None,
        pii_guard: PIIGuard | None = None,
        llm=None,
        top_k: int = 10,
        rerank_top_k: int = 5,
        use_rerank: bool = True,
    ):
        self.retriever = retriever
        self.reranker = reranker or Reranker()
        self.cache = cache or ResponseCache()
        self.llm = llm
        self.query_expander = query_expander or QueryExpander(llm=llm)
        self.pii_guard = pii_guard or PIIGuard()
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.use_rerank = use_rerank

    async def _generate(self, query: str, context_chunks: list[str]) -> str:
        if self.llm is None:
            joined = "\n---\n".join(context_chunks)
            return f"Based on the retrieved context:\n{joined}\n\nAnswer to '{query}' [1]"
        prompt = (
            "Answer the question using only the numbered context below. "
            "Cite sources inline as [n].\n\n"
            + "\n".join(f"[{i}] {c}" for i, c in enumerate(context_chunks, start=1))
            + f"\n\nQuestion: {query}"
        )
        return await self.llm.ainvoke(prompt)

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
            queries = await self.query_expander.multi_query(safe_query)
            hyde_passage = await self.query_expander.hyde(safe_query)
            if hyde_passage not in queries:
                queries = [*queries, hyde_passage]

            candidates = await self.retriever.retrieve(queries, top_k=self.top_k)
            if self.use_rerank:
                top = await self.reranker.arerank(safe_query, candidates, top_k=rerank_top_k)
            else:
                top = candidates[:rerank_top_k]
            citations = build_citations(top)
            raw_answer = await self._generate(safe_query, [c.chunk.text for c in top])
            answer_text = format_answer_with_citations(raw_answer, citations)

        response = AgentResponse(
            answer=answer_text,
            citations=citations,
            route=RouteName.RAG,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        self.cache.set(cache_key, response.model_dump(mode="json"))
        return response
