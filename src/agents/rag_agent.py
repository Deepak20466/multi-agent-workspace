"""RAG agent: expand query -> hybrid retrieve (RAG-Fusion) -> rerank ->
cite -> generate.

Ties together query_expansion (multi-query), hybrid_retrieval
(alpha-weighted RRF), reranking, and citation into a single grounded-
answer pipeline used by the router for RAG-classified queries.
"""

from __future__ import annotations

import time

from src.cache import ResponseCache
from src.citation import build_citations, format_answer_with_citations
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
        llm=None,
    ):
        self.retriever = retriever
        self.reranker = reranker or Reranker()
        self.cache = cache or ResponseCache()
        self.llm = llm
        self.query_expander = query_expander or QueryExpander(llm=llm)

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

    async def answer(self, query: str, k: int = 5) -> AgentResponse:
        cache_key = ResponseCache.make_key("rag", query)
        cached = self.cache.get(cache_key)
        if cached is not None:
            log_event("rag_cache_hit", query=query)
            return AgentResponse(**cached)

        start = time.perf_counter()
        with traced_call("rag"):
            queries = await self.query_expander.multi_query(query)
            candidates = await self.retriever.retrieve(queries, top_k=k * 3)
            top = self.reranker.rerank(query, candidates, top_k=k)
            citations = build_citations(top)
            raw_answer = await self._generate(query, [c.chunk.text for c in top])
            answer_text = format_answer_with_citations(raw_answer, citations)

        response = AgentResponse(
            answer=answer_text,
            citations=citations,
            route=RouteName.RAG,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        self.cache.set(cache_key, response.model_dump(mode="json"))
        return response
