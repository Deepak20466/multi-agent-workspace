"""Web agent: retrieval-augmented answers over live web search results,
for queries the router determines aren't answerable from the local
corpus (e.g. current events, "latest version of X").
"""

from __future__ import annotations

import time

from src.telemetry import traced_call
from src.utils.retry_handler import with_retry, RetryableError
from src.utils.schemas import AgentResponse, Citation, RouteName


class WebSearchError(RuntimeError):
    pass


class WebAgent:
    def __init__(self, search_fn=None, llm=None):
        """`search_fn(query, k) -> list[{"title", "url", "snippet"}]` is
        injected so the agent isn't coupled to one search provider.
        """
        self.search_fn = search_fn
        self.llm = llm

    @with_retry(exceptions=(RetryableError, ConnectionError, TimeoutError))
    def _search(self, query: str, k: int) -> list[dict]:
        if self.search_fn is None:
            raise WebSearchError("no search_fn configured for WebAgent")
        try:
            return self.search_fn(query, k)
        except Exception as exc:
            raise RetryableError(str(exc)) from exc

    def _generate(self, query: str, results: list[dict]) -> str:
        snippets = [f"[{i}] {r['snippet']}" for i, r in enumerate(results, start=1)]
        if self.llm is None:
            return f"Web results for '{query}':\n" + "\n".join(snippets)
        prompt = (
            "Answer the question using only the numbered web snippets below. "
            "Cite as [n].\n\n" + "\n".join(snippets) + f"\n\nQuestion: {query}"
        )
        return self.llm.invoke(prompt)

    def answer(self, query: str, k: int = 5) -> AgentResponse:
        start = time.perf_counter()
        with traced_call("web"):
            results = self._search(query, k)
            answer_text = self._generate(query, results)
            citations = [
                Citation(id=i, type="web", source=r.get("url", ""), quote=r.get("snippet", ""))
                for i, r in enumerate(results, start=1)
            ]

        return AgentResponse(
            answer=answer_text,
            citations=citations,
            route=RouteName.WEB,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
