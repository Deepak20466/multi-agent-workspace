"""Web agent: retrieval-augmented answers over live Tavily web search
results, for queries the router determines aren't answerable from the
local corpus (e.g. current events, "latest version of X").
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Callable, Optional

from src.telemetry import traced_call, log_event
from src.utils.retry_handler import (
    APIConnectionError,
    RateLimitError,
    async_api_rate_limit_retry,
)
from src.utils.schemas import AgentResponse, Citation, RouteName

DEFAULT_MAX_RESULTS = 3


class WebSearchError(RuntimeError):
    pass


class WebAgent:
    def __init__(
        self,
        search_fn: Optional[Callable[[str, int], list[dict]]] = None,
        llm=None,
        tavily_api_key: Optional[str] = None,
        max_results: int = DEFAULT_MAX_RESULTS,
    ):
        """`search_fn(query, k) -> list[{"title", "url", "snippet"}]` can be
        injected (e.g. for tests or an alternate provider); otherwise a
        Tavily client is built lazily from `tavily_api_key`/`TAVILY_API_KEY`.
        """

        self.search_fn = search_fn
        self.llm = llm
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.max_results = max_results
        self._tavily_client = None

    @property
    def tavily_client(self):
        if self._tavily_client is None:
            from tavily import TavilyClient

            self._tavily_client = TavilyClient(api_key=self.tavily_api_key)
        return self._tavily_client

    def _tavily_search(self, query: str, k: int) -> list[dict]:
        response = self.tavily_client.search(query=query, max_results=k)
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in response.get("results", [])
        ]

    @async_api_rate_limit_retry
    async def _search(self, query: str, k: int) -> list[dict]:
        if self.search_fn is not None:
            fn = self.search_fn
        elif self.tavily_api_key:
            fn = self._tavily_search
        else:
            raise WebSearchError("no search_fn or TAVILY_API_KEY configured for WebAgent")

        try:
            return await asyncio.to_thread(fn, query, k)
        except Exception as exc:
            if "rate" in type(exc).__name__.lower() or "429" in str(exc):
                raise RateLimitError(str(exc)) from exc
            raise APIConnectionError(str(exc)) from exc

    async def _generate(self, query: str, results: list[dict]) -> str:
        snippets = [f"[{i}] {r.get('title', '')}: {r.get('snippet', '')}" for i, r in enumerate(results, start=1)]
        if self.llm is None:
            return f"Web results for '{query}':\n" + "\n".join(snippets)
        prompt = (
            "Answer the question using only the numbered web snippets below. "
            "Cite as [n].\n\n" + "\n".join(snippets) + f"\n\nQuestion: {query}"
        )
        return await self.llm.ainvoke(prompt)

    async def answer(self, query: str, k: Optional[int] = None) -> AgentResponse:
        start = time.perf_counter()
        with traced_call("web"):
            try:
                results = await self._search(query, k or self.max_results)
            except (WebSearchError, RateLimitError, APIConnectionError) as exc:
                log_event("web_search_failed", query=query, error=str(exc))
                return AgentResponse(
                    answer=f"Web search unavailable: {exc}",
                    route=RouteName.WEB,
                    latency_ms=(time.perf_counter() - start) * 1000,
                )

            answer_text = await self._generate(query, results)
            citations = [
                Citation(
                    id=i,
                    type="web",
                    source=r.get("url", ""),
                    title=r.get("title") or None,
                    quote=r.get("snippet", ""),
                )
                for i, r in enumerate(results, start=1)
            ]

        return AgentResponse(
            answer=answer_text,
            citations=citations,
            route=RouteName.WEB,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
