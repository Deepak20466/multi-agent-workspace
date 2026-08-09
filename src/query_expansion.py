"""Multi-query expansion for RAG-Fusion style retrieval.

Asks the LLM to paraphrase a user query into several alternative
phrasings so hybrid retrieval isn't limited to recall for one exact
wording; the caller fuses per-query result lists together (see
hybrid_retrieval.HybridRetriever.retrieve).
"""

from __future__ import annotations

from typing import List, Optional

from loguru import logger

from src.utils.retry_handler import async_api_rate_limit_retry

DEFAULT_N_ALTERNATIVES = 3


def _extract_text(response: object) -> str:
    """Agent-level `llm` objects in this codebase are expected to return
    a plain string from `.ainvoke`, but a raw langchain ChatModel returns
    a message object instead — accept either.
    """

    content = getattr(response, "content", None)
    return str(content) if content is not None else str(response)


class QueryExpander:
    """Generates paraphrased variants of a query via an LLM."""

    def __init__(self, llm: Optional[object] = None):
        self.llm = llm

    @async_api_rate_limit_retry
    async def multi_query(self, query: str, n: int = DEFAULT_N_ALTERNATIVES) -> List[str]:
        """Return the original query plus `n` LLM-generated alternative
        phrasings. Degrades gracefully to just [query] when no LLM is
        configured.
        """

        if self.llm is None:
            return [query]

        prompt = (
            f"Generate {n} alternative phrasings of the following search query, "
            "preserving its meaning but varying the wording. Return only the "
            "alternatives, one per line, with no numbering or commentary.\n\n"
            f"Query: {query}"
        )
        raw = await self.llm.ainvoke(prompt)
        alternatives = [line.strip() for line in _extract_text(raw).splitlines() if line.strip()][:n]

        logger.info("expanded query '{}' into {} alternative(s)", query, len(alternatives))
        return [query, *alternatives]

    @async_api_rate_limit_retry
    async def hyde(self, query: str) -> str:
        """Generate a Hypothetical Document Embedding: an LLM-written
        passage that *answers* the query, as if it were an excerpt from
        an authoritative source. Embedding this passage (instead of, or
        alongside, the bare question) often lands closer in vector space
        to the real supporting documents than the question itself does.

        Degrades gracefully to the original `query` when no LLM is
        configured.
        """

        if self.llm is None:
            return query

        prompt = (
            "Write a short hypothetical passage (3-5 sentences) that would "
            "directly answer the following question, as if it were an "
            "excerpt from an authoritative document. Do not mention that "
            "it is hypothetical or that you are an AI.\n\n"
            f"Question: {query}"
        )
        raw = await self.llm.ainvoke(prompt)
        passage = _extract_text(raw).strip()

        logger.info("generated HyDE passage for query '{}'", query)
        return passage or query
