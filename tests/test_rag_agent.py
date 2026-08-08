"""Tests for RAGAgent.answer()'s citation-marker verification.

Root cause: `verify_citation_markers`/`strip_invalid_citation_markers`
(src/citation.py) existed and were unit-tested in isolation, but nothing
in the actual RAG answer-generation path ever called them -- a
hallucinated `[n]` citation marker from the LLM (referencing a source
number that doesn't exist) would render in the final answer unfiltered.
These tests exercise `RAGAgent.answer()` directly (mocking the
retriever/LLM, not the citation-verification logic itself) to prove the
wiring actually runs in production code, not just in test_citations.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from src.agents.rag_agent import RAGAgent
from src.cache import ResponseCache
from src.utils.schemas import Chunk, RetrievedChunk


def _fake_pii_guard() -> MagicMock:
    guard = MagicMock()
    guard.anonymize.side_effect = lambda text: (text, [])
    return guard


def _retrieved_chunk(chunk_id: str, text: str, source_path: str) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="d1",
        text=text,
        chunk_index=0,
        metadata={"source_path": source_path},
    )
    return RetrievedChunk(chunk=chunk, vector_score=0.9, rerank_score=0.9)


def _make_agent(llm_response_text: str, chunks: list[RetrievedChunk]) -> RAGAgent:
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=chunks)

    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=llm_response_text))

    return RAGAgent(
        retriever=retriever,
        reranker=MagicMock(),
        cache=ResponseCache(redis_url=None),
        pii_guard=_fake_pii_guard(),
        llm=llm,
        use_multi_query=False,
        use_hyde=False,
        use_rerank=False,
    )


async def test_answer_with_valid_citation_markers_passes_through_unchanged():
    chunks = [
        _retrieved_chunk("1", "Refunds are allowed within 30 days.", "policy.txt"),
        _retrieved_chunk("2", "Refunds are processed within 5 business days.", "policy.txt"),
    ]
    agent = _make_agent("You have 30 days to return an item [1], refunded within 5 days [2].", chunks)

    response = await agent.answer("What is the refund window?")

    assert "[1]" in response.answer
    assert "[2]" in response.answer
    assert len(response.citations) == 2
    assert response.citations[0].source == "policy.txt"


async def test_answer_strips_hallucinated_citation_marker():
    """Only one real chunk is retrieved, but the LLM hallucinates a [2]
    marker -- the final answer must not contain it, while the one real
    citation (its ID and source metadata) must still be present.
    """

    chunks = [_retrieved_chunk("1", "Refunds are allowed within 30 days.", "policy.txt")]
    agent = _make_agent("You have 30 days to return an item [1], see also [2] for exchanges.", chunks)

    response = await agent.answer("What is the refund window?")

    assert "[2]" not in response.answer
    assert "[1]" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].id == 1
    assert response.citations[0].source == "policy.txt"


async def test_answer_citation_verification_failure_does_not_raise():
    """The verification failure path must degrade gracefully (strip and
    continue), never propagate an exception out of `.answer()`.
    """

    chunks = [_retrieved_chunk("1", "Refunds are allowed within 30 days.", "policy.txt")]
    agent = _make_agent("Totally hallucinated citation [99].", chunks)

    response = await agent.answer("What is the refund window?")

    assert "[99]" not in response.answer
    assert response.citations


async def test_answer_returns_no_citations_when_no_relevant_context():
    """No candidates clear the relevance bar -- the agent must answer
    honestly instead of asking the LLM to generate from empty context
    (which invites hallucination), and never call the LLM at all.
    """

    agent = _make_agent("irrelevant -- llm must not be called", chunks=[])

    response = await agent.answer("What is the refund window?")

    assert response.citations == []
    assert "don't have enough relevant information" in response.answer
    agent.llm.ainvoke.assert_not_called()
