import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

from src.reranking import Reranker
from src.utils.schemas import Chunk, RetrievedChunk


def _candidate(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(chunk_id=chunk_id, doc_id="d", text=text, chunk_index=0), rrf_score=0.1)


def test_rerank_orders_by_cross_encoder_score():
    reranker = Reranker()
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.2, 0.9]

    with patch.object(Reranker, "model", new=mock_model):
        results = reranker.rerank("query", [_candidate("a", "low"), _candidate("b", "high")], top_k=2)

    assert [r.chunk.chunk_id for r in results] == ["b", "a"]
    assert results[0].rerank_score == 0.9


def test_rerank_empty_candidates_returns_empty():
    reranker = Reranker()
    assert reranker.rerank("query", [], top_k=5) == []


def test_rerank_respects_top_k():
    reranker = Reranker()
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.5, 0.9]

    candidates = [_candidate("a", "x"), _candidate("b", "y"), _candidate("c", "z")]
    with patch.object(Reranker, "model", new=mock_model):
        results = reranker.rerank("query", candidates, top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c"


async def test_arerank_uses_local_cross_encoder_when_no_cohere_key():
    reranker = Reranker(cohere_api_key=None)
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9, 0.1]

    candidates = [_candidate("a", "high"), _candidate("b", "low")]
    with patch.object(Reranker, "model", new=mock_model):
        results = await reranker.arerank("query", candidates, top_k=2)

    assert [r.chunk.chunk_id for r in results] == ["a", "b"]


async def test_arerank_uses_cohere_when_api_key_configured():
    reranker = Reranker(cohere_api_key="fake-key")
    mock_client = MagicMock()
    mock_result = MagicMock(
        results=[MagicMock(index=1, relevance_score=0.9), MagicMock(index=0, relevance_score=0.2)]
    )
    mock_client.rerank = AsyncMock(return_value=mock_result)

    candidates = [_candidate("a", "low"), _candidate("b", "high")]
    with patch.object(Reranker, "cohere_client", new=mock_client):
        results = await reranker.arerank("query", candidates, top_k=2)

    assert [r.chunk.chunk_id for r in results] == ["b", "a"]
    assert results[0].rerank_score == 0.9


async def test_arerank_falls_back_to_local_when_cohere_call_fails():
    reranker = Reranker(cohere_api_key="fake-key")
    mock_cohere_client = MagicMock()
    mock_cohere_client.rerank = AsyncMock(side_effect=RuntimeError("cohere down"))
    mock_local_model = MagicMock()
    mock_local_model.predict.return_value = [0.3, 0.8]

    candidates = [_candidate("a", "low"), _candidate("b", "high")]
    with patch.object(Reranker, "cohere_client", new=mock_cohere_client), patch.object(
        Reranker, "model", new=mock_local_model
    ):
        results = await reranker.arerank("query", candidates, top_k=2)

    assert [r.chunk.chunk_id for r in results] == ["b", "a"]


async def test_arerank_empty_candidates_returns_empty():
    reranker = Reranker(cohere_api_key="fake-key")
    assert await reranker.arerank("query", [], top_k=5) == []


# ---------------------------------------------------------------------------
# flashrank tier
# ---------------------------------------------------------------------------


async def test_arerank_uses_flashrank_when_enabled_and_no_cohere_key():
    reranker = Reranker(cohere_api_key=None, use_flashrank=True)
    candidates = [_candidate("a", "low"), _candidate("b", "high")]
    flashrank_result = [
        RetrievedChunk(chunk=candidates[1].chunk, rerank_score=0.9),
        RetrievedChunk(chunk=candidates[0].chunk, rerank_score=0.2),
    ]
    reranker._flashrank_rerank = MagicMock(return_value=flashrank_result)

    results = await reranker.arerank("query", candidates, top_k=2)

    reranker._flashrank_rerank.assert_called_once_with("query", candidates, 2)
    assert [r.chunk.chunk_id for r in results] == ["b", "a"]


async def test_arerank_falls_back_to_local_when_flashrank_fails():
    reranker = Reranker(cohere_api_key=None, use_flashrank=True)
    reranker._flashrank_rerank = MagicMock(side_effect=RuntimeError("flashrank down"))
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9]
    candidates = [_candidate("a", "low"), _candidate("b", "high")]

    with patch.object(Reranker, "model", new=mock_model):
        results = await reranker.arerank("query", candidates, top_k=2)

    assert [r.chunk.chunk_id for r in results] == ["b", "a"]


async def test_arerank_skips_flashrank_when_not_enabled():
    reranker = Reranker(cohere_api_key=None, use_flashrank=False)
    reranker._flashrank_rerank = MagicMock(side_effect=AssertionError("flashrank should not be called"))
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9, 0.1]
    candidates = [_candidate("a", "high"), _candidate("b", "low")]

    with patch.object(Reranker, "model", new=mock_model):
        results = await reranker.arerank("query", candidates, top_k=2)

    reranker._flashrank_rerank.assert_not_called()
    assert [r.chunk.chunk_id for r in results] == ["a", "b"]


async def test_arerank_cohere_takes_priority_over_flashrank():
    reranker = Reranker(cohere_api_key="fake-key", use_flashrank=True)
    mock_client = MagicMock()
    mock_result = MagicMock(
        results=[MagicMock(index=1, relevance_score=0.9), MagicMock(index=0, relevance_score=0.1)]
    )
    mock_client.rerank = AsyncMock(return_value=mock_result)
    reranker._flashrank_rerank = MagicMock(side_effect=AssertionError("flashrank should not be called"))

    candidates = [_candidate("a", "low"), _candidate("b", "high")]
    with patch.object(Reranker, "cohere_client", new=mock_client):
        results = await reranker.arerank("query", candidates, top_k=2)

    reranker._flashrank_rerank.assert_not_called()
    assert [r.chunk.chunk_id for r in results] == ["b", "a"]


def test_flashrank_rerank_orders_by_score():
    reranker = Reranker()
    candidates = [_candidate("a", "low"), _candidate("b", "high")]

    class _FakeRerankRequest:
        def __init__(self, query, passages):
            self.query = query
            self.passages = passages

    fake_flashrank = types.ModuleType("flashrank")
    fake_flashrank.RerankRequest = _FakeRerankRequest
    fake_flashrank.Ranker = MagicMock()

    mock_ranker = MagicMock()
    mock_ranker.rerank.return_value = [{"id": 1, "score": 0.9}, {"id": 0, "score": 0.2}]

    with patch.dict(sys.modules, {"flashrank": fake_flashrank}), patch.object(
        Reranker, "flashrank_ranker", new=mock_ranker
    ):
        results = reranker._flashrank_rerank("query", candidates, top_k=2)

    assert [r.chunk.chunk_id for r in results] == ["b", "a"]
    assert results[0].rerank_score == 0.9
