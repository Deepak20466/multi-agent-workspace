from unittest.mock import MagicMock, patch

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
