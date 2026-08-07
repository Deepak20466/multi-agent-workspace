from unittest.mock import MagicMock

from src.hybrid_retrieval import BM25Index, HybridRetriever, reciprocal_rank_fusion
from src.utils.schemas import Chunk, RetrievedChunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="doc-1", text=text, chunk_index=0)


def test_reciprocal_rank_fusion_favors_items_ranked_high_in_both_lists():
    a = RetrievedChunk(chunk=_chunk("a", "alpha"), vector_score=0.9)
    b = RetrievedChunk(chunk=_chunk("b", "beta"), vector_score=0.5)
    c = RetrievedChunk(chunk=_chunk("c", "gamma"), bm25_score=5.0)

    vector_list = [a, b]
    bm25_list = [b, c]

    fused = reciprocal_rank_fusion([vector_list, bm25_list])

    assert fused[0].chunk.chunk_id == "b"  # ranked in both lists
    assert {r.chunk.chunk_id for r in fused} == {"a", "b", "c"}
    assert fused[0].rrf_score >= fused[1].rrf_score >= fused[2].rrf_score


def test_bm25_index_search_returns_empty_before_build():
    index = BM25Index()
    assert index.search("query") == []


def test_bm25_index_search_ranks_by_term_overlap():
    index = BM25Index()
    index.build([_chunk("1", "cats and dogs"), _chunk("2", "only dogs here"), _chunk("3", "unrelated text")])

    results = index.search("dogs", k=2)
    ids = [r.chunk.chunk_id for r in results]
    assert "3" not in ids


async def test_hybrid_retriever_falls_back_to_vector_only_when_bm25_empty():
    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [RetrievedChunk(chunk=_chunk("a", "alpha"), vector_score=0.9)]

    retriever = HybridRetriever(mock_store)
    results = await retriever.retrieve(["query"], top_k=5)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "a"


async def test_hybrid_retriever_fuses_across_multiple_query_variants():
    mock_store = MagicMock()
    mock_store.similarity_search.side_effect = [
        [RetrievedChunk(chunk=_chunk("a", "alpha"), vector_score=0.9)],
        [RetrievedChunk(chunk=_chunk("b", "beta"), vector_score=0.8)],
    ]
    bm25 = BM25Index()
    bm25.build([_chunk("a", "alpha"), _chunk("b", "beta")])

    retriever = HybridRetriever(mock_store, bm25=bm25)
    results = await retriever.retrieve(["query one", "query two"], top_k=5)

    ids = {r.chunk.chunk_id for r in results}
    assert ids == {"a", "b"}
    assert mock_store.similarity_search.call_count == 2


async def test_hybrid_retriever_alpha_zero_ignores_vector_only_results():
    # Needs >=3 docs: with exactly 2 docs, BM25's IDF for a term in half
    # the corpus is 0, which would zero out every real score too.
    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [RetrievedChunk(chunk=_chunk("a", "alpha"), vector_score=0.9)]
    bm25 = BM25Index()
    bm25.build([_chunk("a", "alpha"), _chunk("b", "dogs"), _chunk("c", "unrelated text")])

    retriever = HybridRetriever(mock_store, bm25=bm25, alpha=0.0)
    results = await retriever.retrieve(["dogs"], top_k=5)

    ids = {r.chunk.chunk_id for r in results}
    assert ids == {"b"}
