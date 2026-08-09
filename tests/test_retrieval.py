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


# ---------------------------------------------------------------------------
# BM25Index.build() -- accumulate-across-calls regression tests.
#
# main.py's /ingest endpoint and `ingest` CLI command both call
# index_corpus(chunks) (-> BM25Index.build(chunks)) once per newly
# processed file, passing only that file's chunks each time -- never the
# full corpus. build() must merge into whatever's already indexed rather
# than replacing it, or every ingest after the first silently makes BM25
# blind to every previously ingested document (while dense/Chroma search,
# an upsert, keeps accumulating normally -- see VectorStore.add_chunks).
# ---------------------------------------------------------------------------


def test_bm25_index_build_accumulates_across_multiple_calls():
    # >=3 docs -- with exactly 2, BM25's IDF for a term in half the
    # corpus is 0 (see the alpha=0.0 test above), which would zero out
    # "alpha"'s own score too.
    index = BM25Index()
    index.build([_chunk("a", "alpha content")])
    index.build([_chunk("b", "beta content")])
    index.build([_chunk("c", "unrelated filler content")])

    assert {c.chunk_id for c in index.chunks} == {"a", "b", "c"}
    assert [r.chunk.chunk_id for r in index.search("alpha")] == ["a"]


def test_bm25_index_build_sequential_documents_all_remain_searchable():
    # >=3 docs, one distinguishing term each, no shared IDF-zeroing terms
    # -- see the alpha=0.0 test above for why.
    index = BM25Index()
    index.build([_chunk("a", "alpha astronomy content topic")])
    index.build([_chunk("b", "beta accounting content topic")])
    index.build([_chunk("c", "gamma gardening content topic")])

    assert {c.chunk_id for c in index.chunks} == {"a", "b", "c"}
    assert [r.chunk.chunk_id for r in index.search("astronomy")] == ["a"]
    assert [r.chunk.chunk_id for r in index.search("accounting")] == ["b"]
    assert [r.chunk.chunk_id for r in index.search("gardening")] == ["c"]


def test_bm25_index_build_reindexing_same_chunk_id_upserts_not_duplicates():
    index = BM25Index()
    index.build([_chunk("a", "original alpha content")])
    index.build([_chunk("b", "beta content")])

    # Re-indexing "a" (e.g. re-ingesting an edited file -- chunk_id is
    # stable per (source_path, chunk_index), see
    # document_processing._stable_chunk_id) must replace the existing
    # entry, not add a duplicate.
    index.build([_chunk("a", "updated alpha content")])

    assert len(index.chunks) == 2
    ids = [c.chunk_id for c in index.chunks]
    assert ids.count("a") == 1
    updated = next(c for c in index.chunks if c.chunk_id == "a")
    assert updated.text == "updated alpha content"


def test_bm25_index_build_single_call_behavior_unchanged():
    # One build() call on a fresh index -- the existing single-document
    # indexing path -- behaves exactly as before this fix.
    index = BM25Index()
    index.build([_chunk("a", "alpha content"), _chunk("b", "beta content")])

    assert {c.chunk_id for c in index.chunks} == {"a", "b"}


def test_bm25_index_build_empty_chunks_on_fresh_index_stays_unbuilt():
    index = BM25Index()
    index.build([])

    assert index.chunks == []
    assert index.is_built is False
    assert index.search("anything") == []


def test_bm25_index_build_empty_chunks_after_existing_index_is_a_no_op():
    # index_corpus([]) (e.g. a file that produced zero chunks) must not
    # wipe out chunks from documents already indexed.
    index = BM25Index()
    index.build([_chunk("a", "alpha content")])
    index.build([])

    assert {c.chunk_id for c in index.chunks} == {"a"}
    assert index.is_built is True


async def test_hybrid_retriever_index_corpus_accumulates_across_ingests_like_production():
    """End-to-end at the HybridRetriever/index_corpus level, mirroring
    main.py's actual /ingest and `ingest` CLI call pattern: one
    index_corpus(chunks) call per file, sequentially, on the same
    retriever instance."""

    mock_store = MagicMock()
    mock_store.similarity_search.return_value = []  # isolate the BM25 signal

    retriever = HybridRetriever(mock_store)
    retriever.index_corpus([_chunk("a", "alpha astronomy content topic")])
    retriever.index_corpus([_chunk("b", "beta accounting content topic")])
    retriever.index_corpus([_chunk("c", "gamma gardening content topic")])

    results_a = await retriever.retrieve(["astronomy"], top_k=5)
    results_b = await retriever.retrieve(["accounting"], top_k=5)

    assert [r.chunk.chunk_id for r in results_a] == ["a"]
    assert [r.chunk.chunk_id for r in results_b] == ["b"]
