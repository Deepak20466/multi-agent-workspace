"""Tests for eval/retrieval_benchmark.py: deterministic metric math,
ground-truth derivation from the real (non-fabricated) eval testsets,
and the benchmark runner's orchestration -- all offline, no Ollama, no
network, no real Chroma/cross-encoder model load (the orchestration test
injects a stub retriever/reranker the same way tests/test_eval_v3.py
injects stub agents into run_ragas_eval.py).
"""

import json

import pytest

from eval.retrieval_benchmark import (
    _derive_doc_ground_truth,
    _derive_rag_ground_truth,
    build_queries,
    evaluate_queries,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    run_benchmark,
)
from src.utils.schemas import Chunk, RetrievedChunk

# ---------------------------------------------------------------------------
# hit_at_k
# ---------------------------------------------------------------------------


def test_hit_at_k_relevant_result_at_rank_1():
    assert hit_at_k(["a", "b", "c"], {"a"}, k=1) == 1.0


def test_hit_at_k_relevant_result_at_rank_3():
    retrieved = ["x", "y", "a"]
    assert hit_at_k(retrieved, {"a"}, k=1) == 0.0
    assert hit_at_k(retrieved, {"a"}, k=2) == 0.0
    assert hit_at_k(retrieved, {"a"}, k=3) == 1.0


def test_hit_at_k_no_relevant_result():
    assert hit_at_k(["x", "y", "z"], {"a"}, k=10) == 0.0


def test_hit_at_k_fewer_results_than_k():
    # Only one result exists at all; k=10 must not error, and the single
    # relevant result still counts as a hit.
    assert hit_at_k(["a"], {"a"}, k=10) == 1.0


def test_hit_at_k_raises_on_empty_ground_truth():
    with pytest.raises(ValueError):
        hit_at_k(["a", "b"], set(), k=5)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


def test_recall_at_k_multiple_relevant_results():
    retrieved = ["a", "c", "b"]
    relevant = {"a", "b"}
    assert recall_at_k(retrieved, relevant, k=1) == 0.5
    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_recall_at_k_multiple_ground_truth_relevant_documents():
    # Three ground-truth relevant chunks, only two retrievable within k.
    retrieved = ["a", "b", "z"]
    relevant = {"a", "b", "c"}
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)


def test_recall_at_k_fewer_results_than_k():
    assert recall_at_k(["a"], {"a", "b"}, k=10) == 0.5


def test_recall_at_k_no_relevant_result():
    assert recall_at_k(["x", "y"], {"a"}, k=5) == 0.0


def test_recall_at_k_raises_on_empty_ground_truth():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), k=5)


# ---------------------------------------------------------------------------
# reciprocal_rank
# ---------------------------------------------------------------------------


def test_reciprocal_rank_hit_at_rank_1():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_reciprocal_rank_hit_at_rank_3():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_hit():
    assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0


def test_reciprocal_rank_raises_on_empty_ground_truth():
    with pytest.raises(ValueError):
        reciprocal_rank(["a"], set())


# ---------------------------------------------------------------------------
# Ground-truth derivation -- real DocumentProcessor, no LLM.
# ---------------------------------------------------------------------------


def test_derive_rag_ground_truth_matches_exact_chunk_text(document_processor, tmp_path):
    sample = tmp_path / "doc.txt"
    sample.write_text("Short policy text, well under the thousand-character chunk size.")
    _, chunks = document_processor.process(sample)

    testset_path = tmp_path / "testset.json"
    testset_path.write_text(
        json.dumps([{"question": "What is this about?", "contexts": [chunks[0].text], "source_file": str(sample)}])
    )

    usable, skipped = _derive_rag_ground_truth(str(testset_path), document_processor)

    assert skipped == []
    assert len(usable) == 1
    assert usable[0]["relevant_chunk_ids"] == [chunks[0].chunk_id]
    assert usable[0]["source_dataset"] == "rag_testset"


def test_derive_rag_ground_truth_skips_when_context_does_not_match(document_processor, tmp_path):
    sample = tmp_path / "doc.txt"
    sample.write_text("Real chunk content that exists in the file.")

    testset_path = tmp_path / "testset.json"
    testset_path.write_text(
        json.dumps([{"question": "Q", "contexts": ["text that was never in the document"], "source_file": str(sample)}])
    )

    usable, skipped = _derive_rag_ground_truth(str(testset_path), document_processor)

    assert usable == []
    assert len(skipped) == 1
    assert "verbatim" in skipped[0]["reason"]


def test_derive_rag_ground_truth_skips_row_missing_fields(document_processor, tmp_path):
    testset_path = tmp_path / "testset.json"
    testset_path.write_text(json.dumps([{"question": "Q", "contexts": [], "source_file": None}]))

    usable, skipped = _derive_rag_ground_truth(str(testset_path), document_processor)

    assert usable == []
    assert len(skipped) == 1


# ---------------------------------------------------------------------------
# Doc ground-truth derivation
# ---------------------------------------------------------------------------


def test_derive_doc_ground_truth_matches_substring_case_insensitive(document_processor, tmp_path):
    sample = tmp_path / "doc.txt"
    sample.write_text("Refunds are issued within 30 days of purchase.")
    _, chunks = document_processor.process(sample)

    doc_testset_path = tmp_path / "doc_testset.json"
    doc_testset_path.write_text(
        json.dumps([{"question": "How long is the refund window?", "file_path": str(sample), "expected_contains": ["30 DAYS"]}])
    )

    usable, skipped = _derive_doc_ground_truth(str(doc_testset_path), document_processor)

    assert skipped == []
    assert len(usable) == 1
    assert usable[0]["relevant_chunk_ids"] == [chunks[0].chunk_id]


def test_derive_doc_ground_truth_skips_when_substring_not_found_verbatim(document_processor, tmp_path):
    """Mirrors a real row in eval/doc_testset.json ("What are the
    conditions for returns?" -> expected_contains=["refund_policy"]),
    where the generated substring doesn't literally appear in any chunk.
    Must be reported as skipped, never guessed at."""

    sample = tmp_path / "doc.txt"
    sample.write_text("Refunds are issued within 30 days of purchase.")

    doc_testset_path = tmp_path / "doc_testset.json"
    doc_testset_path.write_text(
        json.dumps([{"question": "Q", "file_path": str(sample), "expected_contains": ["a phrase not in the document"]}])
    )

    usable, skipped = _derive_doc_ground_truth(str(doc_testset_path), document_processor)

    assert usable == []
    assert len(skipped) == 1
    assert "artifact" in skipped[0]["reason"]


def test_build_queries_combines_rag_and_doc_sources(document_processor, tmp_path):
    sample = tmp_path / "doc.txt"
    sample.write_text("Refunds are issued within 30 days of purchase.")
    _, chunks = document_processor.process(sample)

    testset_path = tmp_path / "testset.json"
    testset_path.write_text(json.dumps([{"question": "RAG Q", "contexts": [chunks[0].text], "source_file": str(sample)}]))

    doc_testset_path = tmp_path / "doc_testset.json"
    doc_testset_path.write_text(
        json.dumps([{"question": "Doc Q", "file_path": str(sample), "expected_contains": ["30 days"]}])
    )

    usable, skipped = build_queries(str(testset_path), str(doc_testset_path), processor=document_processor)

    assert skipped == []
    assert len(usable) == 2
    assert {q["source_dataset"] for q in usable} == {"rag_testset", "doc_testset"}


# ---------------------------------------------------------------------------
# evaluate_queries / run_benchmark orchestration -- injected fakes, no
# real retriever/reranker/Ollama.
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="d", text="x", chunk_index=0)


class _StubRetriever:
    """Returns a fixed, per-question ranking of chunk_ids -- deterministic
    stand-in for the real HybridRetriever."""

    def __init__(self, ranking_by_question: dict[str, list[str]]):
        self.ranking_by_question = ranking_by_question

    async def retrieve(self, queries: list[str], top_k: int = 10):
        question = queries[0]
        ids = self.ranking_by_question[question][:top_k]
        return [RetrievedChunk(chunk=_chunk(i)) for i in ids]


class _ReversingReranker:
    """Deterministic stand-in for Reranker.arerank: reverses order, so
    tests can prove reranking actually changes the scored ranking."""

    async def arerank(self, query, candidates, top_k=5):
        return list(reversed(candidates))[:top_k]


class _FailingRetriever:
    async def retrieve(self, queries, top_k=10):
        raise RuntimeError("boom")


async def test_evaluate_queries_aggregates_hit_recall_mrr():
    queries = [
        {"id": "q1", "question": "Q1", "source_dataset": "rag_testset", "relevant_chunk_ids": ["a"]},
        {"id": "q2", "question": "Q2", "source_dataset": "doc_testset", "relevant_chunk_ids": ["x", "y"]},
    ]
    retriever = _StubRetriever({"Q1": ["a", "b", "c"], "Q2": ["z", "x", "y"]})

    result = await evaluate_queries(queries, retriever, reranker=None, k_values=(1, 3))

    agg = result["aggregate"]
    assert agg["n_evaluated"] == 2
    assert agg["n_failed"] == 0
    assert agg["hit_at_k"][1] == 0.5  # only Q1 hits at k=1 (Q2's first hit is rank 2)
    assert agg["hit_at_k"][3] == 1.0
    assert agg["recall_at_k"][3] == 1.0  # both queries fully recalled by k=3
    assert agg["mrr"] == pytest.approx((1 / 1 + 1 / 2) / 2)


async def test_evaluate_queries_applies_reranker_when_given():
    queries = [{"id": "q1", "question": "Q1", "source_dataset": "rag_testset", "relevant_chunk_ids": ["c"]}]
    # k_values' max (3) is what bounds retriever.retrieve's top_k -- all 3
    # stub candidates come back, giving the reranker a real 3-item window
    # to reorder before hit_at_k(1) is scored on the *post-rerank* top-1.
    retriever = _StubRetriever({"Q1": ["a", "b", "c"]})  # relevant chunk is last before rerank

    without_rerank = await evaluate_queries(queries, retriever, reranker=None, k_values=(1, 3))
    with_rerank = await evaluate_queries(queries, retriever, reranker=_ReversingReranker(), k_values=(1, 3))

    assert without_rerank["aggregate"]["hit_at_k"][1] == 0.0  # "c" is rank 3, missed at k=1
    assert with_rerank["aggregate"]["hit_at_k"][1] == 1.0  # reversed order puts "c" first


async def test_evaluate_queries_records_failed_query_without_raising():
    queries = [{"id": "q1", "question": "Q1", "source_dataset": "rag_testset", "relevant_chunk_ids": ["a"]}]

    result = await evaluate_queries(queries, _FailingRetriever(), reranker=None, k_values=(1,))

    assert result["aggregate"]["n_evaluated"] == 0
    assert result["aggregate"]["n_failed"] == 1
    assert result["failed"][0]["id"] == "q1"
    assert "boom" in result["failed"][0]["error"]


def test_run_benchmark_aggregates_and_writes_output(tmp_path):
    queries = [
        {
            "id": "q1", "question": "Q1", "source_dataset": "rag_testset",
            "source_file": "f1", "relevant_chunk_ids": ["a"],
        },
        {
            "id": "q2", "question": "Q2", "source_dataset": "doc_testset",
            "source_file": "f2", "relevant_chunk_ids": ["x", "y"],
        },
    ]
    retriever = _StubRetriever({"Q1": ["a", "b", "c"], "Q2": ["z", "x", "y"]})
    output_path = tmp_path / "retrieval_results.json"

    result = run_benchmark(
        queries=queries,
        skipped_queries=[{"id": "q3", "reason": "no ground truth"}],
        retriever=retriever,
        reranker=None,
        use_rerank=False,
        k_values=(1, 3),
        limit=10,
        output_path=str(output_path),
    )

    assert result["n_evaluated"] == 2
    assert result["n_skipped_ground_truth"] == 1
    assert result["n_query_pool"] == 3
    assert result["hit_at_k"][1] == 0.5
    assert result["hit_at_k"][3] == 1.0
    assert result["mrr"] == pytest.approx((1 / 1 + 1 / 2) / 2)

    assert output_path.exists()
    saved = json.loads(output_path.read_text())
    assert saved["n_evaluated"] == 2
    assert saved["benchmark"] == "retrieval_benchmark"


def test_run_benchmark_respects_limit():
    queries = [
        {"id": f"q{i}", "question": f"Q{i}", "source_dataset": "rag_testset", "source_file": "f", "relevant_chunk_ids": ["a"]}
        for i in range(5)
    ]
    retriever = _StubRetriever({f"Q{i}": ["a"] for i in range(5)})

    result = run_benchmark(
        queries=queries, skipped_queries=[], retriever=retriever, use_rerank=False,
        k_values=(1,), limit=2, output_path=None,
    )

    assert result["n_evaluated"] == 2
    assert result["n_usable_ground_truth"] == 5


def test_run_benchmark_records_failed_queries_without_raising():
    queries = [{"id": "q1", "question": "Q1", "source_dataset": "rag_testset", "source_file": "f", "relevant_chunk_ids": ["a"]}]

    result = run_benchmark(
        queries=queries, skipped_queries=[], retriever=_FailingRetriever(), use_rerank=False, output_path=None,
    )

    assert result["n_evaluated"] == 0
    assert result["n_failed"] == 1
    assert result["failed_queries"][0]["id"] == "q1"


def test_run_benchmark_reports_status_when_no_usable_ground_truth():
    result = run_benchmark(
        queries=[], skipped_queries=[{"id": "q1", "reason": "no match"}], output_path=None,
    )

    assert result["status"].startswith("skipped:")
    assert result["n_query_pool"] == 1
