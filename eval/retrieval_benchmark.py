"""Lightweight retrieval-quality benchmark for the existing hybrid RAG
retrieval pipeline: Hit@K / Recall@K / MRR against a small (~10-20
query), non-LLM ground-truth set derived deterministically from the
project's existing eval testsets.

This is deliberately narrower than eval/run_ragas_eval.py's RAG section:
it never calls an LLM (no Ollama, no judge, no query-expansion), so it
stays cheap enough to run on a laptop CPU. It exercises the real
`HybridRetriever` (dense + BM25 + alpha-weighted RRF, see
src/hybrid_retrieval.py) and, optionally, the real `Reranker` (see
src/reranking.py) -- the same objects `RAGAgent`/main.py wire up --
just without `RAGAgent`'s own LLM-backed multi-query/HyDE expansion and
answer generation stages, which this benchmark has no need for and which
would require the configured LLM backend to be reachable.

Ground truth provenance (see `_derive_rag_ground_truth`/
`_derive_doc_ground_truth` for exact per-row derivation):

- eval/testset.json rows: a row's `contexts` field is already the exact
  chunk text(s) of its `source_file` -- `generate_testset.py` sets
  `contexts = [c.text for c in chunks[:3]]` verbatim, no paraphrasing.
  Relevant chunk_ids are recovered by re-chunking `source_file` with the
  project's real `DocumentProcessor` (same defaults `generate_testset.py`
  used) and matching by exact text equality.
- eval/doc_testset.json rows: `expected_contains` substrings are matched
  by case-insensitive containment against the real chunks of `file_path`;
  any chunk containing at least one expected substring counts as
  relevant for that question. A handful of rows don't match anything
  verbatim -- an artifact of the small local model that generated
  `expected_contains` -- and are reported as skipped, not guessed at.

Both derivations are plain, deterministic string matching against chunks
produced by the project's real `DocumentProcessor` -- no LLM, no
fabricated relevance labels.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from datetime import datetime, timezone
from math import ceil
from pathlib import Path

from dotenv import load_dotenv

# Standalone entry point -- mirrors eval/run_ragas_eval.py's own
# load_dotenv() call (see its comment): DATABASE_URL/etc. from a local
# .env are otherwise invisible to this script's own env reads.
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retrieval_benchmark")

BENCHMARK_NAME = "retrieval_benchmark"
BENCHMARK_VERSION = "1.0"
DEFAULT_K_VALUES = (1, 3, 5, 10)
DEFAULT_LIMIT = 20
DEFAULT_RAG_TESTSET = "eval/testset.json"
DEFAULT_DOC_TESTSET = "eval/doc_testset.json"
DEFAULT_SAMPLE_DIR = "data/sample_documents"
DEFAULT_OUTPUT = "eval/retrieval_results.json"


# ---------------------------------------------------------------------------
# Metrics -- pure functions, no I/O, no LLM.
# ---------------------------------------------------------------------------


def hit_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1.0 if at least one relevant id appears in the top `k` retrieved
    ids, else 0.0. Undefined (raises) without ground truth to check
    against -- callers must never score a query with no known relevant
    chunks."""

    if not relevant_ids:
        raise ValueError("relevant_ids must be non-empty -- Hit@K is undefined without ground truth")
    return 1.0 if set(retrieved_ids[:k]) & relevant_ids else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of `relevant_ids` present in the top `k` retrieved ids."""

    if not relevant_ids:
        raise ValueError("relevant_ids must be non-empty -- Recall@K is undefined without ground truth")
    hits = len(set(retrieved_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1/rank of the first relevant id in `retrieved_ids` (1-indexed), or
    0.0 if none of `relevant_ids` appears anywhere in the list."""

    if not relevant_ids:
        raise ValueError("relevant_ids must be non-empty -- MRR is undefined without ground truth")
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. No numpy dependency for the handful of
    samples (~10-20 queries) this benchmark ever produces."""

    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, ceil(pct / 100 * len(ordered)) - 1))
    return ordered[idx]


# ---------------------------------------------------------------------------
# Ground truth derivation -- see module docstring for provenance.
# ---------------------------------------------------------------------------


def _derive_rag_ground_truth(testset_path: str, processor) -> tuple[list[dict], list[dict]]:
    path = Path(testset_path)
    if not path.exists():
        return [], []
    rows = json.loads(path.read_text(encoding="utf-8"))

    usable: list[dict] = []
    skipped: list[dict] = []
    chunks_by_file: dict[str, list | None] = {}

    for i, row in enumerate(rows):
        source_file = row.get("source_file")
        question = row.get("question", "")
        contexts = row.get("contexts") or []
        base = {"id": f"rag:{i}", "question": question, "source_dataset": "rag_testset", "source_file": source_file}

        if not source_file or not contexts:
            skipped.append({**base, "reason": "row has no source_file/contexts to derive ground truth from"})
            continue

        if source_file not in chunks_by_file:
            try:
                _, chunks_by_file[source_file] = processor.process(source_file)
            except Exception as exc:
                chunks_by_file[source_file] = None
                logger.warning("could not re-process %s for ground truth: %s", source_file, exc)

        chunks = chunks_by_file[source_file]
        if not chunks:
            skipped.append({**base, "reason": f"could not process source_file {source_file!r}"})
            continue

        text_to_id = {c.text: c.chunk_id for c in chunks}
        relevant_ids = {text_to_id[ctx] for ctx in contexts if ctx in text_to_id}
        if not relevant_ids:
            skipped.append(
                {
                    **base,
                    "reason": (
                        "none of this row's contexts match the current chunking of source_file verbatim "
                        "(dataset may be stale relative to the document or chunking defaults)"
                    ),
                }
            )
            continue

        usable.append({**base, "relevant_chunk_ids": sorted(relevant_ids)})

    return usable, skipped


def _derive_doc_ground_truth(doc_testset_path: str, processor) -> tuple[list[dict], list[dict]]:
    path = Path(doc_testset_path)
    if not path.exists():
        return [], []
    rows = json.loads(path.read_text(encoding="utf-8"))

    usable: list[dict] = []
    skipped: list[dict] = []
    chunks_by_file: dict[str, list | None] = {}

    for i, row in enumerate(rows):
        file_path = row.get("file_path")
        question = row.get("question", "")
        expected = row.get("expected_contains") or []
        base = {"id": f"doc:{i}", "question": question, "source_dataset": "doc_testset", "source_file": file_path}

        if not file_path or not expected:
            skipped.append({**base, "reason": "row has no file_path/expected_contains to derive ground truth from"})
            continue

        if file_path not in chunks_by_file:
            try:
                _, chunks_by_file[file_path] = processor.process(file_path)
            except Exception as exc:
                chunks_by_file[file_path] = None
                logger.warning("could not re-process %s for ground truth: %s", file_path, exc)

        chunks = chunks_by_file[file_path]
        if not chunks:
            skipped.append({**base, "reason": f"could not process file_path {file_path!r}"})
            continue

        relevant_ids: set[str] = set()
        for substring in expected:
            needle = substring.strip().lower()
            if not needle:
                continue
            for c in chunks:
                if needle in c.text.lower():
                    relevant_ids.add(c.chunk_id)

        if not relevant_ids:
            skipped.append(
                {
                    **base,
                    "reason": (
                        "none of this row's expected_contains substrings appear verbatim in any chunk of "
                        "file_path (likely a testset-generation artifact from a weak local model)"
                    ),
                }
            )
            continue

        usable.append({**base, "relevant_chunk_ids": sorted(relevant_ids)})

    return usable, skipped


def _default_processor():
    from src.config import load_config
    from src.document_processing import DocumentProcessor

    config = load_config()
    return DocumentProcessor(
        ocr_lang=config.doc_intelligence.ocr_lang,
        ocr_dpi=config.doc_intelligence.ocr_dpi,
        mask_pii=config.doc_intelligence.redact_pii,
    )


def build_queries(
    rag_testset_path: str = DEFAULT_RAG_TESTSET,
    doc_testset_path: str = DEFAULT_DOC_TESTSET,
    processor=None,
) -> tuple[list[dict], list[dict]]:
    """Combined (usable, skipped) query pool from both eval testsets. Each
    usable row carries `relevant_chunk_ids`; each skipped row carries a
    human-readable `reason` (never a fabricated relevance label)."""

    processor = processor or _default_processor()
    rag_usable, rag_skipped = _derive_rag_ground_truth(rag_testset_path, processor)
    doc_usable, doc_skipped = _derive_doc_ground_truth(doc_testset_path, processor)
    return rag_usable + doc_usable, rag_skipped + doc_skipped


# ---------------------------------------------------------------------------
# Real retrieval pipeline wiring -- same classes/config main.py uses.
# ---------------------------------------------------------------------------


def _build_real_retriever(sample_dir: str = DEFAULT_SAMPLE_DIR, processor=None):
    """A `HybridRetriever` over the project's real `VectorStore` (already
    persisted at CHROMA_PERSIST_DIR) with alpha from config.yaml, and a
    BM25 side-index built once over the *whole* sample corpus.

    Note on that BM25 build: `BM25Index.build()` replaces rather than
    accumulates its chunk list. main.py's `/ingest` endpoint and `ingest`
    CLI command both call `index_corpus(chunks)` once per uploaded/
    indexed file -- so after ingesting more than one file, the live app's
    in-memory BM25 index only reflects the *last* file processed, not the
    full corpus. That's an existing behavior detail of main.py, out of
    scope for this read-only benchmark to change (flagged in the
    benchmark's final report). This function instead makes the one
    `index_corpus()` call `index_corpus()`'s own docstring describes
    ("call after adding chunks to the vector store") over the complete
    known corpus, so hybrid retrieval is evaluated as actually configured
    rather than against a partially-built BM25 index.
    """

    from src.config import load_config
    from src.hybrid_retrieval import HybridRetriever
    from src.vectorstore import VectorStore

    processor = processor or _default_processor()
    config = load_config()
    vector_store = VectorStore()
    retriever = HybridRetriever(vector_store, alpha=config.retrieval.alpha)

    all_chunks = []
    for file_path in sorted(Path(sample_dir).glob("*")):
        if file_path.is_dir():
            continue
        try:
            _, chunks = processor.process(file_path)
            all_chunks.extend(chunks)
        except Exception as exc:
            logger.warning("skipping %s while building BM25 corpus: %s", file_path, exc)

    if all_chunks:
        retriever.index_corpus(all_chunks)

    return retriever


def _build_real_reranker():
    from src.config import load_config
    from src.reranking import Reranker

    config = load_config()
    return Reranker(use_flashrank=config.retrieval.use_flashrank, flashrank_model=config.retrieval.flashrank_model)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


async def evaluate_queries(
    queries: list[dict],
    retriever,
    reranker=None,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict:
    """Runs each query through `retriever.retrieve([question], top_k=max(k_values))`
    once -- a single query string, no multi-query/HyDE expansion, since
    those require an LLM this benchmark deliberately never calls -- then,
    if `reranker` is given, reorders the full candidate set via its real
    `arerank` (matching the production reranking cascade) before scoring.

    A retrieval failure for one query is recorded in `failed` and
    excluded from aggregate scoring rather than aborting the whole run.
    """

    max_k = max(k_values)
    per_query: list[dict] = []
    failed: list[dict] = []
    latencies_ms: list[float] = []

    for query in queries:
        question = query["question"]
        relevant_ids = set(query["relevant_chunk_ids"])
        start = time.perf_counter()
        try:
            candidates = await retriever.retrieve([question], top_k=max_k)
            if reranker is not None and candidates:
                candidates = await reranker.arerank(question, candidates, top_k=len(candidates))
            elapsed_ms = (time.perf_counter() - start) * 1000
            retrieved_ids = [c.chunk.chunk_id for c in candidates]
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning("retrieval failed for query %r: %s", question, exc)
            failed.append(
                {
                    "id": query["id"],
                    "question": question,
                    "source_dataset": query["source_dataset"],
                    "latency_ms": elapsed_ms,
                    "error": str(exc),
                }
            )
            continue

        latencies_ms.append(elapsed_ms)
        per_query.append(
            {
                "id": query["id"],
                "question": question,
                "source_dataset": query["source_dataset"],
                "n_relevant": len(relevant_ids),
                "n_retrieved": len(retrieved_ids),
                "latency_ms": elapsed_ms,
                "hit_at_k": {k: hit_at_k(retrieved_ids, relevant_ids, k) for k in k_values},
                "recall_at_k": {k: recall_at_k(retrieved_ids, relevant_ids, k) for k in k_values},
                "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
            }
        )

    n = len(per_query)
    aggregate = {
        "n_evaluated": n,
        "n_failed": len(failed),
        "hit_at_k": {k: (sum(r["hit_at_k"][k] for r in per_query) / n if n else None) for k in k_values},
        "recall_at_k": {k: (sum(r["recall_at_k"][k] for r in per_query) / n if n else None) for k in k_values},
        "mrr": (statistics.fmean(r["reciprocal_rank"] for r in per_query) if n else None),
        "avg_latency_ms": (statistics.fmean(latencies_ms) if latencies_ms else None),
        "p95_latency_ms": (_percentile(latencies_ms, 95) if latencies_ms else None),
    }
    return {"aggregate": aggregate, "per_query": per_query, "failed": failed}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _print_summary(result: dict) -> None:
    print(f"\n=== {result['benchmark']} v{result['version']} ({result['timestamp']}) ===")
    print(
        f"query pool: {result['n_query_pool']}  usable ground truth: {result['n_usable_ground_truth']}  "
        f"evaluated: {result['n_evaluated']}  skipped: {result['n_skipped_ground_truth']}  "
        f"failed: {result['n_failed']}"
    )
    for k in result["config"]["k_values"]:
        hit = result["hit_at_k"].get(k)
        recall = result["recall_at_k"].get(k)
        hit_s = f"{hit:.3f}" if hit is not None else "n/a"
        recall_s = f"{recall:.3f}" if recall is not None else "n/a"
        print(f"  Hit@{k}: {hit_s}   Recall@{k}: {recall_s}")
    mrr = result.get("mrr")
    print(f"  MRR: {mrr:.3f}" if mrr is not None else "  MRR: n/a")
    if result.get("avg_latency_ms") is not None:
        print(f"  avg latency: {result['avg_latency_ms']:.1f} ms   p95: {result['p95_latency_ms']:.1f} ms")
    if result["n_skipped_ground_truth"]:
        print(f"  ({result['n_skipped_ground_truth']} query(ies) skipped -- no recoverable ground truth)")
    if result["n_failed"]:
        print(f"  ({result['n_failed']} query(ies) failed during retrieval -- see failed_queries)")


def run_benchmark(
    rag_testset_path: str = DEFAULT_RAG_TESTSET,
    doc_testset_path: str = DEFAULT_DOC_TESTSET,
    sample_dir: str = DEFAULT_SAMPLE_DIR,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    limit: int | None = DEFAULT_LIMIT,
    use_rerank: bool = True,
    output_path: str | None = DEFAULT_OUTPUT,
    queries: list[dict] | None = None,
    skipped_queries: list[dict] | None = None,
    retriever=None,
    reranker=None,
) -> dict:
    """Runs the full benchmark: derive ground truth (unless `queries` is
    injected, e.g. by a test), build the real retriever/reranker (unless
    injected), score Hit@K/Recall@K/MRR, write `output_path`, print a
    summary, and return the result dict.

    `queries`/`skipped_queries`/`retriever`/`reranker` are injectable so
    tests can exercise this orchestration without touching Chroma, the
    sentence-transformers cross-encoder, or the real sample corpus.
    """

    if queries is None:
        queries, skipped_queries = build_queries(rag_testset_path, doc_testset_path)
    skipped_queries = skipped_queries or []

    n_pool = len(queries) + len(skipped_queries)
    evaluated_queries = queries if not limit or limit <= 0 else queries[:limit]

    if not evaluated_queries:
        result = {
            "status": (
                "skipped: no query in the configured testsets has recoverable chunk-level ground truth "
                "(see skipped_queries for reasons)"
            ),
            "benchmark": BENCHMARK_NAME,
            "version": BENCHMARK_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_query_pool": n_pool,
            "n_usable_ground_truth": len(queries),
            "skipped_queries": skipped_queries,
        }
        if output_path:
            Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.warning(result["status"])
        return result

    if retriever is None:
        retriever = _build_real_retriever(sample_dir)
    if use_rerank and reranker is None:
        reranker = _build_real_reranker()
    elif not use_rerank:
        reranker = None

    eval_result = asyncio.run(evaluate_queries(evaluated_queries, retriever, reranker, k_values))
    aggregate = eval_result["aggregate"]

    result = {
        "benchmark": BENCHMARK_NAME,
        "version": BENCHMARK_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "k_values": list(k_values),
            "limit": limit,
            "use_rerank": use_rerank,
            "rag_testset": rag_testset_path,
            "doc_testset": doc_testset_path,
            "sample_dir": sample_dir,
        },
        "n_query_pool": n_pool,
        "n_usable_ground_truth": len(queries),
        "n_evaluated": aggregate["n_evaluated"],
        "n_skipped_ground_truth": len(skipped_queries),
        "n_failed": aggregate["n_failed"],
        "hit_at_k": aggregate["hit_at_k"],
        "recall_at_k": aggregate["recall_at_k"],
        "mrr": aggregate["mrr"],
        "avg_latency_ms": aggregate["avg_latency_ms"],
        "p95_latency_ms": aggregate["p95_latency_ms"],
        "skipped_queries": skipped_queries,
        "failed_queries": eval_result["failed"],
        "per_query": eval_result["per_query"],
    }

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("wrote retrieval benchmark results to %s", output_path)

    _print_summary(result)
    return result


def _parse_k_values(raw: str) -> tuple[int, ...]:
    return tuple(sorted({int(x) for x in raw.split(",") if x.strip()}))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lightweight retrieval-quality benchmark (Hit@K/Recall@K/MRR) -- no LLM, no Ollama."
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help="Max queries to evaluate; <=0 evaluates every usable query. Default: %(default)s",
    )
    parser.add_argument("--k", default=",".join(str(k) for k in DEFAULT_K_VALUES), help="Comma-separated K values.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--rag-testset", default=DEFAULT_RAG_TESTSET)
    parser.add_argument("--doc-testset", default=DEFAULT_DOC_TESTSET)
    parser.add_argument("--sample-dir", default=DEFAULT_SAMPLE_DIR)
    parser.add_argument(
        "--no-rerank", action="store_true",
        help="Score the raw hybrid RRF ranking only, skipping the reranking cascade.",
    )
    args = parser.parse_args()

    run_benchmark(
        rag_testset_path=args.rag_testset,
        doc_testset_path=args.doc_testset,
        sample_dir=args.sample_dir,
        k_values=_parse_k_values(args.k),
        limit=args.limit,
        use_rerank=not args.no_rerank,
        output_path=args.output,
    )


if __name__ == "__main__":
    import sys

    # See eval/run_ragas_eval.py's identical block for why this is needed
    # for `python eval/retrieval_benchmark.py` (though `python -m
    # eval.retrieval_benchmark` is the preferred invocation and doesn't
    # need it).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
