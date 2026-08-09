"""Runs all three specialist agents against their generated testsets and
scores them, writing combined results to eval/results_v3.json for
check_gates.py to consume in CI:

- RAG: RAGAS metrics (faithfulness, answer_relevancy, context_precision,
  context_recall) over eval/testset.json.
- SQL: row-overlap accuracy between generated and ground-truth SQL over
  eval/sql_testset.json.
- Doc: LLM-judge accuracy over eval/doc_testset.json.

Each section degrades to `{"status": "skipped: <reason>"}` instead of
raising when its prerequisites aren't met -- no DATABASE_URL, no
testset, ragas itself failing to import (see `_import_ragas` below), or
the configured chat backend not actually being reachable (e.g.
`agents.llm_backend: ollama` in config.yaml but no Ollama server running,
the case on a plain GitHub-hosted CI runner) -- so a partial environment
still produces a usable results file rather than crashing the whole run.

That last case is deliberately a *pre-flight* check (`src.llm_factory.
backend_reachable`, checked before any real work starts), not a broad
try/except around the actual eval work: if the backend is reported
reachable but a real run still fails, that's let through as a genuine,
loud application error (non-zero exit) rather than being swallowed as
"skipped" -- an eval script should never quietly mask a real bug as an
environment gap. Every default-agent-construction call site below
threads config.agents.llm_backend/ollama_model/ollama_base_url through
explicitly (mirroring main.py/mcp_server.py) so the agent under
evaluation is always genuinely wired to the same backend this module
checks reachability for, rather than silently falling back to a
different backend (e.g. via an ambient LLM_BACKEND env var) than what
was actually probed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import types
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_ragas_eval")

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "claude-haiku-4-5")
EVAL_EMBEDDING_MODEL = os.getenv("EVAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def _extract_text(response: object) -> str:
    content = getattr(response, "content", None)
    return str(content) if content is not None else str(response)


def _default_llm():
    """The project's own configured chat backend -- Ollama by default
    per config.yaml (`agents.llm_backend`), or Anthropic if explicitly
    configured -- via the same `src.llm_factory.build_llm` every agent
    uses. Previously hardcoded to `ChatAnthropic`, which made every eval
    step (RAG generation, doc-eval judging, testset ground_truth) require
    ANTHROPIC_API_KEY even on a machine set up to run fully offline with
    Ollama.
    """

    from src.config import load_config
    from src.llm_factory import build_llm

    config = load_config()
    return build_llm(
        JUDGE_MODEL,
        backend=config.agents.llm_backend,
        ollama_model=config.agents.ollama_model,
        ollama_base_url=config.agents.ollama_base_url,
        temperature=0,
    )


def _ragas_judge_llm_and_embeddings():
    """Wrap the project's configured chat backend + a local
    sentence-transformers embedding model (the same `all-MiniLM-L6-v2`
    model `VectorStore` already uses) as RAGAS's LLM/embeddings judges.

    `ragas.evaluate()` silently defaults to an OpenAI-backed judge and
    embeddings model when `llm`/`embeddings` aren't passed explicitly --
    the actual reason RAGAS eval required OPENAI_API_KEY even when this
    project is configured end-to-end for local Ollama. Returns
    `(None, None)` (letting `evaluate()` fall back to its own defaults)
    if wrapping fails for any reason, so a misconfigured local setup
    degrades instead of blocking the whole eval run.
    """

    try:
        # `ragas.embeddings`/`ragas.llms` transitively hit the same
        # vertexai import break `_import_ragas()` works around -- this
        # function must be safe to call on its own, not just after
        # `_import_ragas()` has already happened to run first.
        _ensure_ragas_importable()

        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from langchain_community.embeddings import HuggingFaceEmbeddings

        judge_llm = LangchainLLMWrapper(_default_llm())
        judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EVAL_EMBEDDING_MODEL))
        return judge_llm, judge_embeddings
    except Exception as exc:
        logger.warning("local ragas judge/embeddings unavailable (%s); using ragas defaults", exc)
        return None, None


def _ensure_ragas_importable() -> None:
    """Working around a real (as of ragas 0.4.3 + langchain-community
    0.4.x) upstream break: ragas.llms.base unconditionally imports
    `langchain_community.chat_models.vertexai.ChatVertexAI`, which
    langchain-community dropped in its 0.4 line (VertexAI moved to the
    standalone `langchain-google-vertexai` package). We never use
    VertexAI, so a stub class satisfies the import -- the isinstance
    checks ragas does against it simply never match, which is exactly
    what we want.

    Pinning langchain-community down to a version that still has that
    module isn't a safe fix here: it drags langchain-core back to the
    0.3.x line, which the installed langchain-anthropic (1.x, requires
    modern langchain-core) doesn't support -- that would break the RAG/
    SQL/doc agents themselves, not just eval.

    Every function in this module that touches `ragas.*` calls this
    first (not just `_import_ragas`) -- any of them can be the first
    `ragas` import in a given process, and the shim must be in place
    before *any* of them, not just whichever happens to run first.
    """

    if "langchain_community.chat_models.vertexai" not in sys.modules:
        try:
            import langchain_community.chat_models.vertexai  # noqa: F401
        except ModuleNotFoundError:
            shim = types.ModuleType("langchain_community.chat_models.vertexai")

            class ChatVertexAI:  # pragma: no cover - never instantiated
                pass

            shim.ChatVertexAI = ChatVertexAI
            sys.modules["langchain_community.chat_models.vertexai"] = shim
            logger.debug("shimmed langchain_community.chat_models.vertexai for ragas import")


def _import_ragas():
    _ensure_ragas_importable()

    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    return evaluate, [faithfulness, answer_relevancy, context_precision, context_recall]


def _llm_backend_skip_reason() -> str | None:
    """`None` if the project's configured chat backend (config.yaml's
    `agents.llm_backend`) is reachable; otherwise a human-readable reason
    to skip. Callers use this as a pre-flight check before building a
    default agent, so "no local Ollama in this environment" comes back
    as a clear, honest `{"status": "skipped: ..."}` instead of either a
    confusing low-level connection traceback or -- worse -- a misleading
    0%-style "accuracy" score that looks like a real quality failure.
    """

    from src.config import load_config
    from src.llm_factory import backend_reachable

    config = load_config()
    reachable, reason = backend_reachable(config.agents.llm_backend, config.agents.ollama_base_url)
    return None if reachable else reason


def _build_rag_agent():
    """The RAGAgent under evaluation must be wired to the project's
    configured backend the same way main.py/mcp_server.py wire it --
    without `llm_backend`, `RAGAgent.__init__` leaves `self.llm = None`
    and every answer is the raw-context stub template, not a genuine
    generation, which would make any RAGAS score meaningless (it'd be
    scoring the stub, not the RAG pipeline).
    """

    from src.agents.rag_agent import RAGAgent
    from src.config import load_config
    from src.hybrid_retrieval import HybridRetriever
    from src.vectorstore import VectorStore

    config = load_config()
    vector_store = VectorStore()
    retriever = HybridRetriever(vector_store)
    return RAGAgent(
        retriever=retriever,
        llm_backend=config.agents.llm_backend,
        ollama_model=config.agents.ollama_model,
        ollama_base_url=config.agents.ollama_base_url,
    )


def _build_sql_agent(database_url: str):
    """Same backend-wiring gap as `_build_rag_agent` had: a bare
    `SQLAgent(database_url)` never passes `llm_backend`, so it falls back
    to `build_llm`'s own `LLM_BACKEND` env var / "anthropic" default
    rather than the project's actually-configured backend.
    """

    from src.agents.sql_agent import SQLAgent
    from src.config import load_config

    config = load_config()
    return SQLAgent(
        database_url,
        llm_backend=config.agents.llm_backend,
        ollama_model=config.agents.ollama_model,
        ollama_base_url=config.agents.ollama_base_url,
    )


def _build_doc_agent():
    from src.agents.doc_agent import DocAgent
    from src.config import load_config
    from src.document_processing import DocumentProcessor

    config = load_config()
    return DocAgent(
        document_processor=DocumentProcessor(),
        llm_backend=config.agents.llm_backend,
        ollama_model=config.agents.ollama_model,
        ollama_base_url=config.agents.ollama_base_url,
    )


async def _answer_all(rag_agent, testset: list[dict]) -> list:
    return [await rag_agent.answer(row["question"]) for row in testset]


def run_rag_eval(testset_path: str, output_path: str | None = None, rag_agent=None) -> dict:
    """RAG eval: run RAGAgent over `testset_path` and score with RAGAS."""

    try:
        evaluate, metrics = _import_ragas()
    except Exception as exc:
        logger.warning("ragas unavailable, skipping RAG eval: %s", exc)
        return {"status": f"skipped: ragas unavailable ({exc})"}

    from datasets import Dataset

    testset = json.loads(Path(testset_path).read_text(encoding="utf-8"))
    if not testset:
        return {"status": f"skipped: testset at {testset_path} is empty"}

    if rag_agent is None:
        skip_reason = _llm_backend_skip_reason()
        if skip_reason:
            logger.warning("skipping RAG eval: %s", skip_reason)
            return {"status": f"skipped: {skip_reason}"}
        rag_agent = _build_rag_agent()

    responses = asyncio.run(_answer_all(rag_agent, testset))

    questions, answers, contexts, ground_truths = [], [], [], []
    for row, response in zip(testset, responses):
        questions.append(row["question"])
        answers.append(response.answer)
        contexts.append(row["contexts"])
        ground_truths.append(row.get("ground_truth", ""))

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    judge_llm, judge_embeddings = _ragas_judge_llm_and_embeddings()
    # A local Ollama server generally serializes concurrent requests
    # rather than truly parallelizing them, so ragas's default
    # max_workers=16 queues most jobs behind each other -- combined with
    # the default 180s timeout, that queuing alone was enough to time
    # out roughly half the jobs in practice against a small local model.
    # A low worker count keeps each job's actual wait time close to its
    # real generation time; the longer timeout covers what's left.
    from ragas.run_config import RunConfig

    run_config = RunConfig(timeout=600, max_workers=2)
    result = evaluate(
        dataset, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings, run_config=run_config
    )
    # ragas 0.4.x's EvaluationResult has no dict-like `.items()` (that was
    # an older ragas version's interface) -- `to_pandas()` is the stable
    # public API, and its per-metric column mean (pandas skips NaN by
    # default) is the same aggregate `evaluate()` computes internally,
    # but survives individual rows that timed out instead of raising.
    scores_df = result.to_pandas()
    scores = {
        metric.name: float(scores_df[metric.name].mean())
        for metric in metrics
        if metric.name in scores_df.columns
    }

    if output_path:
        Path(output_path).write_text(json.dumps(scores, indent=2), encoding="utf-8")
    logger.info("ragas scores: %s", scores)
    return scores


# Backward-compatible alias -- main.py does
# `from eval.run_ragas_eval import run_eval as run_ragas_eval`.
run_eval = run_rag_eval


def sql_row_overlap(gen_rows: list[dict], truth_rows: list[dict]) -> float:
    """Jaccard overlap between two result sets, comparing rows positionally
    by value (not by column name, since generated and ground-truth SQL may
    alias columns differently -- e.g. `COUNT(*)` vs `COUNT(*) AS cnt`).
    Returns 1.0 for two empty result sets (vacuously equal), 0.0 if only
    one side is empty.
    """

    set_gen = {tuple(str(v) for v in row.values()) for row in gen_rows}
    set_truth = {tuple(str(v) for v in row.values()) for row in truth_rows}
    union = set_gen | set_truth
    if not union:
        return 1.0
    return len(set_gen & set_truth) / len(union)


def _execute_ground_truth(database_url: str, sql: str, max_rows: int = 1000) -> list[dict]:
    from sqlalchemy import create_engine, text

    from src.agents.sql_agent import enforce_row_limit, validate_sql

    tree = validate_sql(sql)
    safe_sql = enforce_row_limit(tree, max_rows=max_rows)
    engine = create_engine(database_url)
    with engine.connect() as conn:
        result = conn.execute(text(safe_sql))
        return [dict(row._mapping) for row in result]


async def run_sql_eval(
    testset_path: str, database_url: str | None = None, output_path: str | None = None,
    sql_agent=None, threshold: float = 0.9,
) -> dict:
    """SQL eval: for each NL question, compare the SQLAgent-generated SQL's
    result rows against the ground-truth SQL's result rows via row overlap.
    A pair "passes" when overlap >= `threshold`.
    """

    database_url = database_url or os.getenv("DATABASE_URL")
    if not database_url:
        return {"status": "skipped: DATABASE_URL is not configured"}

    testset = json.loads(Path(testset_path).read_text(encoding="utf-8"))
    if not testset:
        return {"status": f"skipped: testset at {testset_path} is empty"}

    if sql_agent is None:
        skip_reason = _llm_backend_skip_reason()
        if skip_reason:
            logger.warning("skipping SQL eval: %s", skip_reason)
            return {"status": f"skipped: {skip_reason}"}
        sql_agent = _build_sql_agent(database_url)

    correct = 0
    details = []
    for item in testset:
        question = item["question"]
        truth_sql = item["ground_truth_sql"]
        try:
            response = await sql_agent.answer(question)
            gen_sql = response.metadata["sql"]
            gen_rows = response.metadata["rows"]
            truth_rows = _execute_ground_truth(database_url, truth_sql)
            overlap = sql_row_overlap(gen_rows, truth_rows)
            is_correct = overlap >= threshold
            if is_correct:
                correct += 1
            details.append(
                {
                    "question": question,
                    "generated_sql": gen_sql,
                    "ground_truth_sql": truth_sql,
                    "overlap": overlap,
                    "correct": is_correct,
                }
            )
        except Exception as exc:
            logger.warning("sql eval question failed: %s (%s)", question, exc)
            details.append({"question": question, "ground_truth_sql": truth_sql, "error": str(exc)})

    result = {"accuracy": correct / len(testset), "correct": correct, "n": len(testset), "details": details}
    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("sql eval accuracy: %.3f (%d/%d)", result["accuracy"], correct, len(testset))
    return result


_JUDGE_PROMPT = (
    "You are grading whether an answer covers a set of required facts.\n\n"
    "Required facts (the answer must address all of these):\n{expected}\n\n"
    "Answer:\n{answer}\n\n"
    "Does the answer cover all of the required facts above? "
    "Respond with exactly one word: YES or NO."
)


async def run_doc_eval(
    testset_path: str, doc_agent=None, judge_llm=None, output_path: str | None = None
) -> dict:
    """Doc eval: for each extraction query, run DocAgent and have an LLM
    judge whether the answer covers every `expected_contains` fact. Falls
    back to a plain substring check when no judge LLM is available.
    """

    testset = json.loads(Path(testset_path).read_text(encoding="utf-8"))
    if not testset:
        return {"status": f"skipped: testset at {testset_path} is empty"}

    if doc_agent is None:
        # Unlike run_rag_eval/run_sql_eval, DocAgent has a genuinely safe
        # fallback (its raw-context stub) when no llm_backend is wired,
        # so a missing backend alone wouldn't crash this section -- but
        # it *would* silently score the stub instead of a real answer,
        # and (now that _build_doc_agent wires a real backend) an
        # unreachable one would otherwise surface as a wall of confusing
        # per-question connection errors rather than one clear skip.
        skip_reason = _llm_backend_skip_reason()
        if skip_reason:
            logger.warning("skipping doc eval: %s", skip_reason)
            return {"status": f"skipped: {skip_reason}"}
        doc_agent = _build_doc_agent()

    if judge_llm is None:
        try:
            judge_llm = _default_llm()
        except Exception as exc:  # pragma: no cover - missing API key / package
            logger.warning("no judge LLM available (%s); using substring fallback", exc)
            judge_llm = None

    correct = 0
    details = []
    for item in testset:
        question = item["question"]
        file_path = item.get("file_path")
        expected = item.get("expected_contains", [])
        try:
            response = doc_agent.answer_from_file(file_path, question)
            answer = response.answer

            if judge_llm is not None and expected:
                prompt = _JUDGE_PROMPT.format(expected="\n".join(f"- {e}" for e in expected), answer=answer)
                verdict = _extract_text(await judge_llm.ainvoke(prompt))
                is_correct = "YES" in verdict.upper()
            else:
                is_correct = all(e.lower() in answer.lower() for e in expected) if expected else bool(answer)

            if is_correct:
                correct += 1
            details.append({"question": question, "file_path": file_path, "correct": is_correct})
        except Exception as exc:
            logger.warning("doc eval question failed: %s (%s)", question, exc)
            details.append({"question": question, "file_path": file_path, "error": str(exc)})

    result = {"accuracy": correct / len(testset), "correct": correct, "n": len(testset), "details": details}
    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("doc eval accuracy: %.3f (%d/%d)", result["accuracy"], correct, len(testset))
    return result


def run_retrieval_eval(testset_path: str, retriever=None, k: int = 10) -> dict:
    """Retrieval precision/recall eval: for each testset row with a
    `relevant_chunk_ids` ground-truth list, retrieve `k` chunks and score
    the overlap. Skips (rather than fails) when the testset has no rows
    carrying that field, since eval/generate_testset.py doesn't currently
    produce it -- retrieval ground truth has to be supplied separately.
    """

    from eval.metrics import aggregate_retrieval_metrics

    testset = json.loads(Path(testset_path).read_text(encoding="utf-8"))
    rows = [row for row in testset if row.get("relevant_chunk_ids")]
    if not rows:
        return {"status": f"skipped: no rows in {testset_path} carry relevant_chunk_ids"}

    if retriever is None:
        from src.hybrid_retrieval import HybridRetriever
        from src.vectorstore import VectorStore

        retriever = HybridRetriever(VectorStore())

    pairs = []
    for row in rows:
        retrieved = asyncio.run(retriever.retrieve([row["question"]], top_k=k))
        retrieved_ids = [r.chunk.chunk_id for r in retrieved]
        pairs.append((retrieved_ids, row["relevant_chunk_ids"]))

    result = aggregate_retrieval_metrics(pairs)
    logger.info("retrieval eval: precision=%.3f recall=%.3f (n=%d)", result["precision"], result["recall"], result["n"])
    return result


def run_sql_guardrail_eval(cases=None) -> dict:
    """SQL AST guardrail pass-rate eval: reported alongside the other
    sections in results_v3.json (see eval/metrics.py for the case set).
    """

    from eval.metrics import sql_ast_guardrail_pass_rate

    result = sql_ast_guardrail_pass_rate(cases)
    logger.info("sql guardrail pass rate: %.3f (%d/%d)", result["pass_rate"], result["correct"], result["n"])
    return result


def run_all(
    rag_testset: str = "eval/testset.json",
    sql_testset: str = "eval/sql_testset.json",
    doc_testset: str = "eval/doc_testset.json",
    output_path: str = "eval/results_v3.json",
) -> dict:
    results = {
        "rag": run_rag_eval(rag_testset),
        "sql": asyncio.run(run_sql_eval(sql_testset)),
        "doc": asyncio.run(run_doc_eval(doc_testset)),
        "retrieval": run_retrieval_eval(rag_testset),
        "sql_guardrail": run_sql_guardrail_eval(),
    }
    Path(output_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("wrote combined results to %s", output_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG/SQL/doc evaluation")
    parser.add_argument("--type", choices=["all", "rag", "sql", "doc"], default="all")
    parser.add_argument("--rag-testset", default="eval/testset.json")
    parser.add_argument("--sql-testset", default="eval/sql_testset.json")
    parser.add_argument("--doc-testset", default="eval/doc_testset.json")
    parser.add_argument("--output", default="eval/results_v3.json")
    args = parser.parse_args()

    if args.type == "all":
        run_all(args.rag_testset, args.sql_testset, args.doc_testset, args.output)
        return

    if args.type == "rag":
        result = run_rag_eval(args.rag_testset)
    elif args.type == "sql":
        result = asyncio.run(run_sql_eval(args.sql_testset))
    else:
        result = asyncio.run(run_doc_eval(args.doc_testset))

    Path(args.output).write_text(json.dumps({args.type: result}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    # Running this file directly (`python eval/run_ragas_eval.py`, the
    # exact invocation this project's README and CI workflow document)
    # puts this file's own directory -- not the project root -- at
    # sys.path[0]. run_retrieval_eval/run_sql_guardrail_eval below do
    # `from eval.metrics import ...`, which needs the project root (the
    # parent of this eval/ directory) importable as a package root;
    # without it those two sections crash with `ModuleNotFoundError: No
    # module named 'eval'` after rag/sql/doc eval have already run.
    # `python -m eval.run_ragas_eval` and importing this module normally
    # (main.py's CLI, pytest) are unaffected -- the project root is
    # already on sys.path in both cases.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
