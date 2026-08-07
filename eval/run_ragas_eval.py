"""Runs all three specialist agents against their generated testsets and
scores them, writing combined results to eval/results_v3.json for
check_gates.py to consume in CI:

- RAG: RAGAS metrics (faithfulness, answer_relevancy, context_precision,
  context_recall) over eval/testset.json.
- SQL: row-overlap accuracy between generated and ground-truth SQL over
  eval/sql_testset.json.
- Doc: LLM-judge accuracy over eval/doc_testset.json.

Each section degrades to `{"status": "skipped: <reason>"}` instead of
raising when its prerequisites aren't met (no DATABASE_URL, no testset,
ragas itself failing to import) -- see `_import_ragas` below -- so a
partial CI environment still produces a usable results file rather than
crashing the whole run.
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


def _extract_text(response: object) -> str:
    content = getattr(response, "content", None)
    return str(content) if content is not None else str(response)


def _default_llm():
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=JUDGE_MODEL, temperature=0)


def _import_ragas():
    """Import ragas, working around a real (as of ragas 0.4.3 +
    langchain-community 0.4.x) upstream break: ragas.llms.base
    unconditionally imports `langchain_community.chat_models.vertexai
    .ChatVertexAI`, which langchain-community dropped in its 0.4 line
    (VertexAI moved to the standalone `langchain-google-vertexai`
    package). We never use VertexAI, so a stub class satisfies the
    import -- the isinstance checks ragas does against it simply never
    match, which is exactly what we want.

    Pinning langchain-community down to a version that still has that
    module isn't a safe fix here: it drags langchain-core back to the
    0.3.x line, which the installed langchain-anthropic (1.x, requires
    modern langchain-core) doesn't support -- that would break the RAG/
    SQL/doc agents themselves, not just eval.
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

    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    return evaluate, [faithfulness, answer_relevancy, context_precision, context_recall]


def _build_rag_agent():
    from src.agents.rag_agent import RAGAgent
    from src.hybrid_retrieval import HybridRetriever
    from src.vectorstore import VectorStore

    vector_store = VectorStore()
    retriever = HybridRetriever(vector_store)
    return RAGAgent(retriever=retriever)


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

    rag_agent = rag_agent or _build_rag_agent()
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

    result = evaluate(dataset, metrics=metrics)
    scores = {k: float(v) for k, v in result.items()}

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

    from src.agents.sql_agent import SQLAgent

    sql_agent = sql_agent or SQLAgent(database_url)

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

    from src.agents.doc_agent import DocAgent

    doc_agent = doc_agent or DocAgent()

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
    main()
