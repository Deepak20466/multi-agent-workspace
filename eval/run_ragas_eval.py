"""Runs the RAGAgent over eval/testset.json and scores it with RAGAS
(faithfulness, answer_relevancy, context_precision), writing results
to eval/results.json for check_gates.py to consume in CI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from src.agents.rag_agent import RAGAgent
from src.hybrid_retrieval import HybridRetriever
from src.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_ragas_eval")


def _build_rag_agent() -> RAGAgent:
    vector_store = VectorStore()
    retriever = HybridRetriever(vector_store)
    return RAGAgent(retriever=retriever)


async def _answer_all(rag_agent: RAGAgent, testset: list[dict]) -> list:
    return [await rag_agent.answer(row["question"]) for row in testset]


def run_eval(testset_path: str, output_path: str) -> dict:
    testset = json.loads(Path(testset_path).read_text(encoding="utf-8"))
    if not testset:
        raise ValueError(f"testset at {testset_path} is empty; run generate_testset.py first")

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

    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
    scores = {k: float(v) for k, v in result.items()}

    Path(output_path).write_text(json.dumps(scores, indent=2), encoding="utf-8")
    logger.info("ragas scores: %s", scores)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation over the RAG agent")
    parser.add_argument("--testset", default="eval/testset.json")
    parser.add_argument("--output", default="eval/results.json")
    args = parser.parse_args()

    run_eval(args.testset, args.output)


if __name__ == "__main__":
    main()
