"""Generates a RAGAS-compatible testset (question / ground_truth /
contexts) from the sample documents, so `run_ragas_eval.py` has
something to score against without hand-writing fixtures.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.document_processing import DocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_testset")

DEFAULT_QUESTIONS = [
    "What is the main topic of this document?",
    "Summarize the key figures mentioned in the document.",
    "What conclusions does the document draw?",
]


def generate_testset(sample_dir: str, output_path: str, questions: list[str] | None = None) -> list[dict]:
    questions = questions or DEFAULT_QUESTIONS
    sample_dir_path = Path(sample_dir)
    testset: list[dict] = []
    processor = DocumentProcessor()

    for file_path in sorted(sample_dir_path.glob("*")):
        if file_path.is_dir():
            continue
        try:
            _, chunks = processor.process(file_path)
        except ValueError as exc:
            logger.warning("skipping unsupported file %s: %s", file_path, exc)
            continue

        if not chunks:
            continue

        contexts = [c.text for c in chunks[:3]]
        for question in questions:
            testset.append(
                {
                    "question": question,
                    "contexts": contexts,
                    "ground_truth": "",  # to be filled in by a human reviewer or an LLM judge
                    "source_file": str(file_path),
                }
            )

    Path(output_path).write_text(json.dumps(testset, indent=2), encoding="utf-8")
    logger.info("wrote %d testset rows to %s", len(testset), output_path)
    return testset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a RAGAS testset from sample documents")
    parser.add_argument("--sample-dir", default="data/sample_documents")
    parser.add_argument("--output", default="eval/testset.json")
    args = parser.parse_args()

    generate_testset(args.sample_dir, args.output)


if __name__ == "__main__":
    main()
