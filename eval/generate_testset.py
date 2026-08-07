"""Generates RAGAS-compatible testsets for all three specialist agents:

- `generate_testset` (RAG): question / ground_truth / contexts, mined from
  the sample documents, so `run_ragas_eval.py` has something to score
  against without hand-writing fixtures.
- `generate_sql_testset` (SQL): NL question / ground_truth_sql pairs,
  generated from the live database schema.
- `generate_doc_testset` (Doc): extraction queries / expected_contains
  substrings, generated from the sample documents.

SQL and doc testset generation use an LLM (Claude Haiku by default) when
one is available, and fall back to small deterministic testsets otherwise
so `--type all` still produces something runnable without ANTHROPIC_API_KEY
configured (mirrors the DEFAULT_QUESTIONS fallback already used for RAG).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

from src.document_processing import DocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_testset")

TESTSET_MODEL = os.getenv("EVAL_TESTSET_MODEL", "claude-haiku-4-5")

DEFAULT_QUESTIONS = [
    "What is the main topic of this document?",
    "Summarize the key figures mentioned in the document.",
    "What conclusions does the document draw?",
]

# Deterministic fallback SQL testset, matching the seed schema created by
# eval/seed_sql_db.py (sales(id, region, product, amount, sale_date)) —
# used when no LLM is configured.
DEFAULT_SQL_TESTSET = [
    {"question": "How many sales are there?", "ground_truth_sql": "SELECT COUNT(*) FROM sales"},
    {
        "question": "What is the total amount of sales in the East region?",
        "ground_truth_sql": "SELECT SUM(amount) FROM sales WHERE region = 'East'",
    },
    {
        "question": "How many sales does each region have?",
        "ground_truth_sql": "SELECT region, COUNT(*) FROM sales GROUP BY region",
    },
]

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _extract_text(response: object) -> str:
    content = getattr(response, "content", None)
    return str(content) if content is not None else str(response)


def _parse_json_array(raw_text: str) -> list[dict]:
    match = _JSON_ARRAY_RE.search(raw_text)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("testset LLM response was not valid JSON")
        return []
    return data if isinstance(data, list) else []


def _default_llm():
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=TESTSET_MODEL, temperature=0)


def generate_testset(
    sample_dir: str, output_path: str, questions: list[str] | None = None, llm=None
) -> list[dict]:
    """RAG testset: question/contexts pairs mined directly from chunks of
    each sample document. `ground_truth` is left blank unless `llm` is
    given, in which case it's generated from the contexts so that
    context_recall (which needs a ground_truth to compare against) is
    meaningful rather than trivially near-zero.
    """

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
            ground_truth = ""
            if llm is not None:
                try:
                    prompt = (
                        "Answer the question in 1-2 sentences using only this context:\n"
                        + "\n---\n".join(contexts)
                        + f"\n\nQuestion: {question}"
                    )
                    ground_truth = _extract_text(llm.invoke(prompt)).strip()
                except Exception as exc:
                    logger.warning("ground_truth generation failed for %r: %s", question, exc)

            testset.append(
                {
                    "question": question,
                    "contexts": contexts,
                    "ground_truth": ground_truth,  # blank unless `llm` was given
                    "source_file": str(file_path),
                }
            )

    Path(output_path).write_text(json.dumps(testset, indent=2), encoding="utf-8")
    logger.info("wrote %d testset rows to %s", len(testset), output_path)
    return testset


def generate_sql_testset(
    database_url: str | None, output_path: str, llm=None, n_pairs: int = 10
) -> list[dict]:
    """SQL testset: NL question / ground_truth_sql pairs generated from the
    live database schema. Skips (writes an empty list) when no database is
    configured; falls back to DEFAULT_SQL_TESTSET when no LLM is available.
    """

    if not database_url:
        logger.warning("DATABASE_URL not configured; writing empty SQL testset")
        Path(output_path).write_text(json.dumps([], indent=2), encoding="utf-8")
        return []

    from src.agents.sql_agent import SQLAgent

    agent = SQLAgent(database_url)
    schema = agent.get_schema()

    if llm is None:
        try:
            llm = _default_llm()
        except Exception as exc:  # pragma: no cover - missing API key / package
            logger.warning("no LLM available for SQL testset generation (%s); using fallback", exc)
            Path(output_path).write_text(json.dumps(DEFAULT_SQL_TESTSET, indent=2), encoding="utf-8")
            return DEFAULT_SQL_TESTSET

    prompt = (
        f"Given this PostgreSQL schema:\n{schema}\n\n"
        f"Generate {n_pairs} natural-language question -> SQL pairs as a JSON list "
        'of objects: [{"question": "...", "ground_truth_sql": "..."}]. '
        "Only write read-only SELECT statements. Output ONLY the JSON list, no commentary."
    )

    try:
        response = llm.invoke(prompt)
        testset = _parse_json_array(_extract_text(response))
    except Exception as exc:
        logger.warning("SQL testset generation failed (%s); using fallback", exc)
        testset = []

    if not testset:
        testset = DEFAULT_SQL_TESTSET

    Path(output_path).write_text(json.dumps(testset, indent=2), encoding="utf-8")
    logger.info("wrote %d SQL testset rows to %s", len(testset), output_path)
    return testset


def generate_doc_testset(sample_dir: str, output_path: str, llm=None, n_items: int = 10) -> list[dict]:
    """Doc testset: extraction queries with expected_contains substrings,
    one batch generated per sample document. Falls back to a single
    "summarize this document" style query per file when no LLM is
    available, using the first chunk's text as the expected substring.
    """

    sample_dir_path = Path(sample_dir)
    processor = DocumentProcessor()

    if llm is None:
        try:
            llm = _default_llm()
        except Exception as exc:  # pragma: no cover - missing API key / package
            logger.warning("no LLM available for doc testset generation (%s); using fallback", exc)
            llm = None

    def _fallback_item(file_path: Path, first_chunk_text: str) -> dict:
        snippet = first_chunk_text[:80].strip()
        return {
            "question": "What does this document say?",
            "file_path": str(file_path),
            "expected_contains": [snippet] if snippet else [],
        }

    testset: list[dict] = []
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

        if llm is None:
            testset.append(_fallback_item(file_path, chunks[0].text))
            continue

        excerpt = "\n".join(c.text for c in chunks[:5])[:4000]
        prompt = (
            f"From this document excerpt:\n{excerpt}\n\n"
            f"Generate up to {n_items} document extraction queries as a JSON list of objects: "
            '[{"question": "...", "expected_contains": ["substring1", "substring2"]}]. '
            "expected_contains should be short substrings that a correct answer must mention. "
            "Output ONLY the JSON list, no commentary."
        )
        try:
            response = llm.invoke(prompt)
            items = _parse_json_array(_extract_text(response))
        except Exception as exc:
            logger.warning("doc testset generation failed for %s (%s); using fallback", file_path, exc)
            items = []

        if not items:
            testset.append(_fallback_item(file_path, chunks[0].text))
            continue

        for item in items:
            if isinstance(item, dict) and item.get("question"):
                item.setdefault("file_path", str(file_path))
                testset.append(item)

    Path(output_path).write_text(json.dumps(testset, indent=2), encoding="utf-8")
    logger.info("wrote %d doc testset rows to %s", len(testset), output_path)
    return testset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAGAS-compatible testsets")
    parser.add_argument("--type", choices=["all", "rag", "sql", "doc"], default="all")
    parser.add_argument("--sample-dir", default="data/sample_documents")
    parser.add_argument("--output-dir", default="eval")
    args = parser.parse_args()

    try:
        llm = _default_llm()
    except Exception as exc:  # pragma: no cover - missing API key / package
        logger.warning("no LLM available (%s); testsets will use deterministic fallbacks", exc)
        llm = None

    types = ["rag", "sql", "doc"] if args.type == "all" else [args.type]
    if "rag" in types:
        generate_testset(args.sample_dir, str(Path(args.output_dir) / "testset.json"), llm=llm)
    if "sql" in types:
        generate_sql_testset(
            os.getenv("DATABASE_URL"), str(Path(args.output_dir) / "sql_testset.json"), llm=llm
        )
    if "doc" in types:
        generate_doc_testset(args.sample_dir, str(Path(args.output_dir) / "doc_testset.json"), llm=llm)


if __name__ == "__main__":
    main()
