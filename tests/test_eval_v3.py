"""Tests for the v3.1 eval pipeline: SQL row-overlap scoring, gate
checking against the nested results_v3.json shape, and the SQL/doc
testset generators' deterministic fallbacks when no LLM is available.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from eval.check_gates import check_gates
from eval.generate_testset import generate_doc_testset, generate_sql_testset
from eval.run_ragas_eval import run_doc_eval, run_sql_eval, sql_row_overlap
from eval.seed_sql_db import seed
from src.agents.doc_agent import DocAgent
from src.agents.sql_agent import SQLAgent

# ---------------------------------------------------------------------------
# sql_row_overlap
# ---------------------------------------------------------------------------


def test_sql_row_overlap_identical_rows_by_value_ignores_column_names():
    gen = [{"count": 3}]
    truth = [{"cnt": 3}]
    assert sql_row_overlap(gen, truth) == 1.0


def test_sql_row_overlap_disjoint_rows():
    assert sql_row_overlap([{"a": 1}], [{"a": 2}]) == 0.0


def test_sql_row_overlap_both_empty_is_vacuously_equal():
    assert sql_row_overlap([], []) == 1.0


def test_sql_row_overlap_one_empty_is_zero():
    assert sql_row_overlap([], [{"a": 1}]) == 0.0


def test_sql_row_overlap_partial():
    gen = [{"a": 1}, {"a": 2}]
    truth = [{"a": 1}, {"a": 3}]
    assert sql_row_overlap(gen, truth) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# check_gates
# ---------------------------------------------------------------------------


def test_check_gates_passes_when_all_metrics_meet_threshold(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "rag": {
                    "faithfulness": 0.9,
                    "answer_relevancy": 0.9,
                    "context_precision": 0.9,
                    "context_recall": 0.9,
                },
                "sql": {"accuracy": 0.9},
                "doc": {"accuracy": 0.9},
            }
        )
    )
    assert check_gates(str(results)) is True


def test_check_gates_fails_on_low_metric(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "rag": {
                    "faithfulness": 0.1,
                    "answer_relevancy": 0.9,
                    "context_precision": 0.9,
                    "context_recall": 0.9,
                },
                "sql": {"accuracy": 0.9},
                "doc": {"accuracy": 0.9},
            }
        )
    )
    assert check_gates(str(results)) is False


def test_check_gates_skips_unconfigured_section_without_failing(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "rag": {
                    "faithfulness": 0.9,
                    "answer_relevancy": 0.9,
                    "context_precision": 0.9,
                    "context_recall": 0.9,
                },
                "sql": {"status": "skipped: DATABASE_URL is not configured"},
                "doc": {"accuracy": 0.9},
            }
        )
    )
    assert check_gates(str(results)) is True


def test_check_gates_supports_legacy_flat_rag_only_results(tmp_path):
    results = tmp_path / "legacy.json"
    results.write_text(
        json.dumps({"faithfulness": 0.9, "answer_relevancy": 0.9, "context_precision": 0.9})
    )
    gates = {
        "rag.faithfulness": 0.75,
        "rag.answer_relevancy": 0.70,
        "rag.context_precision": 0.70,
    }
    assert check_gates(str(results), gates=gates) is True


def test_check_gates_missing_metric_fails(tmp_path):
    results = tmp_path / "empty.json"
    results.write_text(json.dumps({}))
    assert check_gates(str(results), gates={"sql.accuracy": 0.8}) is False


# ---------------------------------------------------------------------------
# generate_sql_testset
# ---------------------------------------------------------------------------


def test_generate_sql_testset_skips_without_database_url(tmp_path):
    output = tmp_path / "sql_testset.json"
    testset = generate_sql_testset(None, str(output))
    assert testset == []
    assert json.loads(output.read_text()) == []


def test_generate_sql_testset_falls_back_without_llm(tmp_path):
    db_path = tmp_path / "seed.db"
    db_url = f"sqlite:///{db_path}"
    seed(db_url)

    output = tmp_path / "sql_testset.json"
    testset = generate_sql_testset(db_url, str(output), llm=None)

    assert len(testset) > 0
    assert all("question" in row and "ground_truth_sql" in row for row in testset)


def test_generate_sql_testset_falls_back_when_llm_invoke_fails(tmp_path):
    db_path = tmp_path / "seed.db"
    db_url = f"sqlite:///{db_path}"
    seed(db_url)

    failing_llm = MagicMock()
    failing_llm.invoke.side_effect = RuntimeError("no credentials")

    output = tmp_path / "sql_testset.json"
    testset = generate_sql_testset(db_url, str(output), llm=failing_llm)

    assert len(testset) > 0  # falls back to DEFAULT_SQL_TESTSET, not an empty list


# ---------------------------------------------------------------------------
# generate_doc_testset
# ---------------------------------------------------------------------------


def test_generate_doc_testset_falls_back_per_document_without_llm(document_processor, tmp_path):
    sample_dir = tmp_path / "docs"
    sample_dir.mkdir()
    (sample_dir / "note.txt").write_text("The quarterly revenue was $5M in Q3.")

    output = tmp_path / "doc_testset.json"
    testset = generate_doc_testset(str(sample_dir), str(output), llm=None)

    assert len(testset) == 1
    assert testset[0]["file_path"].endswith("note.txt")
    assert testset[0]["expected_contains"]


def test_generate_doc_testset_falls_back_when_llm_invoke_fails(document_processor, tmp_path):
    sample_dir = tmp_path / "docs"
    sample_dir.mkdir()
    (sample_dir / "note.txt").write_text("The quarterly revenue was $5M in Q3.")

    failing_llm = MagicMock()
    failing_llm.invoke.side_effect = RuntimeError("no credentials")

    output = tmp_path / "doc_testset.json"
    testset = generate_doc_testset(str(sample_dir), str(output), llm=failing_llm)

    # Regression test: an LLM that fails at invoke-time (not construction
    # time) must still produce a fallback item, not silently drop the file.
    assert len(testset) == 1
    assert testset[0]["file_path"].endswith("note.txt")


# ---------------------------------------------------------------------------
# run_sql_eval
# ---------------------------------------------------------------------------


async def test_run_sql_eval_skips_without_database_url(tmp_path, monkeypatch):
    # run_sql_eval falls back to os.getenv("DATABASE_URL") when no
    # database_url is passed explicitly (matches the CLI's usage) -- other
    # tests importing main.py can leak a real DATABASE_URL from .env via
    # load_dotenv(), so isolate explicitly rather than relying on ambient
    # absence.
    monkeypatch.delenv("DATABASE_URL", raising=False)

    testset_path = tmp_path / "sql_testset.json"
    testset_path.write_text(json.dumps([{"question": "q", "ground_truth_sql": "SELECT 1"}]))

    result = await run_sql_eval(str(testset_path), database_url=None)
    assert result["status"].startswith("skipped")


async def test_run_sql_eval_scores_matching_sql_as_correct(tmp_path):
    db_path = tmp_path / "seed.db"
    db_url = f"sqlite:///{db_path}"
    seed(db_url)

    testset_path = tmp_path / "sql_testset.json"
    testset_path.write_text(
        json.dumps([{"question": "How many users are there?", "ground_truth_sql": "SELECT COUNT(*) FROM users"}])
    )

    stub_llm = MagicMock()
    stub_llm.ainvoke = AsyncMock(return_value="SELECT COUNT(*) FROM users")
    agent = SQLAgent(db_url, llm=stub_llm)

    result = await run_sql_eval(str(testset_path), db_url, sql_agent=agent)

    assert result["accuracy"] == 1.0
    assert result["n"] == 1


async def test_run_sql_eval_records_errors_without_raising(tmp_path):
    db_path = tmp_path / "seed.db"
    db_url = f"sqlite:///{db_path}"
    seed(db_url)

    testset_path = tmp_path / "sql_testset.json"
    testset_path.write_text(
        json.dumps([{"question": "drop everything", "ground_truth_sql": "SELECT COUNT(*) FROM users"}])
    )

    stub_llm = MagicMock()
    stub_llm.ainvoke = AsyncMock(return_value="DROP TABLE users")  # rejected by validate_sql
    agent = SQLAgent(db_url, llm=stub_llm)

    result = await run_sql_eval(str(testset_path), db_url, sql_agent=agent)

    assert result["accuracy"] == 0.0
    assert "error" in result["details"][0]


# ---------------------------------------------------------------------------
# run_doc_eval
# ---------------------------------------------------------------------------


async def test_run_doc_eval_uses_judge_llm_verdict(document_processor, tmp_path):
    sample_file = tmp_path / "policy.txt"
    sample_file.write_text("Refunds are issued within 30 days of purchase.")

    testset_path = tmp_path / "doc_testset.json"
    testset_path.write_text(
        json.dumps(
            [{"question": "What is the refund window?", "file_path": str(sample_file), "expected_contains": ["30 days"]}]
        )
    )

    doc_agent = DocAgent(document_processor=document_processor)
    judge = MagicMock()
    judge.ainvoke = AsyncMock(return_value="YES")

    result = await run_doc_eval(str(testset_path), doc_agent=doc_agent, judge_llm=judge)

    assert result["accuracy"] == 1.0
    judge.ainvoke.assert_awaited_once()


async def test_run_doc_eval_falls_back_to_substring_match_without_judge(document_processor, tmp_path):
    sample_file = tmp_path / "policy.txt"
    sample_file.write_text("Refunds are issued within 30 days of purchase.")

    testset_path = tmp_path / "doc_testset.json"
    testset_path.write_text(
        json.dumps(
            [{"question": "What is the refund window?", "file_path": str(sample_file), "expected_contains": ["nonexistent phrase"]}]
        )
    )

    doc_agent = DocAgent(document_processor=document_processor)
    result = await run_doc_eval(str(testset_path), doc_agent=doc_agent, judge_llm=None)

    assert result["accuracy"] == 0.0


async def test_run_doc_eval_skips_on_empty_testset(tmp_path):
    testset_path = tmp_path / "doc_testset.json"
    testset_path.write_text(json.dumps([]))

    result = await run_doc_eval(str(testset_path))
    assert result["status"].startswith("skipped")
