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


def _patch_chroma(monkeypatch) -> None:
    import chromadb
    from chromadb.utils import embedding_functions

    monkeypatch.setattr(chromadb, "PersistentClient", MagicMock())
    monkeypatch.setattr(embedding_functions, "SentenceTransformerEmbeddingFunction", MagicMock())

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


def test_generate_doc_testset_falls_back_per_document_without_llm(document_processor, tmp_path, monkeypatch):
    """`llm=None` here means "no LLM was passed", not "no LLM is
    reachable" -- `generate_doc_testset` still tries its own
    `_default_llm()` internally. Force that to fail explicitly so this
    test deterministically exercises the fallback path regardless of
    whether this machine happens to have a local Ollama (or Anthropic
    credentials) available -- `_default_llm()` now resolves the
    project's actually-configured backend (Ollama per config.yaml) via
    `build_llm`, so it would otherwise silently succeed in this dev
    environment and take the LLM-generation branch instead.
    """

    import eval.generate_testset as generate_testset_module

    monkeypatch.setattr(
        generate_testset_module, "_default_llm", MagicMock(side_effect=RuntimeError("no llm configured"))
    )

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
        json.dumps([{"question": "How many sales are there?", "ground_truth_sql": "SELECT COUNT(*) FROM sales"}])
    )

    stub_llm = MagicMock()
    stub_llm.ainvoke = AsyncMock(return_value="SELECT COUNT(*) FROM sales")
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
        json.dumps([{"question": "drop everything", "ground_truth_sql": "SELECT COUNT(*) FROM sales"}])
    )

    stub_llm = MagicMock()
    stub_llm.ainvoke = AsyncMock(return_value="DROP TABLE sales")  # rejected by validate_sql
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


async def test_run_doc_eval_falls_back_to_substring_match_without_judge(document_processor, tmp_path, monkeypatch):
    """`judge_llm=None` means "no judge was passed", not "no judge is
    reachable" -- `run_doc_eval` still tries its own `_default_llm()`
    internally. Force that to fail explicitly so this test
    deterministically exercises the substring-fallback path rather than
    (as it would silently do in this dev environment, where
    `_default_llm()` now resolves the project's configured Ollama
    backend via `build_llm`) actually invoking a real LLM judge and
    passing for the wrong reason.
    """

    import eval.run_ragas_eval as run_ragas_eval_module

    monkeypatch.setattr(
        run_ragas_eval_module, "_default_llm", MagicMock(side_effect=RuntimeError("no llm configured"))
    )

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


# ---------------------------------------------------------------------------
# RAG eval: RAGAgent + RAGAS wired to the project's configured (local
# Ollama by default) backend instead of silently requiring OPENAI_API_KEY
# ---------------------------------------------------------------------------


def test_build_rag_agent_wires_configured_backend(monkeypatch):
    """`_build_rag_agent()` previously constructed `RAGAgent(retriever=
    retriever)` with no backend at all, leaving `self.llm = None` --
    every "evaluated" answer would have been the raw-context stub
    template, not a genuine generation, making any RAGAS score
    meaningless. It must now build a real LLM from config.yaml's
    `agents.llm_backend` (Ollama by default in this project).
    """

    from eval.run_ragas_eval import _build_rag_agent
    from src.config import load_config

    _patch_chroma(monkeypatch)
    config = load_config()

    agent = _build_rag_agent()

    assert agent.llm is not None
    if config.agents.llm_backend == "ollama":
        assert agent.llm.model == config.agents.ollama_model


def test_ragas_judge_llm_and_embeddings_wraps_configured_backend(ollama_available):
    """RAGAS's `evaluate()` silently defaults to an OpenAI-backed judge/
    embeddings model when `llm`/`embeddings` aren't passed explicitly --
    this is what made RAGAS eval require OPENAI_API_KEY even though the
    project is configured to run fully locally with Ollama. Skipped (not
    failed) when Ollama isn't reachable, mirroring the other real-Ollama
    integration tests in this suite.
    """

    if not ollama_available:
        pytest.skip("Ollama not reachable at localhost:11434")

    from eval.run_ragas_eval import _ragas_judge_llm_and_embeddings

    judge_llm, judge_embeddings = _ragas_judge_llm_and_embeddings()

    assert judge_llm is not None
    assert judge_embeddings is not None
    assert type(judge_llm).__name__ == "LangchainLLMWrapper"
    assert type(judge_embeddings).__name__ == "LangchainEmbeddingsWrapper"


def test_ragas_judge_llm_and_embeddings_degrades_to_none_on_failure(monkeypatch):
    """A broken local setup (e.g. the embedding model can't be loaded)
    must degrade to ragas's own defaults instead of blocking the whole
    eval run -- `evaluate()` accepts `llm=None`/`embeddings=None`.
    """

    import eval.run_ragas_eval as run_ragas_eval_module

    monkeypatch.setattr(
        run_ragas_eval_module, "_default_llm", MagicMock(side_effect=RuntimeError("backend unavailable"))
    )

    judge_llm, judge_embeddings = run_ragas_eval_module._ragas_judge_llm_and_embeddings()

    assert judge_llm is None
    assert judge_embeddings is None


# ---------------------------------------------------------------------------
# Graceful skip when the configured LLM backend is unreachable (e.g. no local
# Ollama server in CI) -- must be a clean, honest "skipped", never fabricated
# scores and never silently confused with a genuine application failure.
# ---------------------------------------------------------------------------


def test_run_rag_eval_skips_when_llm_backend_unreachable(tmp_path, monkeypatch):
    import eval.run_ragas_eval as run_ragas_eval_module

    monkeypatch.setattr(
        run_ragas_eval_module, "_llm_backend_skip_reason", lambda: "Ollama backend configured but unreachable"
    )
    build_agent_spy = MagicMock(side_effect=AssertionError("should not build an agent when skipping"))
    monkeypatch.setattr(run_ragas_eval_module, "_build_rag_agent", build_agent_spy)

    testset_path = tmp_path / "testset.json"
    testset_path.write_text(
        json.dumps([{"question": "q", "contexts": ["c"], "ground_truth": "gt"}]), encoding="utf-8"
    )

    result = run_ragas_eval_module.run_rag_eval(str(testset_path))

    assert result["status"].startswith("skipped:")
    assert "unreachable" in result["status"]
    build_agent_spy.assert_not_called()


def test_run_rag_eval_does_not_swallow_genuine_answer_failures(tmp_path):
    """A real application error (not a reachability problem) must
    propagate, not be silently reported as "skipped" -- CI needs to be
    able to tell "no local LLM available" apart from "the RAG pipeline
    is actually broken". Passing `rag_agent` explicitly bypasses the
    backend pre-flight check entirely (same as any other test-injected
    agent), so this exercises the real, unmocked exception path.
    """

    import eval.run_ragas_eval as run_ragas_eval_module

    failing_agent = MagicMock()
    failing_agent.answer = AsyncMock(side_effect=RuntimeError("unexpected application bug"))

    testset_path = tmp_path / "testset.json"
    testset_path.write_text(
        json.dumps([{"question": "q", "contexts": ["c"], "ground_truth": "gt"}]), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="unexpected application bug"):
        run_ragas_eval_module.run_rag_eval(str(testset_path), rag_agent=failing_agent)


async def test_run_sql_eval_skips_when_llm_backend_unreachable(tmp_path, monkeypatch):
    import eval.run_ragas_eval as run_ragas_eval_module

    monkeypatch.setattr(
        run_ragas_eval_module, "_llm_backend_skip_reason", lambda: "Ollama backend configured but unreachable"
    )
    build_agent_spy = MagicMock(side_effect=AssertionError("should not build an agent when skipping"))
    monkeypatch.setattr(run_ragas_eval_module, "_build_sql_agent", build_agent_spy)

    testset_path = tmp_path / "sql_testset.json"
    testset_path.write_text(
        json.dumps([{"question": "how many?", "ground_truth_sql": "SELECT 1"}]), encoding="utf-8"
    )

    result = await run_ragas_eval_module.run_sql_eval(str(testset_path), database_url="sqlite:///:memory:")

    assert result["status"].startswith("skipped:")
    build_agent_spy.assert_not_called()


async def test_run_doc_eval_skips_when_llm_backend_unreachable(tmp_path, monkeypatch):
    import eval.run_ragas_eval as run_ragas_eval_module

    monkeypatch.setattr(
        run_ragas_eval_module, "_llm_backend_skip_reason", lambda: "Ollama backend configured but unreachable"
    )
    build_agent_spy = MagicMock(side_effect=AssertionError("should not build an agent when skipping"))
    monkeypatch.setattr(run_ragas_eval_module, "_build_doc_agent", build_agent_spy)

    testset_path = tmp_path / "doc_testset.json"
    testset_path.write_text(
        json.dumps([{"question": "what does it say?", "file_path": "irrelevant.txt", "expected_contains": ["x"]}]),
        encoding="utf-8",
    )

    result = await run_ragas_eval_module.run_doc_eval(str(testset_path))

    assert result["status"].startswith("skipped:")
    build_agent_spy.assert_not_called()


def test_build_sql_agent_wires_configured_backend(tmp_path):
    from eval.run_ragas_eval import _build_sql_agent

    db_path = tmp_path / "seed.db"
    db_url = f"sqlite:///{db_path}"
    seed(db_url)

    from src.config import load_config

    config = load_config()
    agent = _build_sql_agent(db_url)

    assert agent.llm_backend == config.agents.llm_backend
    assert agent.ollama_model == config.agents.ollama_model


def test_build_doc_agent_wires_configured_backend():
    from eval.run_ragas_eval import _build_doc_agent
    from src.config import load_config

    config = load_config()
    agent = _build_doc_agent()

    if config.agents.llm_backend == "ollama":
        assert agent.llm is not None
        assert agent.llm.model == config.agents.ollama_model
    else:
        assert agent.llm is not None
