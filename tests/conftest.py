from unittest.mock import AsyncMock, MagicMock

import pytest

from src.document_processing import DocumentProcessor


@pytest.fixture(scope="session")
def document_processor() -> DocumentProcessor:
    """Shared DocumentProcessor instance.

    Presidio/spaCy model loading in DocumentProcessor.__init__ is
    expensive, so the whole test session reuses one instance instead of
    each test module constructing its own.
    """

    return DocumentProcessor(mask_pii=False)


def _patch_chroma(monkeypatch) -> None:
    """Replace Chroma's persistent client + embedding model with mocks so
    constructing a real VectorStore (directly, or transitively via
    `main.py`/`src/mcp_server.py` module-level wiring) doesn't touch disk,
    download a sentence-transformers model, or need network access.
    """

    import chromadb
    from chromadb.utils import embedding_functions

    monkeypatch.setattr(chromadb, "PersistentClient", MagicMock())
    monkeypatch.setattr(embedding_functions, "SentenceTransformerEmbeddingFunction", MagicMock())


@pytest.fixture
def mock_anthropic(monkeypatch):
    """Patches `langchain_anthropic.ChatAnthropic` so any code path that
    lazily constructs a real Claude client (e.g. `SQLAgent._default_llm`,
    `AgentGraph`'s default `classifier_llm`) doesn't need `ANTHROPIC_API_KEY`
    or a live network call.
    """

    import langchain_anthropic

    mock_cls = MagicMock(return_value=MagicMock(ainvoke=AsyncMock(return_value="mocked")))
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", mock_cls)
    return mock_cls


@pytest.fixture(scope="session")
def app_module():
    """Import `main` once for the whole session with Chroma/embedding-model
    construction short-circuited and REDIS_URL/DATABASE_URL/API_KEY unset,
    so hitting the FastAPI app in tests needs no real infra.

    `main.py` builds its VectorStore/RAGAgent/SQLAgent/etc. as module-level
    globals at import time, so these patches must be in place *before* the
    first `import main` -- callers must not import `main` themselves.
    """

    mp = pytest.MonkeyPatch()
    mp.delenv("REDIS_URL", raising=False)
    mp.delenv("DATABASE_URL", raising=False)
    mp.delenv("API_KEY", raising=False)
    _patch_chroma(mp)

    import main

    yield main

    mp.undo()


@pytest.fixture
def sales_sql_agent():
    """Factory for a `SQLAgent` backed by a persistent in-memory sqlite
    'sales' table (id, region, amount), for tests that exercise the
    NL->SQL pipeline over real (if tiny) data without a live Postgres.

    A plain `sqlite:///:memory:` engine gets a fresh, empty database per
    connection, so this uses `StaticPool` to keep one connection (and thus
    the table) alive for the agent's lifetime.
    """

    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import StaticPool

    from src.agents.sql_agent import SQLAgent

    def _make(llm=None) -> SQLAgent:
        engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE sales (id INTEGER, region TEXT, amount REAL)"))
            conn.execute(
                text(
                    "INSERT INTO sales (id, region, amount) VALUES "
                    "(1, 'East', 100.0), (2, 'West', 150.0), (3, 'East', 50.0)"
                )
            )
        agent = SQLAgent("sqlite:///:memory:", llm=llm)
        agent.engine = engine
        return agent

    return _make
