"""Tests for Redis-backed conversation memory: `AgentGraph`'s checkpointer
wiring, and the module-level `src.agents.router._get_default_graph()`
convenience singleton (used by `from src.agents.router import
astream_events` outside the FastAPI app).

Root cause covered here: `_get_default_graph()` used to hard-crash
(`raise RuntimeError(...)` with no fallback, and zero try/except around
the Redis checkpointer setup) whenever REDIS_URL was unset or Redis was
unreachable -- unlike every other Redis-touching path in this codebase
(`src/cache.py`, `src/middleware.py`, `main.py`'s `lifespan`), which all
degrade to an in-memory fallback instead. Redis isn't running in this
environment, so the "Redis available" cases mock `AsyncRedisSaver` for
determinism -- see the task report for what was and wasn't live-tested.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from src.agents.router import AgentGraph
from src.utils.schemas import AgentResponse, RouteName


def _mock_classifier(route: str = "rag") -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content=route))
    return llm


def _patch_chroma(monkeypatch) -> None:
    import chromadb
    from chromadb.utils import embedding_functions

    monkeypatch.setattr(chromadb, "PersistentClient", MagicMock())
    monkeypatch.setattr(embedding_functions, "SentenceTransformerEmbeddingFunction", MagicMock())


@pytest.fixture
def reset_default_graph():
    """`_get_default_graph()` caches its result in a module-level global
    after the first successful build -- reset it before and after each
    test so tests don't leak state into each other.
    """

    import src.agents.router as router_module

    router_module._default_graph = None
    yield
    router_module._default_graph = None


# --- AgentGraph: conversation memory (default in-memory MemorySaver) -------


async def _make_graph(route: str = "rag") -> AgentGraph:
    rag_agent = MagicMock()
    rag_agent.answer = AsyncMock(return_value=AgentResponse(answer="rag answer", route=RouteName.RAG))
    return AgentGraph(rag_agent=rag_agent, classifier_llm=_mock_classifier(route))


async def test_agent_graph_defaults_to_memory_saver_without_checkpointer():
    graph = await _make_graph()
    assert isinstance(graph.checkpointer, MemorySaver)


async def test_agent_graph_persists_checkpoint_state_across_turns():
    """Genuine "conversation memory works" check: two `.run()` calls
    against the same session_id must leave a real, non-empty checkpoint
    behind (proving the graph is actually being checkpointed, not just
    that a MemorySaver object exists unused).
    """

    graph = await _make_graph()
    config = graph._config(session_id="session-1", user_id="")

    assert graph.checkpointer.get_tuple(config) is None  # nothing yet

    await graph.run("What is our refund policy?", session_id="session-1")
    first_checkpoint = graph.checkpointer.get_tuple(config)
    assert first_checkpoint is not None

    await graph.run("And what about exchanges?", session_id="session-1")
    second_checkpoint = graph.checkpointer.get_tuple(config)
    assert second_checkpoint is not None
    assert second_checkpoint.config["configurable"]["checkpoint_id"] != first_checkpoint.config["configurable"]["checkpoint_id"]


async def test_agent_graph_checkpoint_state_isolated_per_session():
    graph = await _make_graph()

    await graph.run("question from alice", session_id="alice-session")
    await graph.run("question from bob", session_id="bob-session")

    alice_config = graph._config(session_id="alice-session", user_id="")
    bob_config = graph._config(session_id="bob-session", user_id="")

    assert graph.checkpointer.get_tuple(alice_config) is not None
    assert graph.checkpointer.get_tuple(bob_config) is not None
    assert (
        graph.checkpointer.get_tuple(alice_config).config["configurable"]["checkpoint_id"]
        != graph.checkpointer.get_tuple(bob_config).config["configurable"]["checkpoint_id"]
    )


# --- _get_default_graph(): the fixed hard-crash path ------------------------


async def test_get_default_graph_falls_back_to_memory_saver_when_no_redis_url(
    monkeypatch, reset_default_graph, mock_anthropic
):
    """Regression test for the reported crash: REDIS_URL unset used to
    raise `RuntimeError("REDIS_URL is not set...")` with no fallback.
    """

    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    _patch_chroma(monkeypatch)

    from src.agents.router import _get_default_graph

    graph = await _get_default_graph()

    assert isinstance(graph.checkpointer, MemorySaver)


async def test_get_default_graph_falls_back_to_memory_saver_when_redis_unreachable(
    monkeypatch, reset_default_graph, mock_anthropic
):
    """REDIS_URL set but the server unreachable must also degrade
    gracefully -- this used to have zero try/except at all.
    """

    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    _patch_chroma(monkeypatch)

    with patch(
        "langgraph.checkpoint.redis.aio.AsyncRedisSaver.from_conn_string",
        side_effect=ConnectionError("connection refused"),
    ):
        from src.agents.router import _get_default_graph

        graph = await _get_default_graph()

    assert isinstance(graph.checkpointer, MemorySaver)


async def test_get_default_graph_uses_redis_checkpointer_when_available(
    monkeypatch, reset_default_graph, mock_anthropic
):
    """When Redis setup succeeds, the real (mocked) Redis checkpointer
    must be wired in -- not silently ignored in favor of MemorySaver.
    """

    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    _patch_chroma(monkeypatch)

    # A real MemorySaver stands in for AsyncRedisSaver here -- it's a
    # genuine BaseCheckpointSaver (langgraph's `.compile()` rejects a
    # bare Mock), and using identity (`is`) below still proves this
    # specific object flowed through rather than a *different* fallback
    # MemorySaver AgentGraph might have constructed on its own.
    fake_checkpointer = MemorySaver()
    fake_checkpointer.asetup = AsyncMock(return_value=None)

    class _FakeAsyncCM:
        async def __aenter__(self):
            return fake_checkpointer

        async def __aexit__(self, *exc_info):
            return False

    with patch(
        "langgraph.checkpoint.redis.aio.AsyncRedisSaver.from_conn_string",
        return_value=_FakeAsyncCM(),
    ):
        from src.agents.router import _get_default_graph

        graph = await _get_default_graph()

    assert graph.checkpointer is fake_checkpointer
    fake_checkpointer.asetup.assert_awaited_once()


# --- main.py's lifespan handler: the already-correct pattern this fix mirrors


def test_main_lifespan_falls_back_to_memory_saver_when_redis_unavailable(app_module):
    """`app_module` (session-scoped) already runs with REDIS_URL unset,
    so entering the lifespan context here exercises the same fallback
    path as `_get_default_graph()` above -- proving `main.py`'s handler
    (the pattern the fix was modeled on) actually works, not just that
    it looks correct on paper.
    """

    with TestClient(app_module.app) as client:
        assert isinstance(client.app.state.agent_graph.checkpointer, MemorySaver)


def test_main_lifespan_uses_redis_checkpointer_when_available(app_module, monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    fake_checkpointer = MemorySaver()
    fake_checkpointer.asetup = AsyncMock(return_value=None)

    class _FakeAsyncCM:
        async def __aenter__(self):
            return fake_checkpointer

        async def __aexit__(self, *exc_info):
            return False

    with patch(
        "langgraph.checkpoint.redis.aio.AsyncRedisSaver.from_conn_string",
        return_value=_FakeAsyncCM(),
    ):
        with TestClient(app_module.app) as client:
            assert client.app.state.agent_graph.checkpointer is fake_checkpointer
