from unittest.mock import AsyncMock, MagicMock

from src.agents.web_agent import WebAgent
from src.utils.schemas import RouteName


def _fake_results():
    return [
        {"title": "Result One", "url": "https://a.example", "snippet": "snippet a"},
        {"title": "Result Two", "url": "https://b.example", "snippet": "snippet b"},
    ]


async def test_answer_uses_injected_search_fn_and_builds_citations():
    search_fn = MagicMock(return_value=_fake_results())
    agent = WebAgent(search_fn=search_fn)

    response = await agent.answer("latest news")

    assert response.route == RouteName.WEB
    assert len(response.citations) == 2
    assert response.citations[0].type == "web"
    assert response.citations[0].title == "Result One"
    assert response.citations[0].source == "https://a.example"
    search_fn.assert_called_once_with("latest news", 3)


async def test_answer_returns_graceful_error_without_search_fn_or_tavily_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    agent = WebAgent(search_fn=None, tavily_api_key=None)

    response = await agent.answer("latest news")

    assert response.route == RouteName.WEB
    assert "unavailable" in response.answer.lower()
    assert response.citations == []


async def test_answer_uses_llm_to_synthesize_when_configured():
    search_fn = MagicMock(return_value=_fake_results())
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value="Synthesized answer [1][2].")
    agent = WebAgent(search_fn=search_fn, llm=llm)

    response = await agent.answer("latest news")

    assert response.answer == "Synthesized answer [1][2]."
    llm.ainvoke.assert_awaited_once()


async def test_answer_respects_custom_max_results():
    search_fn = MagicMock(return_value=_fake_results())
    agent = WebAgent(search_fn=search_fn, max_results=1)

    await agent.answer("latest news")

    search_fn.assert_called_once_with("latest news", 1)
