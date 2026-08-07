from unittest.mock import AsyncMock, MagicMock

from src.query_expansion import QueryExpander


async def test_hyde_returns_query_when_no_llm():
    expander = QueryExpander(llm=None)
    assert await expander.hyde("what is our refund policy") == "what is our refund policy"


async def test_hyde_returns_llm_generated_passage():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value="Our refund policy allows returns within 30 days.")
    expander = QueryExpander(llm=llm)

    passage = await expander.hyde("what is our refund policy")

    assert passage == "Our refund policy allows returns within 30 days."
    llm.ainvoke.assert_awaited_once()
