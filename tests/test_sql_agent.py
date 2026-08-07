from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import text

from src.agents.sql_agent import SQLAgent, UnsafeSQLError, enforce_row_limit, validate_sql


def _make_agent(tmp_path, llm=None) -> SQLAgent:
    db_path = tmp_path / "test.db"
    agent = SQLAgent(f"sqlite:///{db_path}", llm=llm)
    with agent.engine.begin() as conn:
        conn.execute(text("CREATE TABLE users (id INTEGER, name TEXT)"))
        conn.execute(text("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')"))
    return agent


def test_validate_sql_rejects_non_select():
    with pytest.raises(UnsafeSQLError):
        validate_sql("DROP TABLE users")


def test_validate_sql_accepts_select():
    assert validate_sql("SELECT * FROM users") is not None


def test_validate_sql_rejects_multiple_statements():
    with pytest.raises(UnsafeSQLError):
        validate_sql("SELECT * FROM users; DROP TABLE users;")


def test_enforce_row_limit_caps_existing_limit():
    tree = validate_sql("SELECT * FROM users LIMIT 100000")
    assert "LIMIT 10" in enforce_row_limit(tree, max_rows=10)


def test_run_query_executes_validated_sql(tmp_path):
    agent = _make_agent(tmp_path)
    response = agent.run_query("SELECT * FROM users")
    assert response.metadata["row_count"] == 2


async def test_generate_sql_strips_code_fence_and_uses_schema(tmp_path):
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value="```sql\nSELECT * FROM users\n```")
    agent = _make_agent(tmp_path, llm=llm)

    sql = await agent._generate_sql("how many users are there")

    assert sql == "SELECT * FROM users"
    prompt = llm.ainvoke.call_args[0][0]
    assert "users" in prompt


async def test_answer_generates_validates_and_executes(tmp_path):
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value="SELECT * FROM users")
    agent = _make_agent(tmp_path, llm=llm)

    response = await agent.answer("list all users")

    assert response.metadata["row_count"] == 2
    assert response.metadata["nl_query"] == "list all users"


async def test_answer_rejects_unsafe_generated_sql(tmp_path):
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value="DROP TABLE users")
    agent = _make_agent(tmp_path, llm=llm)

    with pytest.raises(UnsafeSQLError):
        await agent.answer("delete everything")
