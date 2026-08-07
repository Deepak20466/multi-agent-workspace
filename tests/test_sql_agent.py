from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import text

from src.agents.sql_agent import SQL_MODEL, SQLAgent, UnsafeSQLError, enforce_row_limit, validate_sql


def _make_agent(tmp_path, llm=None) -> SQLAgent:
    db_path = tmp_path / "test.db"
    agent = SQLAgent(f"sqlite:///{db_path}", llm=llm)
    with agent.engine.begin() as conn:
        conn.execute(text("CREATE TABLE sales (id INTEGER, region TEXT)"))
        conn.execute(text("INSERT INTO sales (id, region) VALUES (1, 'East'), (2, 'West')"))
    return agent


def test_validate_sql_rejects_non_select():
    with pytest.raises(UnsafeSQLError):
        validate_sql("DROP TABLE sales")


def test_validate_sql_accepts_select():
    assert validate_sql("SELECT * FROM sales") is not None


def test_validate_sql_rejects_multiple_statements():
    with pytest.raises(UnsafeSQLError):
        validate_sql("SELECT * FROM sales; DROP TABLE sales;")


def test_enforce_row_limit_caps_existing_limit():
    tree = validate_sql("SELECT * FROM sales LIMIT 100000")
    assert "LIMIT 10" in enforce_row_limit(tree, max_rows=10)


def test_run_query_executes_validated_sql(tmp_path):
    agent = _make_agent(tmp_path)
    response = agent.run_query("SELECT * FROM sales")
    assert response.metadata["row_count"] == 2


async def test_generate_sql_strips_code_fence_and_uses_schema(tmp_path):
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value="```sql\nSELECT * FROM sales\n```")
    agent = _make_agent(tmp_path, llm=llm)

    sql = await agent._generate_sql("how many sales are there")

    assert sql == "SELECT * FROM sales"
    prompt = llm.ainvoke.call_args[0][0]
    assert "sales" in prompt


async def test_answer_generates_validates_and_executes(tmp_path):
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value="SELECT * FROM sales")
    agent = _make_agent(tmp_path, llm=llm)

    response = await agent.answer("list all sales")

    assert response.metadata["row_count"] == 2
    assert response.metadata["nl_query"] == "list all sales"


async def test_answer_rejects_unsafe_generated_sql(tmp_path):
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value="DROP TABLE sales")
    agent = _make_agent(tmp_path, llm=llm)

    with pytest.raises(UnsafeSQLError):
        await agent.answer("delete everything")


async def test_nl_to_sql_generates_valid_select_from_mocked_llm(sales_sql_agent):
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value="SELECT region, SUM(amount) AS total FROM sales GROUP BY region")
    agent = sales_sql_agent(llm=llm)

    sql = await agent._generate_sql("total sales by region")

    assert sql.strip().upper().startswith("SELECT")
    assert "sales" in sql.lower()
    validate_sql(sql)  # doesn't raise: it's a single, safe SELECT


def test_sql_ast_guardrail_blocks_delete_and_cte_writes():
    """Postgres allows data-modifying statements inside a WITH clause (e.g.
    `WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x`), so checking
    only the outermost statement's type isn't enough -- validate_sql must
    walk the whole AST to catch a write hidden inside a CTE.
    """
    with pytest.raises(UnsafeSQLError):
        validate_sql("WITH deleted AS (DELETE FROM sales RETURNING *) SELECT * FROM deleted")


def test_default_llm_uses_configured_model_without_real_anthropic_client(mock_anthropic, tmp_path):
    agent = _make_agent(tmp_path)

    llm = agent._default_llm()

    mock_anthropic.assert_called_once()
    assert llm is mock_anthropic.return_value
    _, kwargs = mock_anthropic.call_args
    assert kwargs["model"] == SQL_MODEL
