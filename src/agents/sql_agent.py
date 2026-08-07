"""SQL agent: NL -> SQL via LLM, validated through an AST (sqlglot)
safety pass before ever touching the database.

The AST check is what makes this safe to expose to an LLM-generated
query: rather than regex-matching for "DROP"/"DELETE" (easy to evade),
we parse the SQL into a real syntax tree and only allow SELECT
statements with no forbidden expression types.
"""

from __future__ import annotations

import time

import sqlglot
from sqlglot import exp
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.telemetry import traced_call
from src.utils.retry_handler import with_retry, RetryableError
from src.utils.schemas import AgentResponse, RouteName

_FORBIDDEN_EXPR_TYPES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.TruncateTable,
    exp.Grant,
)


class UnsafeSQLError(ValueError):
    pass


def validate_sql(sql: str, dialect: str = "postgres") -> exp.Expression:
    """Parse `sql` and raise UnsafeSQLError unless it is a single, pure
    read-only SELECT statement. Returns the parsed AST on success.
    """

    statements = sqlglot.parse(sql, read=dialect)
    if len(statements) != 1 or statements[0] is None:
        raise UnsafeSQLError("expected exactly one SQL statement")

    tree = statements[0]

    if not isinstance(tree, exp.Select):
        raise UnsafeSQLError(f"only SELECT statements are allowed, got {type(tree).__name__}")

    for node in tree.walk():
        node_obj = node[0] if isinstance(node, tuple) else node
        if isinstance(node_obj, _FORBIDDEN_EXPR_TYPES):
            raise UnsafeSQLError(f"forbidden expression in query: {type(node_obj).__name__}")

    return tree


def enforce_row_limit(tree: exp.Expression, max_rows: int = 1000) -> str:
    """Cap result size by injecting/lowering a LIMIT clause in the AST,
    then re-render to SQL text so the guard can't be bypassed by
    string-level edits after validation.
    """

    existing_limit = tree.args.get("limit")
    if existing_limit is not None:
        try:
            current = int(existing_limit.expression.this)
            if current > max_rows:
                tree.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
        except (AttributeError, ValueError):
            tree.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
    else:
        tree.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))

    return tree.sql(dialect="postgres")


class SQLAgent:
    def __init__(self, database_url: str, dialect: str = "postgres", max_rows: int = 1000):
        self.engine: Engine = create_engine(database_url)
        self.dialect = dialect
        self.max_rows = max_rows

    @with_retry(exceptions=(RetryableError, ConnectionError, TimeoutError))
    def _execute(self, sql: str) -> list[dict]:
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                return [dict(row._mapping) for row in result]
        except Exception as exc:
            raise RetryableError(str(exc)) from exc

    def run_query(self, sql: str) -> AgentResponse:
        start = time.perf_counter()
        with traced_call("sql"):
            tree = validate_sql(sql, dialect=self.dialect)
            safe_sql = enforce_row_limit(tree, max_rows=self.max_rows)
            rows = self._execute(safe_sql)
            answer = f"Query returned {len(rows)} row(s):\n{rows[:20]}"

        return AgentResponse(
            answer=answer,
            route=RouteName.SQL,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={"sql": safe_sql, "row_count": len(rows), "rows": rows},
        )
