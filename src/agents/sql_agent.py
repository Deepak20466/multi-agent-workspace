"""SQL agent: NL -> SQL via an LLM (Haiku by default), validated
through an AST (sqlglot) safety pass before ever touching the database.

The AST check is what makes this safe to expose to an LLM-generated
query: rather than regex-matching for "DROP"/"DELETE" (easy to evade),
we parse the SQL into a real syntax tree and only allow SELECT
statements with no forbidden expression types.
"""

from __future__ import annotations

import asyncio
import os
import re
import time

import sqlglot
from sqlglot import exp
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from src.telemetry import traced_call
from src.utils.retry_handler import (
    APIConnectionError,
    RateLimitError,
    RetryableError,
    async_api_rate_limit_retry,
    with_retry,
)
from src.utils.schemas import AgentResponse, RouteName

SQL_MODEL = os.getenv("SQL_MODEL", "claude-haiku-4-5")

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

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")

# Words that imply a ranking/ordering the LLM has to invent a metric and a
# row count for if the question doesn't supply them -- e.g. "show top
# customers" could mean top-5-by-revenue or top-10-by-order-count, and
# guessing silently produces a plausible-looking but arbitrary answer.
_VAGUE_SUPERLATIVES = (
    "top", "best", "highest", "most", "leading", "popular",
    "worst", "lowest", "largest", "biggest", "greatest",
)
_NUMBER_RE = re.compile(r"\d+")
_SCHEMA_COLUMNS_RE = re.compile(r"\((.*?)\)")

_SQL_GENERATION_PROMPT = (
    "You translate natural-language questions into a single read-only "
    "PostgreSQL SELECT statement.\n\n"
    "Database schema:\n{schema}\n\n"
    "Rules:\n"
    "- Output ONLY the SQL statement — no markdown fences, no commentary.\n"
    "- Use only the tables/columns listed in the schema above.\n"
    "- Write exactly one statement: a SELECT. Never write INSERT, UPDATE, "
    "DELETE, DROP, ALTER, CREATE, GRANT, or multiple statements.\n\n"
    "Question: {question}"
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


def _schema_columns(schema: str) -> set[str]:
    """Extract lowercased column names out of the compact
    `table(col type, col type)` string `get_schema()` produces."""

    columns: set[str] = set()
    for match in _SCHEMA_COLUMNS_RE.finditer(schema):
        for col_def in match.group(1).split(","):
            name = col_def.strip().split(" ")[0]
            if name:
                columns.add(name.lower())
    return columns


def detect_ambiguity(nl_query: str, schema: str) -> dict | None:
    """Heuristic ambiguity check run ahead of NL->SQL generation.

    Flags questions that use a vague superlative ("top", "best", ...)
    without either a concrete result count ("top 10") or a named
    metric/column to rank by ("best... by revenue"). Without both, the
    LLM has to silently invent a LIMIT and an ORDER BY column, which
    produces a plausible-looking but arbitrary query rather than an
    error -- surfacing a clarification question is safer than guessing.

    Returns None when the question is unambiguous (or matches no
    superlative at all), else a dict with `ambiguous_terms`, `missing`,
    and a ready-to-show `clarification_question`.
    """

    lowered = nl_query.lower()
    matched_terms = [t for t in _VAGUE_SUPERLATIVES if re.search(rf"\b{t}\b", lowered)]
    if not matched_terms:
        return None

    has_number = bool(_NUMBER_RE.search(nl_query))
    columns = _schema_columns(schema)
    mentions_metric = any(re.search(rf"\b{re.escape(col)}\b", lowered) for col in columns)

    missing = []
    if not has_number:
        missing.append('how many results you want (e.g. "top 10")')
    if not mentions_metric:
        missing.append("which column or metric to rank by (e.g. revenue, order count)")

    if not missing:
        return None

    return {
        "ambiguous_terms": matched_terms,
        "missing": missing,
        "clarification_question": (
            f'Your question uses "{matched_terms[0]}" but doesn\'t specify {" and ".join(missing)}. '
            "Could you clarify?"
        ),
    }


def _strip_code_fence(text_: str) -> str:
    return _CODE_FENCE_RE.sub("", text_.strip()).strip()


def _extract_text(response: object) -> str:
    """Agent-level `llm` objects in this codebase are expected to return
    a plain string from `.ainvoke`, but a raw langchain ChatModel returns
    a message object instead — accept either.
    """

    content = getattr(response, "content", None)
    return str(content) if content is not None else str(response)


class SQLAgent:
    def __init__(
        self,
        database_url: str,
        dialect: str = "postgres",
        max_rows: int = 1000,
        llm=None,
        check_ambiguity: bool = True,
        llm_backend: str | None = None,
        ollama_model: str | None = None,
        ollama_base_url: str | None = None,
    ):
        self.engine: Engine = create_engine(database_url)
        self.dialect = dialect
        self.max_rows = max_rows
        self.llm = llm
        self.check_ambiguity = check_ambiguity
        self.llm_backend = llm_backend
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self._schema_cache: str | None = None

    def _default_llm(self):
        from src.llm_factory import build_llm

        return build_llm(
            SQL_MODEL,
            backend=self.llm_backend,
            ollama_model=self.ollama_model,
            ollama_base_url=self.ollama_base_url,
        )

    @with_retry(exceptions=(RetryableError, ConnectionError, TimeoutError))
    def get_schema(self) -> str:
        """Introspect the connected database and render a compact
        `table(col type, ...)` description for the NL->SQL prompt.
        Cached after the first successful call.
        """

        if self._schema_cache is not None:
            return self._schema_cache

        try:
            inspector = inspect(self.engine)
            lines = []
            for table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                col_desc = ", ".join(f"{c['name']} {c['type']}" for c in columns)
                lines.append(f"{table_name}({col_desc})")
        except Exception as exc:
            raise RetryableError(str(exc)) from exc

        self._schema_cache = "\n".join(lines)
        return self._schema_cache

    @async_api_rate_limit_retry
    async def _generate_sql(self, nl_query: str) -> str:
        if self.llm is None:
            self.llm = self._default_llm()
        llm = self.llm
        schema = self.get_schema()
        prompt = _SQL_GENERATION_PROMPT.format(schema=schema, question=nl_query)

        try:
            response = await llm.ainvoke(prompt)
        except Exception as exc:  # normalize provider errors for the retry policy
            if "rate" in type(exc).__name__.lower() or "429" in str(exc):
                raise RateLimitError(str(exc)) from exc
            raise APIConnectionError(str(exc)) from exc

        return _strip_code_fence(_extract_text(response))

    @with_retry(exceptions=(RetryableError, ConnectionError, TimeoutError))
    def _execute(self, sql: str) -> list[dict]:
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                return [dict(row._mapping) for row in result]
        except Exception as exc:
            raise RetryableError(str(exc)) from exc

    def run_query(self, sql: str) -> AgentResponse:
        """Validate and execute an already-written SQL statement directly,
        bypassing NL->SQL generation.
        """

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

    async def answer(self, nl_query: str) -> AgentResponse:
        """Full NL->SQL pipeline: generate SQL for `nl_query` against the
        live schema, AST-validate it, cap its row limit, execute it, and
        return a grounded AgentResponse.
        """

        start = time.perf_counter()

        if self.check_ambiguity:
            ambiguity = detect_ambiguity(nl_query, self.get_schema())
            if ambiguity is not None:
                return AgentResponse(
                    answer=ambiguity["clarification_question"],
                    route=RouteName.SQL,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    metadata={"needs_clarification": True, "nl_query": nl_query, **ambiguity},
                )

        with traced_call("sql"):
            generated_sql = await self._generate_sql(nl_query)
            tree = validate_sql(generated_sql, dialect=self.dialect)
            safe_sql = enforce_row_limit(tree, max_rows=self.max_rows)
            rows = await asyncio.to_thread(self._execute, safe_sql)
            answer_text = f"Query returned {len(rows)} row(s):\n{rows[:20]}"

        return AgentResponse(
            answer=answer_text,
            route=RouteName.SQL,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={"nl_query": nl_query, "sql": safe_sql, "row_count": len(rows), "rows": rows},
        )
