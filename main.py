"""FastAPI entrypoint (and CLI) for the multi-agent workspace.

Wires VectorStore -> HybridRetriever -> RAGAgent, plus the optional
SQL/Doc/Web agents, into the LangGraph AgentGraph ("the brain") and
exposes it over HTTP for the demo UI / integration tests. The Redis
checkpointer is opened once in the app's lifespan handler and shared by
every request.

`POST /api/v1/agent` is the primary API: it accepts `agent_type="auto"`
(routed by the LangGraph classifier) or an explicit route, and can
either return a single ChatResponse or stream an SSE event feed
(metadata/token/chart/citations/done) via `stream=true`.

Runtime settings live in config.yaml (see src/config.py); env vars still
carry secrets (${DATABASE_URL}, API keys). This module is dual-purpose:
`uvicorn main:app` serves the FastAPI app (unaffected by anything below),
while `python main.py <command>` drives the click CLI defined at the
bottom, for one-shot/offline use (indexing, ad-hoc queries, eval) that
doesn't need a running server.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Optional

import click
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sse_starlette.sse import EventSourceResponse

load_dotenv()

from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from src.agents.doc_agent import DocAgent
from src.agents.rag_agent import RAGAgent
from src.agents.router import AgentGraph
from src.agents.sql_agent import SQLAgent
from src.agents.web_agent import WebAgent
from src.cache import ResponseCache
from src.config import load_config
from src.document_processing import DocumentProcessor
from src.hybrid_retrieval import HybridRetriever
from src.middleware import APIKeyMiddleware, RateLimiter
from src.reranking import Reranker
from src.utils.schemas import Citation, RouteName
from src.vectorstore import VectorStore

_config = load_config()
console = Console()

_vector_store = VectorStore()
_retriever = HybridRetriever(_vector_store, alpha=_config.retrieval.alpha)
_document_processor = DocumentProcessor(
    ocr_lang=_config.doc_intelligence.ocr_lang,
    ocr_dpi=_config.doc_intelligence.ocr_dpi,
    mask_pii=_config.doc_intelligence.redact_pii,
)
_response_cache = ResponseCache(ttl_seconds=_config.cache.ttl)
_reranker = Reranker(use_flashrank=_config.retrieval.use_flashrank, flashrank_model=_config.retrieval.flashrank_model)
_rag_agent = RAGAgent(
    retriever=_retriever,
    reranker=_reranker,
    cache=_response_cache,
    top_k=_config.retrieval.top_k,
    rerank_top_k=_config.retrieval.rerank_top_k,
    use_rerank=_config.retrieval.use_rerank,
)
_doc_agent = DocAgent(document_processor=_document_processor)
_web_agent = WebAgent(max_results=_config.web.max_results)

_database_url = _config.sql.database_url or os.getenv("DATABASE_URL")
_sql_agent = (
    SQLAgent(
        _database_url,
        max_rows=_config.sql.max_rows,
        check_ambiguity=_config.sql.check_ambiguity,
        llm_backend=_config.agents.llm_backend,
        ollama_model=_config.agents.ollama_model,
        ollama_base_url=_config.agents.ollama_base_url,
    )
    if _database_url
    else None
)

_rate_limiter = RateLimiter(redis_url=os.getenv("REDIS_URL"), capacity=60, window_seconds=60.0)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise RuntimeError("REDIS_URL is not set; required for RedisSaver checkpointing")

    async with AsyncRedisSaver.from_conn_string(redis_url) as checkpointer:
        await checkpointer.asetup()
        app.state.agent_graph = AgentGraph(
            rag_agent=_rag_agent,
            sql_agent=_sql_agent,
            doc_agent=_doc_agent,
            web_agent=_web_agent,
            checkpointer=checkpointer,
            llm_backend=_config.agents.llm_backend,
            ollama_model=_config.agents.ollama_model,
            ollama_base_url=_config.agents.ollama_base_url,
        )
        yield


app = FastAPI(title="Multi-Agent Workspace", version="3.1.0", lifespan=lifespan)
app.add_middleware(APIKeyMiddleware)


class QueryRequest(BaseModel):
    query: str
    file_path: Optional[str] = None
    user_id: str = ""
    session_id: str = ""


class ChatRequest(BaseModel):
    query: str
    user_id: str = ""
    session_id: str = ""
    chat_history: list[dict] = Field(default_factory=list)
    agent_type: Literal["auto", "rag", "sql", "doc", "web"] = "auto"
    file_path: Optional[str] = None
    stream: bool = False


class ChatResponse(BaseModel):
    query_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    route: RouteName
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


async def enforce_rate_limit(chat_request: ChatRequest) -> ChatRequest:
    key = chat_request.user_id or "anonymous"
    if not await _rate_limiter.allow(key):
        raise HTTPException(status_code=429, detail="rate limit exceeded: 60 requests/min per user_id")
    return chat_request


def _build_chart(rows: list[dict]) -> Optional[dict]:
    """Best-effort bar chart (first numeric column vs. first label
    column) from SQL result rows, for the SSE `chart` event. Returns
    None if the rows don't have an obvious numeric column to plot.
    """

    if not rows:
        return None

    import plotly.graph_objects as go

    columns = list(rows[0].keys())
    numeric_cols = [c for c in columns if all(isinstance(r.get(c), (int, float)) for r in rows)]
    if not numeric_cols:
        return None

    label_col = next((c for c in columns if c not in numeric_cols), columns[0])
    value_col = numeric_cols[0]
    fig = go.Figure(
        data=[go.Bar(x=[str(r.get(label_col)) for r in rows], y=[r.get(value_col) for r in rows])]
    )
    return fig.to_dict()


async def _sse_agent_stream(chat_request: ChatRequest, query_id: str, route: str) -> AsyncIterator[dict]:
    yield {
        "event": "metadata",
        "data": json.dumps({"query_id": query_id, "route": route, "session_id": chat_request.session_id}),
    }

    charts_enabled = _config.sql.enable_charts

    try:
        async for event in app.state.agent_graph.astream_events(
            chat_request.query,
            user_id=chat_request.user_id,
            session_id=chat_request.session_id,
            chat_history=chat_request.chat_history,
            file_path=chat_request.file_path,
            forced_route=route,
        ):
            kind = event.get("event")
            name = event.get("name")

            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                content = getattr(chunk, "content", None) if chunk is not None else None
                if content:
                    yield {"event": "token", "data": json.dumps({"query_id": query_id, "content": content})}

            elif kind == "on_chain_end" and name in {"rag", "sql", "doc", "web", "blocked"}:
                output = event.get("data", {}).get("output") or {}

                citations = output.get("citations") or []
                if citations:
                    yield {
                        "event": "citations",
                        "data": json.dumps(
                            [c.model_dump(mode="json") if hasattr(c, "model_dump") else c for c in citations]
                        ),
                    }

                sql_rows = output.get("sql_result")
                if charts_enabled and sql_rows:
                    chart = _build_chart(sql_rows)
                    if chart:
                        yield {"event": "chart", "data": json.dumps(chart, default=str)}
    except Exception as exc:
        logger.bind(query_id=query_id).exception("agent stream failed")
        yield {"event": "error", "data": json.dumps({"query_id": query_id, "error": str(exc)})}
    finally:
        yield {"event": "done", "data": json.dumps({"query_id": query_id})}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "3.1.0"}


@app.post("/api/v1/agent")
async def agent_endpoint(chat_request: ChatRequest = Depends(enforce_rate_limit)):
    query_id = str(uuid.uuid4())

    if chat_request.agent_type != "auto":
        route = chat_request.agent_type
    else:
        route = await app.state.agent_graph.classify_route(
            chat_request.query, has_file=bool(chat_request.file_path)
        )

    logger.bind(query_id=query_id, route=route, user_id=chat_request.user_id).info("agent_request")

    if chat_request.stream:
        return EventSourceResponse(
            _sse_agent_stream(chat_request, query_id, route),
            headers={"X-Query-ID": query_id, "X-Agent-Route": route},
        )

    result = await app.state.agent_graph.run(
        chat_request.query,
        file_path=chat_request.file_path,
        user_id=chat_request.user_id,
        session_id=chat_request.session_id,
        chat_history=chat_request.chat_history,
        forced_route=route,
    )
    response_payload = ChatResponse(
        query_id=query_id,
        answer=result.answer,
        citations=result.citations,
        route=result.route,
        latency_ms=result.latency_ms,
        metadata=result.metadata,
    )
    return JSONResponse(
        content=response_payload.model_dump(mode="json"),
        headers={"X-Query-ID": query_id, "X-Agent-Route": route},
    )


@app.post("/query")
async def query(request: QueryRequest) -> dict:
    response = await app.state.agent_graph.run(
        request.query,
        file_path=request.file_path,
        user_id=request.user_id,
        session_id=request.session_id,
    )
    return response.model_dump(mode="json")


@app.post("/ingest")
def ingest(file: UploadFile) -> dict:
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename
    dest.write_bytes(file.file.read())

    _, chunks = _document_processor.process(dest)
    n_added = _vector_store.add_chunks(chunks)
    _retriever.index_corpus(chunks)

    return {"file": file.filename, "chunks_indexed": n_added}



# ---------------------------------------------------------------------
# CLI (`python main.py <command>`) -- only reached when this module is
# run directly, never when uvicorn imports it as `main:app`.
# ---------------------------------------------------------------------


async def _build_cli_agent_graph() -> AgentGraph:
    """AgentGraph for one-shot CLI use. Tries the Redis checkpointer
    (per config `agents.memory: redis`) so `--session-id`/`chat`
    continuity survives across separate CLI invocations; falls back to
    an in-memory checkpointer (scoped to this single process) if Redis
    isn't reachable, so the CLI still works without it.
    """

    checkpointer = None
    if _config.agents.memory == "redis" and os.getenv("REDIS_URL"):
        try:
            checkpointer = await AsyncRedisSaver.from_conn_string(os.getenv("REDIS_URL")).__aenter__()
            await checkpointer.asetup()
        except Exception as exc:
            console.print(f"[yellow]Redis checkpointer unavailable ({exc}); using in-memory session state.[/yellow]")
            checkpointer = None

    return AgentGraph(
        rag_agent=_rag_agent,
        sql_agent=_sql_agent,
        doc_agent=_doc_agent,
        web_agent=_web_agent,
        checkpointer=checkpointer,
        llm_backend=_config.agents.llm_backend,
        ollama_model=_config.agents.ollama_model,
        ollama_base_url=_config.agents.ollama_base_url,
    )


def _print_agent_response(result) -> None:
    console.print(Panel(result.answer, title=f"[bold]{result.route.value}[/bold]", border_style="cyan"))
    if result.citations:
        table = Table(title="Citations")
        table.add_column("#", justify="right")
        table.add_column("Source")
        table.add_column("Quote")
        for c in result.citations:
            table.add_row(str(c.id), c.source, (c.quote or "")[:80])
        console.print(table)


@click.group()
def cli() -> None:
    """Multi-Agent Workspace CLI."""


@cli.command()
@click.option(
    "--sources",
    default="data/sample_documents",
    show_default=True,
    help="File or directory to index into the vector store + BM25 corpus.",
)
def index(sources: str) -> None:
    """Index documents for retrieval (DocumentProcessor -> VectorStore)."""

    source_path = Path(sources)
    if not source_path.exists():
        console.print(f"[red]no such file or directory: {sources}[/red]")
        raise SystemExit(1)
    files = [source_path] if source_path.is_file() else sorted(p for p in source_path.glob("*") if p.is_file())

    table = Table(title=f"Indexing {len(files)} file(s) from {sources}")
    table.add_column("File")
    table.add_column("Chunks", justify="right")
    table.add_column("Status")

    total_chunks = 0
    for file_path in files:
        try:
            _, chunks = _document_processor.process(file_path)
            n_added = _vector_store.add_chunks(chunks)
            _retriever.index_corpus(chunks)
            total_chunks += n_added
            table.add_row(file_path.name, str(n_added), "[green]ok[/green]")
        except Exception as exc:
            table.add_row(file_path.name, "-", f"[red]error: {exc}[/red]")

    console.print(table)
    console.print(f"[bold]Total chunks indexed:[/bold] {total_chunks}")


@cli.command()
@click.argument("query_text")
@click.option(
    "--agent",
    "agent_type",
    type=click.Choice(["auto", "rag", "sql", "doc", "web"]),
    default="auto",
    show_default=True,
    help="Force a route, bypassing the LangGraph classifier.",
)
@click.option("--stream", is_flag=True, default=False, help="Stream events via astream_events.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Print JSON instead of rich formatting.")
@click.option("--user-id", default="cli-user", show_default=True)
@click.option("--session-id", default="", help="Session id for conversational memory continuity.")
@click.option("--file-path", default=None, help="Path to a file for the doc agent.")
def agent(
    query_text: str,
    agent_type: str,
    stream: bool,
    as_json: bool,
    user_id: str,
    session_id: str,
    file_path: Optional[str],
) -> None:
    """Run a single query through the router (or a forced --agent)."""

    asyncio.run(_run_agent(query_text, agent_type, stream, as_json, user_id, session_id, file_path))


async def _run_agent(
    query_text: str,
    agent_type: str,
    stream: bool,
    as_json: bool,
    user_id: str,
    session_id: str,
    file_path: Optional[str],
) -> None:
    graph = await _build_cli_agent_graph()
    forced_route = None if agent_type == "auto" else agent_type

    if not stream:
        result = await graph.run(
            query_text, file_path=file_path, user_id=user_id, session_id=session_id, forced_route=forced_route
        )
        if as_json:
            console.print_json(json.dumps(result.model_dump(mode="json")))
        else:
            _print_agent_response(result)
        return

    async for event in graph.astream_events(
        query_text, user_id=user_id, session_id=session_id, file_path=file_path, forced_route=forced_route
    ):
        kind = event.get("event")
        name = event.get("name")

        if as_json:
            if kind in {"on_chat_model_stream", "on_chain_end"}:
                console.print_json(json.dumps({"event": kind, "name": name}, default=str))
            continue

        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            content = getattr(chunk, "content", None) if chunk is not None else None
            if content:
                console.print(content, end="")
        elif kind == "on_chain_end" and name in {"rag", "sql", "doc", "web", "blocked"}:
            output = event.get("data", {}).get("output") or {}
            answer = output.get("answer")
            if answer:
                console.print(answer)
            for c in output.get("citations") or []:
                console.print(f"  [dim][{c.id}][/dim] {c.source}")
    console.print()


@cli.command()
@click.argument("message", required=False)
@click.option("--user-id", default="cli-user", show_default=True)
@click.option("--session-id", default=None, help="Reuse a session id for continuity; generated if omitted.")
def chat(message: Optional[str], user_id: str, session_id: Optional[str]) -> None:
    """One-shot chat turn, or an interactive REPL when MESSAGE is omitted."""

    asyncio.run(_run_chat(message, user_id, session_id))


async def _run_chat(message: Optional[str], user_id: str, session_id: Optional[str]) -> None:
    session_id = session_id or str(uuid.uuid4())
    graph = await _build_cli_agent_graph()

    if message is not None:
        result = await graph.run(message, user_id=user_id, session_id=session_id)
        _print_agent_response(result)
        return

    console.print(f"[dim]session: {session_id} -- type 'exit' to quit[/dim]")
    while True:
        try:
            turn = console.input("[bold cyan]you>[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            break
        if turn.strip().lower() in {"exit", "quit"}:
            break
        if not turn.strip():
            continue
        result = await graph.run(turn, user_id=user_id, session_id=session_id)
        _print_agent_response(result)


@cli.command("mcp-serve")
def mcp_serve() -> None:
    """Start the MCP server exposing the RAG/SQL/Doc/Web agents as tools."""

    from src.mcp_server import mcp

    transport = _config.mcp.transport
    logger.info("starting MCP server (transport={})", transport)
    mcp.run(transport=transport)


def _eval_rag() -> dict:
    from eval.generate_testset import generate_testset
    from eval.run_ragas_eval import run_eval as run_ragas_eval

    testset_path = Path("eval/testset.json")
    if not testset_path.exists():
        generate_testset("data/sample_documents", str(testset_path))
    return run_ragas_eval(str(testset_path), "eval/results.json")


_SQL_EVAL_QUESTIONS = [
    "How many users are there?",
    "What is the total amount of completed orders?",
]

_DOC_EVAL_QUESTIONS = [
    ("data/sample_documents/refund_policy.txt", "What is the refund policy?"),
]


async def _eval_sql() -> dict:
    if _sql_agent is None:
        return {"status": "skipped: DATABASE_URL is not configured"}

    passed = 0
    for question in _SQL_EVAL_QUESTIONS:
        try:
            response = await _sql_agent.answer(question)
            if response.metadata.get("rows") is not None:
                passed += 1
        except Exception as exc:
            logger.warning("sql eval question failed: {} ({})", question, exc)

    return {"passed": passed, "total": len(_SQL_EVAL_QUESTIONS)}


async def _eval_doc() -> dict:
    passed = 0
    for file_path, question in _DOC_EVAL_QUESTIONS:
        try:
            response = _doc_agent.answer_from_file(file_path, question)
            if response.answer:
                passed += 1
        except Exception as exc:
            logger.warning("doc eval question failed: {} ({})", question, exc)

    return {"passed": passed, "total": len(_DOC_EVAL_QUESTIONS)}


def _eval_sql_guardrail() -> dict:
    from eval.run_ragas_eval import run_sql_guardrail_eval

    return run_sql_guardrail_eval()


async def _run_eval(eval_type: str) -> dict:
    results: dict[str, dict] = {}
    types = ["rag", "sql", "doc", "sql_guardrail"] if eval_type == "all" else [eval_type]

    for t in types:
        if t == "rag":
            results["rag"] = await asyncio.to_thread(_eval_rag)
        elif t == "sql":
            results["sql"] = await _eval_sql()
        elif t == "doc":
            results["doc"] = await _eval_doc()
        elif t == "sql_guardrail":
            results["sql_guardrail"] = await asyncio.to_thread(_eval_sql_guardrail)

    return results


@cli.command("eval")
@click.option(
    "--type",
    "eval_type",
    type=click.Choice(["all", "rag", "sql", "doc", "sql_guardrail"]),
    default="all",
    show_default=True,
)
@click.option("--json", "as_json", is_flag=True, default=False)
def eval_cmd(eval_type: str, as_json: bool) -> None:
    """Run quality evaluation (RAGAS for rag; smoke checks for sql/doc)."""

    results = asyncio.run(_run_eval(eval_type))
    if as_json:
        console.print_json(json.dumps(results, default=str))
        return

    table = Table(title=f"Eval results ({eval_type})")
    table.add_column("Type")
    table.add_column("Metric")
    table.add_column("Value")
    for type_name, metrics in results.items():
        for metric, value in metrics.items():
            value_str = f"{value:.3f}" if isinstance(value, float) else str(value)
            table.add_row(type_name, metric, value_str)
    console.print(table)


# (query, expected_route) pairs for a lightweight router-accuracy check.
_ROUTING_EVAL_CASES = [
    ("What does our refund policy say about returns?", "rag"),
    ("Summarize the key points in our knowledge base about onboarding.", "rag"),
    ("How many orders were completed this month?", "sql"),
    ("What's the total revenue from all users?", "sql"),
    ("Extract the totals table from invoice.pdf", "doc"),
    ("Summarize the attached report.docx", "doc"),
    ("What's the latest news about the Federal Reserve today?", "web"),
    ("Who won the game last night?", "web"),
]


async def _run_eval_agent(target: str) -> list[dict]:
    graph = await _build_cli_agent_graph()
    cases = _ROUTING_EVAL_CASES if target == "all" else [c for c in _ROUTING_EVAL_CASES if c[1] == target]

    rows = []
    for query_text, expected in cases:
        predicted = await graph.classify_route(query_text)
        rows.append({"query": query_text, "expected": expected, "predicted": predicted, "correct": predicted == expected})
    return rows


@cli.command("eval-agent")
@click.argument("target", type=click.Choice(["all", "rag", "sql", "doc", "web"]), default="all")
@click.option("--json", "as_json", is_flag=True, default=False)
def eval_agent_cmd(target: str, as_json: bool) -> None:
    """Evaluate router classification accuracy against labeled queries."""

    from eval.metrics import router_tool_selection_accuracy

    rows = asyncio.run(_run_eval_agent(target))
    accuracy = router_tool_selection_accuracy(rows)

    if as_json:
        console.print_json(json.dumps({"rows": rows, **accuracy}))
        return

    table = Table(title=f"Router accuracy ({target})")
    table.add_column("Query")
    table.add_column("Expected")
    table.add_column("Predicted")
    table.add_column("Result")
    for row in rows:
        status = "[green]correct[/green]" if row["correct"] else "[red]wrong[/red]"
        table.add_row(row["query"][:50], row["expected"], row["predicted"], status)
    console.print(table)
    console.print(f"[bold]Accuracy: {accuracy['correct']}/{accuracy['n']} ({accuracy['accuracy']:.1%})[/bold]")


if __name__ == "__main__":
    cli()
