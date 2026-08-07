"""FastAPI entrypoint for the multi-agent workspace.

Wires VectorStore -> HybridRetriever -> RAGAgent, plus the optional
SQL/Doc/Web agents, into the LangGraph AgentGraph ("the brain") and
exposes it over HTTP for the demo UI / integration tests. The Redis
checkpointer is opened once in the app's lifespan handler and shared by
every request.

`POST /api/v1/agent` is the primary API: it accepts `agent_type="auto"`
(routed by the LangGraph classifier) or an explicit route, and can
either return a single ChatResponse or stream an SSE event feed
(metadata/token/chart/citations/done) via `stream=true`.
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

load_dotenv()

from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from src.agents.doc_agent import DocAgent
from src.agents.rag_agent import RAGAgent
from src.agents.router import AgentGraph
from src.agents.sql_agent import SQLAgent
from src.agents.web_agent import WebAgent
from src.document_processing import DocumentProcessor
from src.hybrid_retrieval import HybridRetriever
from src.middleware import APIKeyMiddleware, RateLimiter
from src.utils.schemas import Citation, RouteName
from src.vectorstore import VectorStore

_vector_store = VectorStore()
_retriever = HybridRetriever(_vector_store)
_rag_agent = RAGAgent(retriever=_retriever)
_document_processor = DocumentProcessor()
_doc_agent = DocAgent(document_processor=_document_processor)
_web_agent = WebAgent()

_database_url = os.getenv("DATABASE_URL")
_sql_agent = SQLAgent(_database_url) if _database_url else None

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

    charts_enabled = os.getenv("ENABLE_CHARTS", "false").lower() == "true"

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
