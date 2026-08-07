"""LangGraph router — "the brain": classifies an incoming query with
Claude Haiku and dispatches it to the RAG, SQL, Doc, or Web agent, then
checkpoints every turn in Redis so a conversation (keyed by
`thread_id=session_id`) can resume across process restarts.

Graph shape: START -> router -> {rag,sql,doc,web,blocked} -> (tool_node
if tool_calls else aggregator) -> aggregator -> END. `tool_node` is
wired in per that shape even though none of the wrapped agents (they
call their own tools internally, see rag_agent/sql_agent/doc_agent/
web_agent) currently emit `tool_calls` — it exists so an agent can be
upgraded to emit them without a graph-shape change.

PII anonymization + prompt-injection detection (carried over from the
previous rule-based router) run inside `_router_node` itself, ahead of
classification, rather than as a separate graph node.
"""

from __future__ import annotations

import asyncio
import os
from typing import Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.doc_agent import DocAgent
from src.agents.rag_agent import RAGAgent
from src.agents.sql_agent import SQLAgent, UnsafeSQLError
from src.agents.web_agent import WebAgent
from src.guardrails import PIIGuard, detect_prompt_injection
from src.telemetry import log_event
from src.tools import TOOL_MAP
from src.utils.retry_handler import APIConnectionError, RateLimitError, async_api_rate_limit_retry
from src.utils.schemas import AgentResponse, AgentState, RouteName

ROUTER_MODEL = os.getenv("ROUTER_MODEL", "claude-haiku-4-5")

_VALID_ROUTES = {"rag", "sql", "doc", "web"}

_CLASSIFY_SYSTEM_PROMPT = (
    "You are a routing classifier for a multi-agent system. Read the user's "
    "query and choose exactly one category:\n"
    "- rag: general questions answerable from an internal knowledge base/document corpus\n"
    "- sql: questions needing a structured/aggregate database lookup (counts, sums, filters, \"how many\")\n"
    "- doc: questions about a specific attached/uploaded file\n"
    "- web: questions needing current or live information (news, today, latest)\n\n"
    "Respond with exactly one word — rag, sql, doc, or web — and nothing else."
)


def _unavailable_response(route: RouteName) -> AgentResponse:
    return AgentResponse(answer=f"The '{route.value}' agent is not configured.", route=route)


def _apply_response(response: AgentResponse) -> Dict:
    return {
        "answer": response.answer,
        "citations": response.citations,
        "documents": [c.model_dump() for c in response.citations],
        "latency_ms": response.latency_ms,
    }


class AgentGraph:
    """Builds and runs the LangGraph "brain" described above.

    `checkpointer` must be supplied by the caller — pass an
    `AsyncRedisSaver` (already `.asetup()`) for production use (see
    `main.py`'s lifespan handler); it defaults to an in-memory
    `MemorySaver` so tests and local scripts don't need a live Redis.
    """

    def __init__(
        self,
        rag_agent: RAGAgent,
        sql_agent: Optional[SQLAgent] = None,
        doc_agent: Optional[DocAgent] = None,
        web_agent: Optional[WebAgent] = None,
        pii_guard: Optional[PIIGuard] = None,
        classifier_llm=None,
        checkpointer=None,
    ):
        self.rag_agent = rag_agent
        self.sql_agent = sql_agent
        self.doc_agent = doc_agent
        self.web_agent = web_agent
        self.pii_guard = pii_guard or PIIGuard()
        self.classifier_llm = classifier_llm or ChatAnthropic(model=ROUTER_MODEL, temperature=0, max_tokens=10)
        self.checkpointer = checkpointer or MemorySaver()
        self.workflow = self._build_graph()

    # ---- classification ----

    async def _classify(self, query: str, has_file: bool) -> str:
        content = query if not has_file else f"{query}\n\n[A file is attached to this request.]"
        try:
            response = await self.classifier_llm.ainvoke(
                [SystemMessage(content=_CLASSIFY_SYSTEM_PROMPT), HumanMessage(content=content)]
            )
        except Exception as exc:  # normalize provider errors for async_api_rate_limit_retry
            if "rate" in type(exc).__name__.lower() or "429" in str(exc):
                raise RateLimitError(str(exc)) from exc
            raise APIConnectionError(str(exc)) from exc

        route = str(response.content).strip().lower()
        return route if route in _VALID_ROUTES else "rag"

    # ---- nodes ----

    @async_api_rate_limit_retry
    async def _router_node(self, state: AgentState) -> Dict:
        query = state["query"]
        if detect_prompt_injection(query):
            log_event("router_blocked_injection", query=query)
            return {"blocked": True, "block_reason": "potential prompt injection detected"}

        safe_query, _ = self.pii_guard.anonymize(query)
        route = await self._classify(safe_query, has_file=bool(state.get("file_path")))
        log_event("router_decision", route=route)
        return {"query": safe_query, "route": route, "blocked": False}

    async def _rag_node(self, state: AgentState) -> Dict:
        response = await self.rag_agent.answer(state["query"])
        return _apply_response(response)

    async def _sql_node(self, state: AgentState) -> Dict:
        if self.sql_agent is None:
            return _apply_response(_unavailable_response(RouteName.SQL))
        try:
            response = await self.sql_agent.answer(state["query"])
        except UnsafeSQLError as exc:
            response = AgentResponse(answer=f"Couldn't safely answer that as SQL: {exc}", route=RouteName.SQL)
        update = _apply_response(response)
        update["sql_result"] = response.metadata.get("rows")
        return update

    def _doc_node(self, state: AgentState) -> Dict:
        if self.doc_agent is None:
            return _apply_response(_unavailable_response(RouteName.DOC))
        try:
            response = self.doc_agent.answer_from_file(state.get("file_path"), state["query"])
        except FileNotFoundError as exc:
            response = AgentResponse(answer=f"Couldn't find a file to answer from: {exc}", route=RouteName.DOC)
        return _apply_response(response)

    async def _web_node(self, state: AgentState) -> Dict:
        if self.web_agent is None:
            return _apply_response(_unavailable_response(RouteName.WEB))
        response = await self.web_agent.answer(state["query"])
        return _apply_response(response)

    def _blocked_node(self, state: AgentState) -> Dict:
        response = AgentResponse(
            answer=f"Request blocked: {state.get('block_reason', 'unsafe input')}",
            route=RouteName.UNKNOWN,
        )
        return _apply_response(response)

    def _tool_node(self, state: AgentState) -> Dict:
        results = []
        for call in state.get("tool_calls") or []:
            name = call.get("name")
            entry = TOOL_MAP.get(name)
            if entry is None:
                results.append({"name": name, "error": f"unknown tool: {name}"})
                continue
            try:
                results.append({"name": name, "result": entry.invoke(call.get("args", {}))})
            except Exception as exc:
                results.append({"name": name, "error": str(exc)})
        return {"tool_calls": None, "documents": [*(state.get("documents") or []), *results]}

    def _aggregator_node(self, state: AgentState) -> Dict:
        log_event("router_aggregated", route=state.get("route"), blocked=state.get("blocked", False))
        return {}

    # ---- edges ----

    def _dispatch(self, state: AgentState) -> str:
        if state.get("blocked"):
            return "blocked"
        return state.get("route") or "rag"

    @staticmethod
    def _has_tool_calls(state: AgentState) -> str:
        return "tool_node" if state.get("tool_calls") else "aggregator"

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("router", self._router_node)
        graph.add_node("rag", self._rag_node)
        graph.add_node("sql", self._sql_node)
        graph.add_node("doc", self._doc_node)
        graph.add_node("web", self._web_node)
        graph.add_node("blocked", self._blocked_node)
        graph.add_node("tool_node", self._tool_node)
        graph.add_node("aggregator", self._aggregator_node)

        graph.add_edge(START, "router")
        graph.add_conditional_edges(
            "router",
            self._dispatch,
            {"rag": "rag", "sql": "sql", "doc": "doc", "web": "web", "blocked": "blocked"},
        )
        for node in ("rag", "sql", "doc", "web"):
            graph.add_conditional_edges(
                node, self._has_tool_calls, {"tool_node": "tool_node", "aggregator": "aggregator"}
            )
        graph.add_edge("blocked", "aggregator")
        graph.add_edge("tool_node", "aggregator")
        graph.add_edge("aggregator", END)

        return graph.compile(checkpointer=self.checkpointer)

    # ---- public API ----

    @staticmethod
    def _initial_state(
        query: str,
        file_path: Optional[str],
        user_id: str,
        session_id: str,
        chat_history: Optional[List[dict]],
    ) -> AgentState:
        return {
            "query": query,
            "file_path": file_path,
            "user_id": user_id,
            "session_id": session_id,
            "chat_history": chat_history or [],
        }

    @staticmethod
    def _config(session_id: str, user_id: str) -> dict:
        return {"configurable": {"thread_id": session_id or user_id or "default"}}

    @staticmethod
    def _to_response(final_state: AgentState) -> AgentResponse:
        route = RouteName(final_state["route"]) if final_state.get("route") else RouteName.UNKNOWN
        return AgentResponse(
            answer=final_state.get("answer", ""),
            citations=final_state.get("citations", []),
            route=route,
            latency_ms=final_state.get("latency_ms", 0.0),
            metadata={"sql_result": final_state.get("sql_result")} if final_state.get("sql_result") else {},
        )

    async def run(
        self,
        query: str,
        file_path: Optional[str] = None,
        user_id: str = "",
        session_id: str = "",
        chat_history: Optional[List[dict]] = None,
    ) -> AgentResponse:
        initial_state = self._initial_state(query, file_path, user_id, session_id, chat_history)
        final_state = await self.workflow.ainvoke(initial_state, config=self._config(session_id, user_id))
        return self._to_response(final_state)

    async def astream_events(
        self,
        query: str,
        user_id: str = "",
        session_id: str = "",
        chat_history: Optional[List[dict]] = None,
        file_path: Optional[str] = None,
    ):
        initial_state = self._initial_state(query, file_path, user_id, session_id, chat_history)
        config = self._config(session_id, user_id)
        async for event in self.workflow.astream_events(initial_state, config=config, version="v2"):
            yield event


# ---- module-level default graph, for direct `from src.agents.router
# import astream_events` usage outside of the FastAPI app (main.py
# builds and owns its own AgentGraph via a Redis-backed lifespan
# handler instead of this lazily-constructed singleton). ----

_default_graph: Optional[AgentGraph] = None
_default_graph_lock = asyncio.Lock()


async def _get_default_graph() -> AgentGraph:
    global _default_graph
    if _default_graph is not None:
        return _default_graph

    async with _default_graph_lock:
        if _default_graph is None:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver

            from src.document_processing import DocumentProcessor
            from src.hybrid_retrieval import HybridRetriever
            from src.vectorstore import VectorStore

            redis_url = os.getenv("REDIS_URL")
            if not redis_url:
                raise RuntimeError("REDIS_URL is not set; required for RedisSaver checkpointing")

            # Entered once and kept open for the lifetime of the process —
            # this module-level singleton has no shutdown hook of its own.
            checkpointer = await AsyncRedisSaver.from_conn_string(redis_url).__aenter__()
            await checkpointer.asetup()

            vector_store = VectorStore()
            retriever = HybridRetriever(vector_store)
            rag_agent = RAGAgent(retriever=retriever)
            doc_agent = DocAgent(document_processor=DocumentProcessor())
            database_url = os.getenv("DATABASE_URL")
            sql_agent = SQLAgent(database_url) if database_url else None

            _default_graph = AgentGraph(
                rag_agent=rag_agent,
                sql_agent=sql_agent,
                doc_agent=doc_agent,
                checkpointer=checkpointer,
            )
    return _default_graph


async def astream_events(
    query: str,
    user_id: str = "",
    session_id: str = "",
    chat_history: Optional[List[dict]] = None,
):
    """Stream graph events for `query` on the module-level default
    AgentGraph, checkpointed under `thread_id=session_id`.
    """

    graph = await _get_default_graph()
    async for event in graph.astream_events(query, user_id=user_id, session_id=session_id, chat_history=chat_history):
        yield event
