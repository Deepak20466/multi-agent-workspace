"""LangGraph router: classifies an incoming query and dispatches it to
the RAG, SQL, Doc, or Web agent, with guardrails applied on the way in
and out.

The graph is intentionally small (classify -> guard -> dispatch -> one
of four agent nodes -> END) so each node stays testable in isolation;
see tests/test_agents.py. State is threaded through as `AgentState`
(src/utils/schemas.py) so any node can be inspected/replayed independent
of the final AgentResponse shape returned to callers.
"""

from __future__ import annotations

import re

from langgraph.graph import StateGraph, END

from src.agents.doc_agent import DocAgent
from src.agents.rag_agent import RAGAgent
from src.agents.sql_agent import SQLAgent
from src.agents.web_agent import WebAgent
from src.guardrails import PIIGuard, detect_prompt_injection
from src.telemetry import log_event
from src.utils.schemas import AgentResponse, AgentState, RouteDecision, RouteName

_SQL_KEYWORDS = re.compile(r"\b(select|count|sum|average|how many|top \d+|group by)\b", re.IGNORECASE)
_DOC_KEYWORDS = re.compile(r"\b(this (file|document|attachment|upload)|attached)\b", re.IGNORECASE)
_WEB_KEYWORDS = re.compile(r"\b(latest|today|current|news|right now|as of \d{4})\b", re.IGNORECASE)


def classify_query(query: str, has_file: bool) -> RouteDecision:
    """Rule-based router. Deliberately simple/explainable; swap for an
    LLM-based classifier node if routing accuracy needs to improve —
    the AgentState/RouteDecision contract stays the same either way.
    """

    if has_file:
        return RouteDecision(route=RouteName.DOC, confidence=0.95, rationale="a file was attached to the request")
    if _SQL_KEYWORDS.search(query):
        return RouteDecision(route=RouteName.SQL, confidence=0.8, rationale="query looks like a structured/aggregate lookup")
    if _WEB_KEYWORDS.search(query):
        return RouteDecision(route=RouteName.WEB, confidence=0.75, rationale="query asks about current/live information")
    if _DOC_KEYWORDS.search(query):
        return RouteDecision(route=RouteName.DOC, confidence=0.7, rationale="query references an attached document")
    return RouteDecision(route=RouteName.RAG, confidence=0.6, rationale="default: search local knowledge base")


def _apply_response(state: AgentState, response: AgentResponse) -> AgentState:
    return {
        **state,
        "answer": response.answer,
        "citations": response.citations,
        "documents": [c.model_dump() for c in response.citations],
        "latency_ms": response.latency_ms,
    }


class AgentRouter:
    """Builds and runs the LangGraph StateGraph wiring router -> agents."""

    def __init__(
        self,
        rag_agent: RAGAgent,
        sql_agent: SQLAgent | None = None,
        doc_agent: DocAgent | None = None,
        web_agent: WebAgent | None = None,
        pii_guard: PIIGuard | None = None,
    ):
        self.rag_agent = rag_agent
        self.sql_agent = sql_agent
        self.doc_agent = doc_agent
        self.web_agent = web_agent
        self.pii_guard = pii_guard or PIIGuard()
        self.graph = self._build_graph()

    def _guard_node(self, state: AgentState) -> AgentState:
        query = state["query"]
        if detect_prompt_injection(query):
            log_event("router_blocked_injection", query=query)
            return {**state, "blocked": True, "block_reason": "potential prompt injection detected"}
        safe_query, _ = self.pii_guard.anonymize(query)
        return {**state, "query": safe_query, "blocked": False}

    def _route_node(self, state: AgentState) -> AgentState:
        decision = classify_query(state["query"], has_file=bool(state.get("file_path")))
        log_event("router_decision", route=decision.route.value, confidence=decision.confidence)
        return {**state, "route": decision.route.value}

    async def _rag_node(self, state: AgentState) -> AgentState:
        response = await self.rag_agent.answer(state["query"])
        return _apply_response(state, response)

    def _sql_node(self, state: AgentState) -> AgentState:
        if self.sql_agent is None:
            return _apply_response(state, _unavailable_response(RouteName.SQL))
        response = self.sql_agent.run_query(state["query"])
        new_state = _apply_response(state, response)
        new_state["sql_result"] = response.metadata.get("rows")
        return new_state

    def _doc_node(self, state: AgentState) -> AgentState:
        if self.doc_agent is None or not state.get("file_path"):
            return _apply_response(state, _unavailable_response(RouteName.DOC))
        response = self.doc_agent.answer_from_file(state["file_path"], state["query"])
        return _apply_response(state, response)

    def _web_node(self, state: AgentState) -> AgentState:
        if self.web_agent is None:
            return _apply_response(state, _unavailable_response(RouteName.WEB))
        response = self.web_agent.answer(state["query"])
        return _apply_response(state, response)

    def _blocked_node(self, state: AgentState) -> AgentState:
        response = AgentResponse(
            answer=f"Request blocked: {state.get('block_reason', 'unsafe input')}",
            route=RouteName.UNKNOWN,
        )
        return _apply_response(state, response)

    def _dispatch(self, state: AgentState) -> str:
        if state.get("blocked"):
            return "blocked"
        return state["route"]

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("guard", self._guard_node)
        graph.add_node("route", self._route_node)
        graph.add_node("rag", self._rag_node)
        graph.add_node("sql", self._sql_node)
        graph.add_node("doc", self._doc_node)
        graph.add_node("web", self._web_node)
        graph.add_node("blocked", self._blocked_node)

        graph.set_entry_point("guard")
        graph.add_conditional_edges(
            "guard",
            lambda s: "blocked" if s.get("blocked") else "route",
            {"blocked": "blocked", "route": "route"},
        )
        graph.add_conditional_edges(
            "route",
            self._dispatch,
            {"rag": "rag", "sql": "sql", "doc": "doc", "web": "web", "blocked": "blocked"},
        )
        for node in ("rag", "sql", "doc", "web", "blocked"):
            graph.add_edge(node, END)

        return graph.compile()

    async def run(
        self,
        query: str,
        file_path: str | None = None,
        user_id: str = "",
        session_id: str = "",
        chat_history: list[dict] | None = None,
    ) -> AgentResponse:
        initial_state: AgentState = {
            "query": query,
            "file_path": file_path,
            "user_id": user_id,
            "session_id": session_id,
            "chat_history": chat_history or [],
        }
        final_state = await self.graph.ainvoke(initial_state)

        route = RouteName(final_state["route"]) if final_state.get("route") else RouteName.UNKNOWN
        return AgentResponse(
            answer=final_state.get("answer", ""),
            citations=final_state.get("citations", []),
            route=route,
            latency_ms=final_state.get("latency_ms", 0.0),
            metadata={"sql_result": final_state.get("sql_result")} if final_state.get("sql_result") else {},
        )


def _unavailable_response(route: RouteName) -> AgentResponse:
    return AgentResponse(answer=f"The '{route.value}' agent is not configured.", route=route)
