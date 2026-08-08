"""MCP Server v3.1 — exposes the multi-agent workspace's RAG, SQL, Doc, and
Web agents as MCP tools over stdio (e.g. for Claude Desktop).

Wires VectorStore -> HybridRetriever -> RAGAgent plus the SQL/Doc/Web
agents the same way main.py does, and applies the guardrails input/output
checks (prompt-injection detection + PII anonymization) around every tool
call since these tools are called directly rather than through the
LangGraph router, which normally guards each request first.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

from mcp.server.mcpserver import MCPServer as FastMCP

from src.agents.doc_agent import DocAgent
from src.agents.rag_agent import RAGAgent
from src.agents.sql_agent import SQLAgent, UnsafeSQLError
from src.agents.web_agent import WebAgent
from src.config import load_config
from src.document_processing import DocumentProcessor
from src.guardrails import PIIGuard, guard_input, guard_output
from src.hybrid_retrieval import HybridRetriever
from src.utils.retry_handler import CircuitOpenError, RetryableError
from src.utils.schemas import Citation
from src.vectorstore import VectorStore

mcp = FastMCP("multi-agent-workspace")

_config = load_config()
_pii_guard = PIIGuard()
_vector_store = VectorStore()
_retriever = HybridRetriever(_vector_store)
# Same llm_backend/ollama_model/ollama_base_url config main.py wires its
# RAGAgent/DocAgent with -- MCP tools previously left these unset, which
# silently left both agents on their llm=None stub-answer path even when
# config.yaml configures a real (e.g. Ollama) backend.
_rag_agent = RAGAgent(
    retriever=_retriever,
    pii_guard=_pii_guard,
    llm_backend=_config.agents.llm_backend,
    ollama_model=_config.agents.ollama_model,
    ollama_base_url=_config.agents.ollama_base_url,
)
_doc_agent = DocAgent(
    document_processor=DocumentProcessor(),
    llm_backend=_config.agents.llm_backend,
    ollama_model=_config.agents.ollama_model,
    ollama_base_url=_config.agents.ollama_base_url,
)
_web_agent = WebAgent()

_database_url = os.getenv("DATABASE_URL")
_sql_agent = SQLAgent(_database_url) if _database_url else None

_BLOCKED_MESSAGE = "Request blocked: potential prompt injection detected."


def _format_citations(citations: list[Citation]) -> str:
    if not citations:
        return ""

    lines = []
    for c in citations:
        location = c.source
        if c.page:
            location += f", p.{c.page}"
        if c.sheet:
            location += f", sheet '{c.sheet}'"
        lines.append(f"[{c.id}] {location}")
    return "\n\nCitations:\n" + "\n".join(lines)


@mcp.tool()
async def search_docs(query: str) -> str:
    """Search the indexed document corpus and answer the question with
    citations. RAGAgent guards its own input, so only the output is
    re-checked here.
    """

    response = await _rag_agent.answer(query)
    answer = guard_output(response.answer, _pii_guard)
    return answer + _format_citations(response.citations)


@mcp.tool()
async def query_sql(question: str) -> str:
    """Answer a natural-language question by generating and running a
    read-only SQL query against the configured database. Returns the
    generated SQL, a preview of the result rows, and chart-ready row data.
    """

    if _sql_agent is None:
        return "SQL agent is not configured: set DATABASE_URL."

    safe_question, report = guard_input(question, _pii_guard)
    if report["injection_detected"]:
        return _BLOCKED_MESSAGE

    try:
        response = await _sql_agent.answer(safe_question)
    except UnsafeSQLError as exc:
        return f"Couldn't safely answer that as SQL: {exc}"
    except (RetryableError, CircuitOpenError, SQLAlchemyError) as exc:
        return f"SQL agent unavailable: {exc}"

    answer = guard_output(response.answer, _pii_guard)
    sql = response.metadata.get("sql", "")
    rows = response.metadata.get("rows", [])
    chart_data = json.dumps(rows[:20], default=str)
    return f"SQL:\n{sql}\n\n{answer}\n\nChart data (rows):\n{chart_data}"


@mcp.tool()
async def extract_document(file_path: str, query: str) -> str:
    """Extract and answer a question from a specific document file (PDF,
    Excel, image, or text) without requiring it to be pre-indexed.
    """

    safe_query, report = guard_input(query, _pii_guard)
    if report["injection_detected"]:
        return _BLOCKED_MESSAGE

    try:
        response = _doc_agent.answer_from_file(file_path, safe_query)
    except FileNotFoundError as exc:
        return f"Couldn't find a file to answer from: {exc}"

    answer = guard_output(response.answer, _pii_guard)
    return answer + _format_citations(response.citations)


@mcp.tool()
async def web_search(query: str) -> str:
    """Search the live web and answer the question with citations."""

    safe_query, report = guard_input(query, _pii_guard)
    if report["injection_detected"]:
        return _BLOCKED_MESSAGE

    response = await _web_agent.answer(safe_query)
    answer = guard_output(response.answer, _pii_guard)
    return answer + _format_citations(response.citations)


if __name__ == "__main__":
    mcp.run(transport="stdio")
