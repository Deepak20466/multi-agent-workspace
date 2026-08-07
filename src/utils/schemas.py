"""Shared pydantic data contracts used across agents, retrieval, and eval."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class SourceType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    EXCEL = "excel"
    IMAGE_OCR = "image_ocr"
    TABLE = "table"
    WEB = "web"
    SQL = "sql"
    TEXT = "text"


class Document(BaseModel):
    """A raw ingested document prior to chunking."""

    doc_id: str = Field(default_factory=_uuid)
    source_path: str
    source_type: SourceType
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)


class Chunk(BaseModel):
    """A chunk produced from a Document, ready for embedding."""

    chunk_id: str = Field(default_factory=_uuid)
    doc_id: str
    text: str
    chunk_index: int
    page_number: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk returned from retrieval, carrying per-strategy scores."""

    chunk: Chunk
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None

    @property
    def final_score(self) -> float:
        if self.rerank_score is not None:
            return self.rerank_score
        if self.rrf_score is not None:
            return self.rrf_score
        return self.vector_score or 0.0


class Citation(BaseModel):
    """A grounded citation attached to an agent answer."""

    id: int
    type: Literal["vector", "sql", "web", "doc", "table", "ocr"]
    source: str
    page: Optional[int] = None
    sheet: Optional[str] = None
    quote: Optional[str] = None
    score: float = 0.0


class PIIEntity(BaseModel):
    """A single PII detection from the guardrails layer."""

    entity_type: str
    text: str
    start: int
    end: int
    score: float


class RouteName(str, Enum):
    RAG = "rag"
    SQL = "sql"
    DOC = "doc"
    WEB = "web"
    UNKNOWN = "unknown"


class RouteDecision(BaseModel):
    """Output of the LangGraph router node."""

    route: RouteName
    confidence: float
    rationale: str = ""


class AgentResponse(BaseModel):
    """Normalized response shape returned by every agent."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    route: RouteName
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentState(TypedDict, total=False):
    """LangGraph state schema threaded through the router graph."""

    query: str
    user_id: str
    session_id: str
    chat_history: List[dict]
    route: Literal["rag", "sql", "doc", "web"]
    documents: List[Any]
    sql_result: Optional[List[dict]]
    file_path: Optional[str]
    answer: str
    citations: List[Citation]
    latency_ms: float
    blocked: bool
    block_reason: str
