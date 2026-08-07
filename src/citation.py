"""Builds grounded Citation objects from the chunks actually used to
answer a query, and formats them into inline markers + a source list
so every claim in an answer can be traced back to a source document.
"""

from __future__ import annotations

import re
from typing import Literal

from src.utils.schemas import Citation, RetrievedChunk

CitationType = Literal["vector", "sql", "web", "doc", "table", "ocr"]

_SOURCE_TYPE_TO_CITATION_TYPE: dict[str, CitationType] = {
    "table": "table",
    "image_ocr": "ocr",
}


def _infer_citation_type(chunk_metadata: dict, default: CitationType) -> CitationType:
    source_type = chunk_metadata.get("source_type")
    return _SOURCE_TYPE_TO_CITATION_TYPE.get(source_type, default)


def build_citations(
    retrieved_chunks: list[RetrievedChunk],
    default_type: CitationType = "vector",
    max_quote_chars: int = 240,
) -> list[Citation]:
    citations: list[Citation] = []
    for i, retrieved in enumerate(retrieved_chunks, start=1):
        chunk = retrieved.chunk
        quote = chunk.text[:max_quote_chars].strip()
        citations.append(
            Citation(
                id=i,
                type=_infer_citation_type(chunk.metadata, default_type),
                source=chunk.metadata.get("source_path", "unknown"),
                page=chunk.page_number,
                sheet=chunk.metadata.get("sheet_name"),
                quote=quote,
                score=retrieved.final_score,
            )
        )
    return citations


def format_answer_with_citations(answer: str, citations: list[Citation]) -> str:
    """Appends a numbered source list and inline [n] markers are assumed to
    already be present in `answer` (added by the generating LLM prompt).
    """

    if not citations:
        return answer

    lines = [answer.strip(), "", "Sources:"]
    for citation in citations:
        location = f", p.{citation.page}" if citation.page else ""
        if citation.sheet:
            location += f", sheet '{citation.sheet}'"
        label = f"{citation.title} ({citation.source})" if citation.title else citation.source
        lines.append(f"[{citation.id}] {label}{location} — \"{citation.quote}\"")
    return "\n".join(lines)


def verify_citation_markers(answer: str, n_citations: int) -> bool:
    """Sanity check that every [n] marker used in the answer text refers to
    an actual citation index, catching hallucinated citation numbers.
    """

    markers = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    return all(1 <= m <= n_citations for m in markers)
