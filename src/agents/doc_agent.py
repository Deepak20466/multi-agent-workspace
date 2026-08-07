"""Document agent: ingests a raw file (Excel/OCR/PDF table/text) on
demand and answers questions grounded in that single file, without
needing it pre-indexed in the vector store.

Useful for "answer from this attachment" style requests, as opposed
to the RAGAgent which searches the whole persisted corpus. When no
explicit file path is supplied, the agent falls back to regex-parsing
a filename (*.pdf/.xlsx/.png/...) out of the query text itself, so
"summarize invoice.pdf" works without a separate upload step.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

from src.document_processing import (
    DocumentProcessor,
    EXCEL_EXTENSIONS,
    IMAGE_EXTENSIONS,
    PDF_EXTENSIONS,
    TEXT_EXTENSIONS,
)
from src.reranking import Reranker
from src.citation import build_citations, format_answer_with_citations
from src.telemetry import traced_call
from src.utils.schemas import AgentResponse, RetrievedChunk, RouteName

_SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS | EXCEL_EXTENSIONS | PDF_EXTENSIONS
_EXT_PATTERN = "|".join(re.escape(ext.lstrip(".")) for ext in sorted(_SUPPORTED_EXTENSIONS))
_FILE_REF_RE = re.compile(rf"[\w./\\-]+\.(?:{_EXT_PATTERN})\b", re.IGNORECASE)

DEFAULT_UPLOAD_DIR = Path("data/uploads")


def extract_file_refs(text: str) -> list[str]:
    """Return every filename-with-supported-extension mentioned in `text`,
    in order of appearance (e.g. "invoice.pdf", "Q3 report.xlsx").
    """

    return _FILE_REF_RE.findall(text)


class DocAgent:
    def __init__(
        self,
        reranker: Reranker | None = None,
        llm=None,
        document_processor: DocumentProcessor | None = None,
        upload_dir: Path | str = DEFAULT_UPLOAD_DIR,
    ):
        self.reranker = reranker or Reranker()
        self.llm = llm
        self.document_processor = document_processor or DocumentProcessor()
        self.upload_dir = Path(upload_dir)

    def resolve_file_path(self, query: str, file_path: Optional[str] = None) -> str:
        """Return `file_path` if given, otherwise resolve a file
        reference out of `query`'s text (checking the path as given,
        then `upload_dir/<basename>`). Raises FileNotFoundError with a
        clear message when nothing can be resolved.
        """

        if file_path:
            return file_path

        refs = extract_file_refs(query)
        if not refs:
            raise FileNotFoundError(f"no file reference found in query: {query!r}")

        ref_path = Path(refs[0])
        if ref_path.exists():
            return str(ref_path)

        candidate = self.upload_dir / ref_path.name
        if candidate.exists():
            return str(candidate)

        raise FileNotFoundError(f"referenced file not found: {refs[0]!r}")

    def _generate(self, query: str, context_chunks: list[str]) -> str:
        if self.llm is None:
            joined = "\n---\n".join(context_chunks)
            return f"From the uploaded document:\n{joined}\n\nAnswer to '{query}' [1]"
        prompt = (
            "Answer using only the numbered excerpts from the uploaded document. "
            "Cite as [n].\n\n"
            + "\n".join(f"[{i}] {c}" for i, c in enumerate(context_chunks, start=1))
            + f"\n\nQuestion: {query}"
        )
        return self.llm.invoke(prompt)

    @staticmethod
    def _to_markdown_table(top: list[RetrievedChunk], citations: list) -> str:
        if not top:
            return ""

        rows = ["| # | Excerpt | Source |", "|---|---|---|"]
        for i, (retrieved, citation) in enumerate(zip(top, citations), start=1):
            excerpt = retrieved.chunk.text[:200].replace("|", "\\|").replace("\n", " ").strip()
            location = citation.source
            if citation.page:
                location += f", p.{citation.page}"
            if citation.sheet:
                location += f", sheet '{citation.sheet}'"
            rows.append(f"| {i} | {excerpt} | {location} |")
        return "\n".join(rows)

    def answer_from_file(self, file_path: Optional[str], query: str, k: int = 5) -> AgentResponse:
        start = time.perf_counter()
        with traced_call("doc"):
            resolved_path = self.resolve_file_path(query, file_path)
            _, chunks = self.document_processor.process(resolved_path)
            candidates = [RetrievedChunk(chunk=c, vector_score=1.0) for c in chunks]
            top = self.reranker.rerank(query, candidates, top_k=k) if candidates else []
            citations = build_citations(top, default_type="doc")
            raw_answer = self._generate(query, [c.chunk.text for c in top])
            answer_text = format_answer_with_citations(raw_answer, citations)

            table = self._to_markdown_table(top, citations)
            if table:
                answer_text = f"{answer_text}\n\n{table}"

        return AgentResponse(
            answer=answer_text,
            citations=citations,
            route=RouteName.DOC,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={"file_path": resolved_path, "n_chunks": len(chunks)},
        )
