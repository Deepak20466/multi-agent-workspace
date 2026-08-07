"""Document agent: ingests a raw file (Excel/OCR/PDF table/text) on
demand and answers questions grounded in that single file, without
needing it pre-indexed in the vector store.

Useful for "answer from this attachment" style requests, as opposed
to the RAGAgent which searches the whole persisted corpus.
"""

from __future__ import annotations

import time

from src.document_processing import DocumentProcessor
from src.reranking import Reranker
from src.citation import build_citations, format_answer_with_citations
from src.telemetry import traced_call
from src.utils.schemas import AgentResponse, RetrievedChunk, RouteName


class DocAgent:
    def __init__(
        self,
        reranker: Reranker | None = None,
        llm=None,
        document_processor: DocumentProcessor | None = None,
    ):
        self.reranker = reranker or Reranker()
        self.llm = llm
        self.document_processor = document_processor or DocumentProcessor()

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

    def answer_from_file(self, file_path: str, query: str, k: int = 5) -> AgentResponse:
        start = time.perf_counter()
        with traced_call("doc"):
            _, chunks = self.document_processor.process(file_path)
            candidates = [RetrievedChunk(chunk=c, vector_score=1.0) for c in chunks]
            top = self.reranker.rerank(query, candidates, top_k=k) if candidates else []
            citations = build_citations(top, default_type="doc")
            raw_answer = self._generate(query, [c.chunk.text for c in top])
            answer_text = format_answer_with_citations(raw_answer, citations)

        return AgentResponse(
            answer=answer_text,
            citations=citations,
            route=RouteName.DOC,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={"file_path": file_path, "n_chunks": len(chunks)},
        )
