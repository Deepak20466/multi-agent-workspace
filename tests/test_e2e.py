"""End-to-end test: document_processing -> vectorstore-like retrieval
-> reranking -> citation, wired together the same way main.py does,
but with a fake in-memory vector store so no external services (Chroma
persistence, embedding models) are required to run this in CI.
"""

from __future__ import annotations

from src.citation import build_citations, format_answer_with_citations, verify_citation_markers
from src.hybrid_retrieval import BM25Index
from src.utils.schemas import Document, RetrievedChunk, SourceType


class FakeVectorStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def similarity_search(self, query, k=10, where=None):
        matches = [c for c in self.chunks if any(w in c.text.lower() for w in query.lower().split())]
        return [RetrievedChunk(chunk=c, vector_score=0.9 - 0.1 * i) for i, c in enumerate(matches[:k])]


async def test_end_to_end_ingest_retrieve_cite(document_processor):
    document = Document(
        source_path="policy.txt",
        source_type=SourceType.TEXT,
        text=(
            "Our refund policy allows returns within 30 days of purchase. "
            "Refunds are processed within 5 business days of receiving the item."
        ),
    )
    chunks = document_processor.chunk(document, chunk_size=200, chunk_overlap=20)
    assert chunks, "expected at least one chunk from a non-empty document"

    vector_store = FakeVectorStore(chunks)
    bm25_index = BM25Index()
    bm25_index.build(chunks)

    from src.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(vector_store, bm25_index)
    results = await retriever.retrieve(["refund policy"], top_k=3)

    assert results, "expected the hybrid retriever to find relevant chunks"

    citations = build_citations(results)
    answer = format_answer_with_citations("Refunds must be requested within 30 days [1].", citations)

    assert "Sources:" in answer
    assert verify_citation_markers(answer, len(citations))
