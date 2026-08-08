"""End-to-end test: document_processing -> vectorstore-like retrieval
-> reranking -> citation, wired together the same way main.py does,
but with a fake in-memory vector store so no external services (Chroma
persistence, embedding models) are required to run this in CI.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.agents.router import AgentGraph
from src.citation import build_citations, format_answer_with_citations, verify_citation_markers
from src.hybrid_retrieval import BM25Index
from src.utils.schemas import Document, RetrievedChunk, RouteName, SourceType


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


async def test_end_to_end_docx_ingest_retrieve_cite(document_processor, tmp_path, make_docx):
    """Same ingest -> retrieve -> cite pipeline as the .txt case above,
    but starting from a real .docx file -- proving DOCX ingestion isn't
    just a standalone parser but actually reaches the same
    indexing/retrieval path as every other supported document type.
    """

    path = make_docx(
        tmp_path / "policy.docx",
        heading="Refund Policy",
        paragraphs=[
            "Our refund policy allows returns within 30 days of purchase.",
            "Refunds are processed within 5 business days of receiving the item.",
        ],
    )

    documents, chunks = document_processor.process(path, chunk_size=200, chunk_overlap=20)
    assert chunks, "expected at least one chunk from a non-empty docx"

    vector_store = FakeVectorStore(chunks)
    bm25_index = BM25Index()
    bm25_index.build(chunks)

    from src.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(vector_store, bm25_index)
    results = await retriever.retrieve(["refund policy"], top_k=3)

    assert results, "expected the hybrid retriever to find relevant chunks from the docx"
    assert any(r.chunk.metadata.get("source_type") == "docx" for r in results)

    citations = build_citations(results)
    answer = format_answer_with_citations("Refunds must be requested within 30 days [1].", citations)

    assert "Sources:" in answer
    assert verify_citation_markers(answer, len(citations))
    assert citations[0].source == str(path)


async def test_full_agent_pipeline_sql_plots_sales_by_region(sales_sql_agent, app_module):
    """Full pipeline for a SQL-routed query: LangGraph router -> SQLAgent
    (NL->SQL against a real, if tiny, sqlite 'sales' table) -> the SSE
    chart-building helper main.py uses for a `plot ...` query, matching
    the shape a `/api/v1/agent?stream=true` request builds a `chart` event
    from (see main._sse_agent_stream).
    """
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value="SELECT region, SUM(amount) AS total FROM sales GROUP BY region")
    sql_agent = sales_sql_agent(llm=llm)

    graph = AgentGraph(rag_agent=MagicMock(), sql_agent=sql_agent, classifier_llm=MagicMock())
    response = await graph.run("plot sales by region", forced_route="sql")

    assert response.route == RouteName.SQL
    rows = response.metadata["sql_result"]
    assert {row["region"] for row in rows} == {"East", "West"}

    chart = app_module._build_chart(rows)
    assert chart is not None
    assert chart["data"][0]["type"] == "bar"


async def test_full_agent_pipeline_doc_answers_from_excel(tmp_path, document_processor):
    """Full pipeline for a doc-routed query over an uploaded Excel file:
    LangGraph router -> DocAgent -> DocumentProcessor's ExcelLoader,
    without needing the file pre-indexed in the vector store.
    """
    import openpyxl

    from src.agents.doc_agent import DocAgent

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sales"
    sheet.append(["region", "amount"])
    sheet.append(["East", 150])
    sheet.append(["West", 150])
    file_path = tmp_path / "sales.xlsx"
    workbook.save(file_path)

    reranker = MagicMock()
    reranker.rerank.side_effect = lambda query, candidates, top_k: candidates[:top_k]

    doc_agent = DocAgent(document_processor=document_processor, reranker=reranker)
    graph = AgentGraph(rag_agent=MagicMock(), doc_agent=doc_agent, classifier_llm=MagicMock())

    response = await graph.run("Summarize sales.xlsx", file_path=str(file_path), forced_route="doc")

    assert response.route == RouteName.DOC
    assert response.citations
    assert response.citations[0].sheet == "Sales"
