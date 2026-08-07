from src.citation import build_citations, format_answer_with_citations, verify_citation_markers
from src.utils.schemas import Chunk, RetrievedChunk


def _retrieved(chunk_id: str, text: str, source_path: str = "doc.pdf", page: int | None = 1) -> RetrievedChunk:
    chunk = Chunk(chunk_id=chunk_id, doc_id="d1", text=text, chunk_index=0, page_number=page, metadata={"source_path": source_path})
    return RetrievedChunk(chunk=chunk, vector_score=0.8)


def test_build_citations_truncates_quote():
    long_text = "x" * 500
    citations = build_citations([_retrieved("1", long_text)], max_quote_chars=100)

    assert len(citations) == 1
    assert len(citations[0].quote) == 100
    assert citations[0].page == 1


def test_format_answer_with_citations_appends_source_list():
    citations = build_citations([_retrieved("1", "hello world")])
    formatted = format_answer_with_citations("The answer is 42 [1].", citations)

    assert "Sources:" in formatted
    assert "doc.pdf" in formatted


def test_format_answer_with_no_citations_returns_answer_unchanged():
    assert format_answer_with_citations("plain answer", []) == "plain answer"


def test_verify_citation_markers_detects_out_of_range_marker():
    assert verify_citation_markers("see [1] and [2]", n_citations=2) is True
    assert verify_citation_markers("see [1] and [5]", n_citations=2) is False


def test_verify_citation_markers_no_markers_is_valid():
    assert verify_citation_markers("no citations here", n_citations=0) is True
