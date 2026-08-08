import pytest

from src.document_processing import DocumentProcessor
from src.parsers.ocr_parser import OCRDependencyError
from src.utils.schemas import Document, SourceType


@pytest.fixture
def processor(document_processor: DocumentProcessor) -> DocumentProcessor:
    return document_processor


@pytest.fixture
def ocr_deps_unavailable(processor):
    """Force the shared processor's OCRProcessor to believe neither
    tesseract nor poppler are installed, regardless of what's actually on
    the test machine's PATH -- so these tests deterministically exercise
    the "OCR dependencies missing" path. Restored afterwards since
    `processor` is a session-scoped fixture shared across the whole run.
    """

    ocr = processor.ocr_processor
    original = (ocr.tesseract_available, ocr.poppler_available)
    ocr.tesseract_available = False
    ocr.poppler_available = False
    yield
    ocr.tesseract_available, ocr.poppler_available = original


def test_chunk_document_respects_overlap(processor):
    document = Document(source_path="x.txt", source_type=SourceType.TEXT, text="a" * 2500)
    chunks = processor.chunk(document, chunk_size=1000, chunk_overlap=100)

    assert len(chunks) == 3
    assert all(len(c.text) <= 1000 for c in chunks)
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_chunk_document_empty_text_returns_no_chunks(processor):
    document = Document(source_path="x.txt", source_type=SourceType.TEXT, text="   ")
    assert processor.chunk(document) == []


def test_chunk_overlap_must_be_smaller_than_chunk_size(processor):
    document = Document(source_path="x.txt", source_type=SourceType.TEXT, text="hello world")
    with pytest.raises(ValueError):
        processor.chunk(document, chunk_size=100, chunk_overlap=100)


def test_load_documents_unsupported_extension(processor, tmp_path):
    bad_file = tmp_path / "file.exe"
    bad_file.write_bytes(b"binary")
    with pytest.raises(ValueError):
        processor.load(bad_file)


def test_load_documents_text_file(processor, tmp_path):
    text_file = tmp_path / "note.txt"
    text_file.write_text("hello world", encoding="utf-8")
    docs = processor.load(text_file)

    assert len(docs) == 1
    assert docs[0].source_type == SourceType.TEXT
    assert docs[0].text == "hello world"


# --- DOCX ingestion -----------------------------------------------------
#
# Regression coverage for: `.docx` had a dangling `SourceType.DOCX` enum
# member and a `python-docx` dependency declared in requirements.txt, but
# no parser and no dispatch branch in `DocumentProcessor.load()` -- every
# `.docx` file raised `ValueError: Unsupported file type: .docx`.


def test_load_documents_docx_file_with_paragraphs_and_table(processor, tmp_path, make_docx):
    path = make_docx(
        tmp_path / "policy.docx",
        heading="Refund Policy",
        paragraphs=["Our refund policy allows returns within 30 days of purchase."],
        table_rows=[["Region", "Days"], ["US", "30"]],
    )

    docs = processor.load(path)

    assert any(d.source_type == SourceType.DOCX for d in docs)
    assert any(d.source_type == SourceType.TABLE for d in docs)
    doc = next(d for d in docs if d.source_type == SourceType.DOCX)
    assert "Our refund policy allows returns within 30 days of purchase." in doc.text


def test_process_docx_file_produces_indexable_chunks(processor, tmp_path, make_docx):
    """`.docx` Documents must flow through `chunk()`/`process()` exactly
    like every other source type -- no special-casing needed since
    DocumentProcessor.chunk() is source-type agnostic.
    """

    path = make_docx(
        tmp_path / "policy.docx",
        paragraphs=["Our refund policy allows returns within 30 days of purchase." * 20],
    )

    documents, chunks = processor.process(path, chunk_size=200, chunk_overlap=20)

    assert documents
    assert chunks
    assert all(c.metadata["source_type"] == "docx" for c in chunks if c.metadata.get("source_type") != "table")


def test_load_documents_empty_docx_returns_no_documents(processor, tmp_path, make_docx):
    path = make_docx(tmp_path / "empty.docx")
    assert processor.load(path) == []


def test_load_documents_malformed_docx_raises_value_error(processor, tmp_path):
    path = tmp_path / "malformed.docx"
    path.write_bytes(b"not a real docx file")

    with pytest.raises(ValueError):
        processor.load(path)


def test_is_scanned_pdf_detects_short_text(processor):
    assert processor._is_scanned_pdf("short") is True
    assert processor._is_scanned_pdf("x" * 200) is False


def test_redact_pii_masks_email_when_enabled(processor):
    processor.mask_pii = True
    try:
        redacted, entities = processor._redact_pii("Contact me at alice@example.com for details.")
        assert entities
        assert "alice@example.com" not in redacted
    finally:
        processor.mask_pii = False


def test_redact_pii_noop_when_disabled(processor):
    text = "Contact me at alice@example.com for details."
    redacted, entities = processor._redact_pii(text)

    assert redacted == text
    assert entities == []


# --- OCR/Poppler/Ghostscript dependency handling ----------------------------
#
# Regression coverage for: PDF ingestion could crash on *ordinary*
# documents, not just intentionally-scanned ones. `_is_scanned_pdf` routes
# any PDF whose extracted text layer is under `SCANNED_PDF_CHAR_THRESHOLD`
# chars into `OCRProcessor.process_pdf`, which shells out to poppler via
# pdf2image. When poppler wasn't on PATH, that raised a raw, unhandled
# `pdf2image.exceptions.PDFInfoNotInstalledError` straight out of
# `DocumentProcessor.load()` -- for *any* short-but-real PDF, not only
# scanned ones.


def test_load_pdf_normal_document_ingests_without_ocr(processor, tmp_path, make_pdf_bytes, ocr_deps_unavailable):
    """A normal PDF with a real, sufficiently long text layer never
    touches the OCR path at all, so it must ingest fine even when OCR
    dependencies are completely unavailable.
    """

    pdf_path = tmp_path / "normal.pdf"
    pdf_path.write_bytes(make_pdf_bytes("This is a normal PDF with plenty of real extractable text content. " * 5))

    docs = processor.load(pdf_path)

    pdf_doc = next(d for d in docs if d.source_type == SourceType.PDF)
    assert "normal PDF" in pdf_doc.text
    assert "ocr_skipped" not in pdf_doc.metadata


def test_load_pdf_requiring_ocr_raises_clear_error_when_ocr_unavailable(
    processor, tmp_path, make_pdf_bytes, ocr_deps_unavailable
):
    """A PDF with no real text layer genuinely needs OCR. When OCR
    dependencies are unavailable there is nothing to ingest, so this must
    raise the clear, typed `OCRDependencyError` -- not crash with a raw
    pdf2image/pytesseract internal exception, and not silently produce an
    empty document that looks like a successful ingest.
    """

    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(make_pdf_bytes(""))

    with pytest.raises(OCRDependencyError):
        processor.load(pdf_path)


def test_load_pdf_short_text_falls_back_to_extracted_text_when_ocr_unavailable(
    processor, tmp_path, make_pdf_bytes, ocr_deps_unavailable
):
    """A PDF that merely trips the length heuristic (real text, just
    short) is *not* actually a scanned document. If OCR is unavailable,
    ingestion must fall back to the text layer that was already
    extracted instead of losing the document -- and must mark that
    clearly in metadata rather than pretending a real OCR pass happened.
    """

    pdf_path = tmp_path / "short.pdf"
    pdf_path.write_bytes(make_pdf_bytes("Short doc"))

    docs = processor.load(pdf_path)

    pdf_doc = next(d for d in docs if d.source_type == SourceType.PDF)
    assert "Short doc" in pdf_doc.text
    assert pdf_doc.metadata["ocr_skipped"] is True
    assert "ocr_error" in pdf_doc.metadata


def test_load_pdf_does_not_leak_raw_pdf2image_crash(processor, tmp_path, make_pdf_bytes, ocr_deps_unavailable):
    """Regression test for the reported crash: ingesting a short-text PDF
    with poppler unavailable used to propagate a raw, unhandled
    `pdf2image.exceptions.PDFInfoNotInstalledError` out of
    `DocumentProcessor.load()`. It must now be handled -- either via the
    text-layer fallback above or a clear `OCRDependencyError` -- but the
    raw third-party exception must never escape.
    """

    from pdf2image.exceptions import PDFInfoNotInstalledError

    pdf_path = tmp_path / "short.pdf"
    pdf_path.write_bytes(make_pdf_bytes("Short doc"))

    try:
        processor.load(pdf_path)
    except PDFInfoNotInstalledError:
        pytest.fail("raw pdf2image.exceptions.PDFInfoNotInstalledError leaked out of DocumentProcessor.load()")
    except OCRDependencyError:
        pass  # acceptable: a clear, typed error instead of an unhandled crash
