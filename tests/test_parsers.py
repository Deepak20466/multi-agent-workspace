import time
from unittest.mock import patch

import openpyxl
import pytest

from src.parsers.docx_parser import DocxLoader, _table_to_text as _docx_table_to_text
from src.parsers.excel_parser import ExcelLoader
from src.parsers.ocr_parser import OCRDependencyError, OCRProcessor
from src.parsers.table_parser import TableExtractor
from src.utils.schemas import SourceType


def test_excel_loader_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ExcelLoader().load(tmp_path / "missing.xlsx")


def test_excel_loader_creates_one_document_per_sheet(tmp_path):
    workbook = openpyxl.Workbook()
    sheet1 = workbook.active
    sheet1.title = "Sheet1"
    sheet1.append(["name", "age"])
    sheet1.append(["Alice", 30])
    sheet2 = workbook.create_sheet("Sheet2")
    sheet2.append(["city"])
    sheet2.append(["Berlin"])

    file_path = tmp_path / "book.xlsx"
    workbook.save(file_path)

    documents = ExcelLoader().load(file_path)

    assert len(documents) == 2
    assert all(d.source_type == SourceType.EXCEL for d in documents)
    assert "Alice" in documents[0].text
    assert documents[0].metadata["sheet_name"] == "Sheet1"


# --- DocxLoader ---------------------------------------------------------


def test_docx_loader_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DocxLoader().load(tmp_path / "missing.docx")


def test_docx_loader_extracts_paragraphs_with_headings_and_metadata(tmp_path, make_docx):
    path = make_docx(
        tmp_path / "policy.docx",
        heading="Refund Policy",
        heading_level=1,
        paragraphs=[
            "Our refund policy allows returns within 30 days of purchase.",
            "Refunds are processed within 5 business days.",
        ],
        title="Refund Policy Doc",
        author="QA Team",
    )

    documents = DocxLoader().load(path)

    assert len(documents) == 1
    document = documents[0]
    assert document.source_type == SourceType.DOCX
    assert document.text.startswith("# Refund Policy")
    assert "Our refund policy allows returns within 30 days of purchase." in document.text
    assert "Refunds are processed within 5 business days." in document.text
    assert document.metadata["n_paragraphs"] == 3  # heading + 2 paragraphs
    assert document.metadata["title"] == "Refund Policy Doc"
    assert document.metadata["author"] == "QA Team"


def test_docx_loader_extracts_tables_as_separate_documents(tmp_path, make_docx):
    path = make_docx(
        tmp_path / "with_table.docx",
        paragraphs=["See the table below for allowed return windows by region."],
        table_rows=[["Region", "Days"], ["US", "30"], ["EU", "14"]],
    )

    documents = DocxLoader().load(path)

    paragraph_docs = [d for d in documents if d.source_type == SourceType.DOCX]
    table_docs = [d for d in documents if d.source_type == SourceType.TABLE]

    assert len(paragraph_docs) == 1
    assert len(table_docs) == 1
    assert table_docs[0].metadata["extractor"] == "python-docx"
    assert "Region: US" in table_docs[0].text
    assert "Days: 30" in table_docs[0].text
    assert "Region: EU" in table_docs[0].text


def test_docx_loader_empty_docx_returns_no_documents(tmp_path, make_docx):
    path = make_docx(tmp_path / "empty.docx")

    assert DocxLoader().load(path) == []


def test_docx_loader_malformed_docx_raises_value_error(tmp_path):
    path = tmp_path / "malformed.docx"
    path.write_bytes(b"this is not a real docx file, just plain text")

    with pytest.raises(ValueError, match="(?i)not a valid docx"):
        DocxLoader().load(path)


def test_docx_loader_empty_file_raises_value_error(tmp_path):
    """A zero-byte .docx (e.g. an interrupted upload) hits the same
    PackageNotFoundError-not-a-zip path as other malformed content.
    """

    path = tmp_path / "zero_bytes.docx"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="(?i)not a valid docx"):
        DocxLoader().load(path)


def test_docx_table_to_text_renders_rows():
    class _FakeCell:
        def __init__(self, text):
            self.text = text

    class _FakeRow:
        def __init__(self, cells):
            self.cells = [_FakeCell(c) for c in cells]

    class _FakeTable:
        def __init__(self, rows):
            self.rows = [_FakeRow(r) for r in rows]

    table = _FakeTable([["name", "age"], ["Alice", "30"], ["Bob", "25"]])
    text = _docx_table_to_text(table)

    assert "name: Alice" in text
    assert "name: Bob" in text


def test_docx_table_to_text_empty_table_returns_empty_string():
    class _FakeTable:
        rows = []

    assert _docx_table_to_text(_FakeTable()) == ""


def test_ocr_processor_process_image_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        OCRProcessor().process_image(tmp_path / "missing.png")


def test_ocr_processor_process_image_returns_document(tmp_path):
    from PIL import Image

    image_path = tmp_path / "img.png"
    Image.new("RGB", (10, 10), color="white").save(image_path)

    with patch.object(OCRProcessor, "_ocr_image", return_value="hello world"):
        document = OCRProcessor(lang="eng", dpi=300).process_image(image_path)

    assert document.text == "hello world"
    assert document.source_type == SourceType.IMAGE_OCR
    assert document.metadata["lang"] == "eng"


def test_ocr_processor_warns_when_tesseract_missing():
    with patch("src.parsers.ocr_parser.shutil.which", return_value=None):
        with patch("src.parsers.ocr_parser.logger") as mock_logger:
            OCRProcessor()
            assert mock_logger.warning.called


def test_ocr_processor_process_image_raises_clear_error_when_tesseract_missing(tmp_path):
    """Missing tesseract must surface as the typed `OCRDependencyError`,
    not the raw `pytesseract.TesseractNotFoundError`, and must fail fast
    rather than retrying a binary that isn't coming back mid-retry.
    """

    from PIL import Image

    image_path = tmp_path / "img.png"
    Image.new("RGB", (10, 10), color="white").save(image_path)

    with patch("src.parsers.ocr_parser.shutil.which", return_value=None):
        processor = OCRProcessor()

    start = time.perf_counter()
    with pytest.raises(OCRDependencyError):
        processor.process_image(image_path)
    elapsed = time.perf_counter() - start

    # with_retry's exponential backoff (~5 attempts, up to 10s each) would
    # take many seconds if this were still retried; a missing binary must
    # be reported immediately instead.
    assert elapsed < 2.0


def test_ocr_processor_process_pdf_raises_clear_error_when_poppler_missing(tmp_path, make_pdf_bytes):
    """The reported crash: `OCRProcessor.process_pdf` used to let
    pdf2image's raw `PDFInfoNotInstalledError` escape unhandled when
    poppler wasn't on PATH. It must now raise the typed, clear
    `OCRDependencyError` instead.
    """

    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(make_pdf_bytes("irrelevant"))

    with patch("src.parsers.ocr_parser.shutil.which", return_value=None):
        processor = OCRProcessor()
        with pytest.raises(OCRDependencyError):
            processor.process_pdf(pdf_path)


def test_ocr_processor_reports_dependency_availability(tmp_path):
    with patch("src.parsers.ocr_parser.shutil.which", return_value=None):
        processor = OCRProcessor()
    assert processor.tesseract_available is False
    assert processor.poppler_available is False


def test_table_extractor_table_to_text_renders_rows():
    table = [["name", "age"], ["Alice", "30"], ["Bob", "25"]]
    text = TableExtractor._table_to_text(table)
    assert "name: Alice" in text
    assert "name: Bob" in text


def test_table_extractor_table_to_text_empty_table_returns_empty_string():
    assert TableExtractor._table_to_text([]) == ""


def test_table_extractor_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TableExtractor().extract(tmp_path / "missing.pdf")


def test_table_extractor_extracts_from_real_pdf_without_crashing(tmp_path, make_pdf_bytes):
    """End-to-end (no mocking): running the real camelot/pdfplumber code
    path against an ordinary generated PDF must never crash ingestion,
    regardless of which rasterization backends happen to be installed in
    this environment.
    """

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(make_pdf_bytes("Just a normal document with no tables in it at all."))

    documents = TableExtractor().extract(pdf_path)

    assert isinstance(documents, list)


def test_table_extractor_falls_back_to_pdfplumber_on_ghostscript_error(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    extractor = TableExtractor()
    with patch.object(extractor, "_extract_with_camelot", side_effect=RuntimeError("Ghostscript is not installed")):
        with patch.object(extractor, "_extract_with_pdfplumber", return_value=[]) as mock_fallback:
            result = extractor.extract(pdf_path)

    mock_fallback.assert_called_once()
    assert result == []


def test_table_extractor_falls_back_to_pdfplumber_on_camelot_image_conversion_error(tmp_path):
    """Regression test: camelot's actual failure mode when none of its
    rasterization backends (pdfium/poppler/ghostscript) are usable is
    `camelot.backends.image_conversion.ImageConversionError`, a
    `ValueError` subclass -- *not* the `RuntimeError` the old except
    clause pattern-matched on. That mismatch meant the "fall back to
    pdfplumber" path was dead code and the real error crashed ingestion.
    A bare `OSError` (what the ghostscript/poppler backends raise before
    camelot wraps it) must also be handled.
    """

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    extractor = TableExtractor()
    for exc in (
        ValueError("Image conversion failed with image conversion backend 'ghostscript'"),
        OSError("Ghostscript is not installed."),
    ):
        with patch.object(extractor, "_extract_with_camelot", side_effect=exc):
            with patch.object(extractor, "_extract_with_pdfplumber", return_value=[]) as mock_fallback:
                result = extractor.extract(pdf_path)

        mock_fallback.assert_called_once()
        assert result == []
