from unittest.mock import patch

import openpyxl
import pytest

from src.parsers.excel_parser import ExcelLoader
from src.parsers.ocr_parser import OCRProcessor
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


def test_table_extractor_falls_back_to_pdfplumber_on_ghostscript_error(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    extractor = TableExtractor()
    with patch.object(extractor, "_extract_with_camelot", side_effect=RuntimeError("Ghostscript is not installed")):
        with patch.object(extractor, "_extract_with_pdfplumber", return_value=[]) as mock_fallback:
            result = extractor.extract(pdf_path)

    mock_fallback.assert_called_once()
    assert result == []


def test_table_extractor_reraises_non_ghostscript_runtime_error(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    extractor = TableExtractor()
    with patch.object(extractor, "_extract_with_camelot", side_effect=RuntimeError("something else broke")):
        with pytest.raises(RuntimeError):
            extractor.extract(pdf_path)
