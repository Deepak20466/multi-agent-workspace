import pytest

from src.document_processing import DocumentProcessor
from src.utils.schemas import Document, SourceType


@pytest.fixture
def processor(document_processor: DocumentProcessor) -> DocumentProcessor:
    return document_processor


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
