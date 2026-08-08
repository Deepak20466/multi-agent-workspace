"""Enterprise document ingestion pipeline: load -> OCR-fallback for
scanned PDFs -> redact PII -> chunk.

Fans out to ExcelLoader / OCRProcessor / TableExtractor based on file
extension, redacts PII before the text ever reaches the vector store
(toggle via MASK_PII_ON_INGEST), and splits each Document into
overlapping Chunks ready for embedding.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pdfplumber
from loguru import logger
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from rich.console import Console

from src.parsers.excel_parser import ExcelLoader
from src.parsers.ocr_parser import OCRDependencyError, OCRProcessor
from src.parsers.table_parser import TableExtractor
from src.utils.schemas import Chunk, Document, PIIEntity, SourceType

console = Console()

TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
EXCEL_EXTENSIONS = {".xlsx", ".xls"}
PDF_EXTENSIONS = {".pdf"}

SCANNED_PDF_CHAR_THRESHOLD = 100


def _stable_chunk_id(source_path: str, chunk_index: int) -> str:
    """Deterministic chunk_id derived from (source_path, chunk_index).

    Chunk.chunk_id defaults to a fresh random uuid4, which makes
    VectorStore.add_chunks()'s Chroma upsert() (keyed by id) unable to
    recognize a re-indexed file's chunks as the same ones it saw
    before -- every re-index silently piles up duplicate copies of the
    same content instead of updating them in place. Deriving the id
    from stable identity (same file, same chunk position) instead of
    randomness makes re-indexing idempotent.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_path}::{chunk_index}"))

DEFAULT_PII_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "IBAN_CODE",
    "IP_ADDRESS",
]


class DocumentProcessor:
    """Loads documents, redacts PII, and chunks them for ingestion."""

    def __init__(
        self,
        ocr_lang: Optional[str] = None,
        ocr_dpi: Optional[int] = None,
        mask_pii: Optional[bool] = None,
        pii_entities: Optional[List[str]] = None,
    ):
        self.ocr_lang = ocr_lang or os.getenv("OCR_LANG", "eng")
        self.ocr_dpi = ocr_dpi or int(os.getenv("OCR_DPI", "300"))
        self.mask_pii = (
            mask_pii if mask_pii is not None else os.getenv("MASK_PII_ON_INGEST", "true").lower() == "true"
        )
        self.pii_entities = pii_entities or DEFAULT_PII_ENTITIES

        self.excel_loader = ExcelLoader()
        self.ocr_processor = OCRProcessor(lang=self.ocr_lang, dpi=self.ocr_dpi)
        self.table_extractor = TableExtractor()
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

    def _is_scanned_pdf(self, text: str) -> bool:
        """A PDF whose extracted text layer is this short almost certainly
        has no real text layer and needs OCR instead of direct extraction.
        """

        return len(text.strip()) < SCANNED_PDF_CHAR_THRESHOLD

    def _redact_pii(self, text: str) -> Tuple[str, List[PIIEntity]]:
        """Analyze `text` for PII and return (redacted_text, entities_found).

        A no-op (returns the original text) when MASK_PII_ON_INGEST is
        disabled or the text is empty.
        """

        if not self.mask_pii or not text.strip():
            return text, []

        results = self._analyzer.analyze(text=text, entities=self.pii_entities, language="en")
        if not results:
            return text, []

        anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)
        entities = [
            PIIEntity(entity_type=r.entity_type, text=text[r.start : r.end], start=r.start, end=r.end, score=r.score)
            for r in results
        ]
        logger.info("redacted {} PII entit(y/ies) from text", len(entities))
        return anonymized.text, entities

    def _extract_pdf_text(self, file_path: Path) -> str:
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return "\n".join(pages)

    def load(self, file_path: Union[str, Path]) -> List[Document]:
        """Dispatch a single file to the correct parser, redact PII in
        every resulting Document, and return them.
        """

        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix in EXCEL_EXTENSIONS:
            documents = self.excel_loader.load(file_path)

        elif suffix in IMAGE_EXTENSIONS:
            documents = [self.ocr_processor.process_image(file_path)]

        elif suffix in PDF_EXTENSIONS:
            text = self._extract_pdf_text(file_path)
            if self._is_scanned_pdf(text):
                logger.warning(
                    "{} looks like a scanned PDF (<{} chars extracted) — falling back to OCR",
                    file_path,
                    SCANNED_PDF_CHAR_THRESHOLD,
                )
                try:
                    documents = self.ocr_processor.process_pdf(file_path)
                except OCRDependencyError as exc:
                    if text.strip():
                        # This PDF isn't actually a scanned image -- it has a
                        # real (if short) text layer, it just tripped the
                        # length heuristic above. OCR being unavailable
                        # shouldn't lose ordinary, non-scanned documents;
                        # ingest the text layer we already have instead of
                        # crashing. `ocr_skipped`/`ocr_error` make it explicit
                        # in metadata that this is *not* an OCR pass, so
                        # nothing downstream mistakes it for one.
                        logger.warning(
                            "OCR unavailable for {} ({}); falling back to the {} char(s) of text already extracted",
                            file_path,
                            exc,
                            len(text.strip()),
                        )
                        documents = [
                            Document(
                                source_path=str(file_path),
                                source_type=SourceType.PDF,
                                text=text,
                                metadata={"ocr_skipped": True, "ocr_error": str(exc)},
                            )
                        ]
                    else:
                        # No text layer at all and OCR can't run: there is
                        # nothing to ingest. Surface a clear, typed error
                        # rather than silently producing an empty/blank
                        # document that would look like a successful (if
                        # content-free) ingest.
                        logger.error("OCR unavailable for {} and it has no extractable text layer: {}", file_path, exc)
                        raise
            else:
                documents = [Document(source_path=str(file_path), source_type=SourceType.PDF, text=text)]
            documents.extend(self.table_extractor.extract(file_path))

        elif suffix in TEXT_EXTENSIONS:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            documents = [Document(source_path=str(file_path), source_type=SourceType.TEXT, text=text)]

        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        for document in documents:
            redacted_text, entities = self._redact_pii(document.text)
            document.text = redacted_text
            if entities:
                document.metadata["pii_redacted_count"] = len(entities)

        console.print(f"[green]Loaded[/green] {len(documents)} document(s) from [bold]{file_path.name}[/bold]")
        logger.info("loaded {} document(s) from {}", len(documents), file_path)
        return documents

    def chunk(self, document: Document, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Chunk]:
        """Split a Document's text into overlapping character-window chunks.

        Overlap keeps sentences that straddle a chunk boundary retrievable
        from either neighbor, which materially improves recall in RAG.
        """

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        text = document.text.strip()
        if not text:
            return []

        chunks: List[Chunk] = []
        start = 0
        index = 0
        step = chunk_size - chunk_overlap

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=_stable_chunk_id(document.source_path, index),
                        doc_id=document.doc_id,
                        text=chunk_text,
                        chunk_index=index,
                        page_number=document.metadata.get("page_number"),
                        metadata={
                            **document.metadata,
                            "source_path": document.source_path,
                            "source_type": document.source_type.value,
                        },
                    )
                )
                index += 1
            if end == len(text):
                break
            start += step

        return chunks

    def process(
        self,
        file_path: Union[str, Path],
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> Tuple[List[Document], List[Chunk]]:
        """End-to-end: load a file into Documents, then chunk all of them."""

        documents = self.load(file_path)
        chunks: List[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
        logger.info("processed {} -> {} documents, {} chunks", file_path, len(documents), len(chunks))
        return documents, chunks
