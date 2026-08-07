"""Enterprise OCR processor.

Rasterizes each PDF page via pdf2image/poppler (`pdftoppm`) and runs
Tesseract over it, warning loudly (rather than failing silently) if
either system binary is missing from PATH.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Union

import pytesseract
from loguru import logger
from PIL import Image, ImageOps
from rich.console import Console

from src.utils.retry_handler import RetryableError, with_retry
from src.utils.schemas import Document, SourceType

console = Console()


class OCRProcessor:
    """OCRs scanned PDFs and images via Tesseract."""

    def __init__(self, lang: str = "eng", dpi: int = 300):
        self.lang = lang
        self.dpi = dpi

        if shutil.which("tesseract") is None:
            logger.warning("tesseract binary not found on PATH — OCR calls will fail until it is installed")
        if shutil.which("pdftoppm") is None:
            logger.warning("pdftoppm (poppler-utils) not found on PATH — PDF rasterization will fail")

    def _preprocess(self, image: Image.Image) -> Image.Image:
        """Grayscale + autocontrast measurably improves Tesseract accuracy."""
        return ImageOps.autocontrast(image.convert("L"))

    @with_retry(exceptions=(RetryableError, OSError))
    def _ocr_image(self, image: Image.Image) -> str:
        try:
            return pytesseract.image_to_string(image, lang=self.lang)
        except pytesseract.TesseractNotFoundError as exc:
            raise RetryableError(str(exc)) from exc

    def process_image(self, image_path: Union[str, Path]) -> Document:
        """OCR a single image file into a Document."""

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as raw_image:
            text = self._ocr_image(self._preprocess(raw_image))

        logger.info("OCR'd {} -> {} chars", image_path, len(text))
        return Document(
            source_path=str(image_path),
            source_type=SourceType.IMAGE_OCR,
            text=text.strip(),
            metadata={"source": str(image_path), "lang": self.lang, "dpi": self.dpi},
        )

    def process_pdf(self, pdf_path: Union[str, Path]) -> List[Document]:
        """Rasterize every page of a scanned PDF and OCR each one into
        its own Document.
        """

        from pdf2image import convert_from_path

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info("rasterizing {} at {} dpi", pdf_path, self.dpi)
        pages = convert_from_path(str(pdf_path), dpi=self.dpi)

        documents: List[Document] = []
        for page_number, page_image in enumerate(pages, start=1):
            text = self._ocr_image(self._preprocess(page_image))
            if not text.strip():
                continue
            documents.append(
                Document(
                    source_path=str(pdf_path),
                    source_type=SourceType.IMAGE_OCR,
                    text=text.strip(),
                    metadata={
                        "source": str(pdf_path),
                        "page_number": page_number,
                        "lang": self.lang,
                        "dpi": self.dpi,
                    },
                )
            )

        console.print(f"[green]OCR'd[/green] {len(documents)} page(s) from [bold]{pdf_path.name}[/bold]")
        logger.info("OCR'd {} page(s) from {}", len(documents), pdf_path)
        return documents


def parse_image_ocr(image_path: Union[str, Path], lang: str = "eng") -> Document:
    """Functional convenience wrapper around OCRProcessor().process_image(path)."""

    return OCRProcessor(lang=lang).process_image(image_path)
