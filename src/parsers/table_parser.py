"""Enterprise table extractor.

Camelot in lattice mode is the primary strategy (accurate for ruled
tables), but it hard-depends on a native Ghostscript install. If
Ghostscript is missing, fall back to pdfplumber so table extraction
still works — just less precisely — instead of failing the whole
ingest.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from loguru import logger
from rich.console import Console

from src.utils.schemas import Document, SourceType

console = Console()


class TableExtractor:
    """Extracts tables from a PDF, preferring camelot (lattice) with a
    pdfplumber fallback when Ghostscript isn't available.
    """

    def extract(self, pdf_path: Union[str, Path]) -> List[Document]:
        """Extract every table in `pdf_path` as its own Document."""

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            documents = self._extract_with_camelot(pdf_path)
        except RuntimeError as exc:
            if "ghostscript" in str(exc).lower():
                logger.warning("Ghostscript not found, falling back to pdfplumber for {}", pdf_path)
                console.print(
                    f"[yellow]Ghostscript missing[/yellow] — falling back to pdfplumber for {pdf_path.name}"
                )
                documents = self._extract_with_pdfplumber(pdf_path)
            else:
                raise

        logger.info("extracted {} table(s) from {}", len(documents), pdf_path)
        console.print(f"[green]Extracted[/green] {len(documents)} table(s) from [bold]{pdf_path.name}[/bold]")
        return documents

    def _extract_with_camelot(self, pdf_path: Path) -> List[Document]:
        import camelot

        tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice")

        documents: List[Document] = []
        for table_index, table in enumerate(tables):
            text = table.df.to_markdown(index=False)
            if not text.strip():
                continue
            documents.append(
                Document(
                    source_path=str(pdf_path),
                    source_type=SourceType.TABLE,
                    text=text,
                    metadata={
                        "source": str(pdf_path),
                        "page_number": table.page,
                        "table_index": table_index,
                        "accuracy": table.accuracy,
                        "extractor": "camelot",
                    },
                )
            )
        return documents

    def _extract_with_pdfplumber(self, pdf_path: Path) -> List[Document]:
        import pdfplumber

        documents: List[Document] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                for table_index, table in enumerate(page.extract_tables()):
                    text = self._table_to_text(table)
                    if not text.strip():
                        continue
                    documents.append(
                        Document(
                            source_path=str(pdf_path),
                            source_type=SourceType.TABLE,
                            text=text,
                            metadata={
                                "source": str(pdf_path),
                                "page_number": page_number,
                                "table_index": table_index,
                                "extractor": "pdfplumber",
                            },
                        )
                    )
        return documents

    @staticmethod
    def _table_to_text(table: list[list[str | None]]) -> str:
        if not table:
            return ""
        header, *rows = table
        header = [str(h) if h is not None else "" for h in header]
        lines = []
        for row in rows:
            cells = [f"{h}: {v}" for h, v in zip(header, row) if v is not None]
            lines.append("; ".join(cells))
        return "\n".join(lines)


def parse_tables(pdf_path: Union[str, Path]) -> List[Document]:
    """Functional convenience wrapper around TableExtractor().extract(path)."""

    return TableExtractor().extract(pdf_path)
