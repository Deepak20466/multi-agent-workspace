"""Enterprise DOCX loader.

Mirrors ExcelLoader/TableExtractor's split: a Word document's body
paragraphs become one Document (heading styles rendered as markdown
`#`/`##`/... so section structure survives into the chunked text,
matching how ExcelLoader renders sheets as markdown for the same
readability-for-RAG reason), and every table becomes its own Document,
same as a PDF's tables do via TableExtractor.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from docx import Document as _DocxDocument
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from loguru import logger
from rich.console import Console

from src.utils.schemas import Document, SourceType

console = Console()


def _render_paragraph(paragraph) -> str:
    """Render one paragraph's text, prefixing headings with markdown
    `#`s (by paragraph style, e.g. "Heading 2" -> "##") so section
    structure is legible in the flattened chunk text instead of being
    silently discarded.
    """

    text = paragraph.text.strip()
    if not text:
        return ""

    style_name = (paragraph.style.name if paragraph.style is not None else "") or ""
    if style_name == "Title":
        return f"# {text}"
    if style_name.startswith("Heading "):
        level_str = style_name.rsplit(" ", 1)[-1]
        level = int(level_str) if level_str.isdigit() else 1
        return f"{'#' * max(1, min(level, 6))} {text}"
    return text


def _table_to_text(table: Table) -> str:
    """Render a docx table the same `key: value; key: value` per-row
    shape TableExtractor._table_to_text uses for PDF tables, using the
    first row as the header.
    """

    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return ""

    header, *body_rows = rows
    lines = []
    for row in body_rows:
        cells = [f"{h}: {v}" for h, v in zip(header, row) if v]
        if cells:
            lines.append("; ".join(cells))
    return "\n".join(lines)


class DocxLoader:
    """Loads paragraphs and tables from a .docx file into Documents."""

    def load(self, path: Union[str, Path]) -> List[Document]:
        """Read `path` and return one Document for the body paragraphs
        (skipped if there's no non-empty paragraph text at all) plus one
        Document per non-empty table.
        """

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DOCX file not found: {path}")

        logger.info("loading docx: {}", path)
        try:
            docx_document = _DocxDocument(str(path))
        except PackageNotFoundError as exc:
            # Empty file, non-zip/non-OOXML content, or a corrupted
            # archive all surface here as the same opaque
            # PackageNotFoundError from python-docx's opc layer -- raise
            # a ValueError instead so callers get a clear, expected
            # "bad input" error rather than a third-party exception type
            # they'd need to know to catch.
            raise ValueError(f"Not a valid DOCX file: {path}") from exc

        documents: List[Document] = []

        paragraph_lines = [_render_paragraph(p) for p in docx_document.paragraphs]
        paragraph_lines = [line for line in paragraph_lines if line]
        if paragraph_lines:
            documents.append(
                Document(
                    source_path=str(path),
                    source_type=SourceType.DOCX,
                    text="\n".join(paragraph_lines),
                    metadata={
                        "source": str(path),
                        "n_paragraphs": len(paragraph_lines),
                        **self._core_properties(docx_document),
                    },
                )
            )
        else:
            logger.warning("{} has no extractable paragraph text", path)

        for table_index, table in enumerate(docx_document.tables):
            text = _table_to_text(table)
            if not text.strip():
                continue
            documents.append(
                Document(
                    source_path=str(path),
                    source_type=SourceType.TABLE,
                    text=text,
                    metadata={
                        "source": str(path),
                        "table_index": table_index,
                        "n_rows": len(table.rows),
                        "extractor": "python-docx",
                    },
                )
            )

        console.print(f"[green]Loaded[/green] {len(documents)} document(s) from [bold]{path.name}[/bold]")
        logger.info("loaded {} document(s) from {}", len(documents), path)
        return documents

    @staticmethod
    def _core_properties(docx_document) -> dict:
        """Best-effort OOXML core-properties (title/author/timestamps) --
        every field is optional in the format, so only non-empty ones
        are included.
        """

        props = docx_document.core_properties
        result: dict = {}
        if props.title:
            result["title"] = props.title
        if props.author:
            result["author"] = props.author
        if props.subject:
            result["subject"] = props.subject
        if props.created:
            result["created"] = props.created.isoformat()
        if props.modified:
            result["modified"] = props.modified.isoformat()
        return result


def parse_docx(path: Union[str, Path]) -> List[Document]:
    """Functional convenience wrapper around DocxLoader().load(path)."""

    return DocxLoader().load(path)
