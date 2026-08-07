"""Enterprise Excel loader.

Every sheet in a workbook becomes its own Document, rendered as a
markdown table so the text is both human-readable and embeds well for
RAG, while row/column metadata is preserved for exact lookups.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import pandas as pd
from loguru import logger
from rich.console import Console

from src.utils.schemas import Document, SourceType

console = Console()


class ExcelLoader:
    """Loads every sheet of an .xlsx/.xls workbook into a Document."""

    def load(self, path: Union[str, Path]) -> List[Document]:
        """Read all sheets from the workbook at `path` and return one
        Document per non-empty sheet, with its contents rendered as a
        markdown table.
        """

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {path}")

        logger.info("loading excel workbook: {}", path)
        sheets: dict[str, pd.DataFrame] = pd.read_excel(path, sheet_name=None)

        documents: List[Document] = []
        for sheet_name, df in sheets.items():
            if df.empty:
                logger.warning("sheet '{}' in {} is empty, skipping", sheet_name, path)
                continue

            documents.append(
                Document(
                    source_path=str(path),
                    source_type=SourceType.EXCEL,
                    text=df.to_markdown(index=False),
                    metadata={
                        "source": str(path),
                        "sheet_name": sheet_name,
                        "columns": [str(c) for c in df.columns],
                        "n_rows": len(df),
                    },
                )
            )

        console.print(f"[green]Loaded[/green] {len(documents)} sheet(s) from [bold]{path.name}[/bold]")
        logger.info("loaded {} sheet(s) from {}", len(documents), path)
        return documents


def parse_excel(path: Union[str, Path]) -> List[Document]:
    """Functional convenience wrapper around ExcelLoader().load(path)."""

    return ExcelLoader().load(path)
