"""Metadata filters shared by BM25 and Qdrant search."""

from __future__ import annotations

from pydantic import BaseModel

from ledgerlens.schemas import Chunk


class SearchFilters(BaseModel):
    ticker: str | None = None
    fiscal_year: int | None = None
    form: str | None = None
    filing_id: str | None = None
    doc_name: str | None = None

    def is_empty(self) -> bool:
        return not any(self.model_dump().values())

    def matches(self, chunk: Chunk) -> bool:
        return (
            (self.ticker is None or chunk.ticker == self.ticker.upper())
            and (self.fiscal_year is None or chunk.fiscal_year == self.fiscal_year)
            and (self.form is None or chunk.form == self.form)
            and (self.filing_id is None or chunk.filing_id == self.filing_id)
            and (self.doc_name is None or chunk.doc_name == self.doc_name)
        )
