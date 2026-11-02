"""Metadata filters shared by BM25 and Qdrant search."""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from ledgerlens.schemas import FormType


class SearchFilters(BaseModel):
    ticker: str | None = None
    fiscal_year: int | None = None
    form: FormType | None = None
    filing_id: str | None = None
    doc_name: str | None = None

    @field_validator("ticker", "form", "filing_id", "doc_name", mode="before")
    @classmethod
    def _blank_is_none(cls, v: object) -> object:
        """Models often send "" for unused arguments; treat it as no filter everywhere."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    def is_empty(self) -> bool:
        return not any(v is not None for v in self.model_dump().values())
