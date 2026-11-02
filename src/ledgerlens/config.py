"""Typed settings loaded from the environment and `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"
    sec_user_agent: str = "LedgerLens research contact@example.com"

    data_dir: Path = Field(default=ROOT / "data")
    cache_dir: Path = Field(default=ROOT / ".cache")
    eval_dir: Path = Field(default=ROOT / "eval")

    qdrant_url: str = "http://localhost:6333"
    phoenix_endpoint: str | None = None

    log_level: str = "INFO"
    log_json: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
