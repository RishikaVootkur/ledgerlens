"""Typed settings loaded from the environment and `.env`."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def project_root() -> Path:
    """Where `.env`, `data/`, `.cache/` and `eval/` live.

    `LEDGERLENS_HOME` wins (used by Docker). Otherwise the source checkout this
    module lives in, falling back to the current directory for a wheel install.
    """
    if home := os.environ.get("LEDGERLENS_HOME"):
        return Path(home).resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").exists():
        return checkout
    return Path.cwd()


ROOT = project_root()

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"
    sec_user_agent: str = ""

    data_dir: Path = Field(default=ROOT / "data")
    cache_dir: Path = Field(default=ROOT / ".cache")
    eval_dir: Path = Field(default=ROOT / "eval")

    # live: call providers and record responses. replay: answer only from the cache
    # (CI uses this with the committed eval/cache, so it needs no keys or network).
    llm_mode: Literal["live", "replay"] = "live"

    qdrant_url: str = "http://localhost:6333"
    phoenix_endpoint: str | None = None

    # Comma-separated browser origins allowed to call the API (the Vite dev server).
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    # Agent runs that may execute at once; more requests wait for a free slot.
    max_concurrent_chats: int = 2

    log_level: LogLevel = "INFO"
    log_json: bool = False

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_level(cls, v: object) -> object:
        return v.upper() if isinstance(v, str) else v

    def require_sec_user_agent(self) -> str:
        """SEC blocks anonymous or placeholder agents; fail before any request."""
        ua = self.sec_user_agent.strip()
        if "@" not in ua or "example.com" in ua:
            raise ValueError(
                'Set SEC_USER_AGENT in .env to "Name email@domain" (SEC fair-access rule).'
            )
        return ua


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
