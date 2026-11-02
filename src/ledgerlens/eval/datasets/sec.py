"""Cached, rate-limited access to the SEC JSON APIs used to build eval datasets.

Raw responses are written under `{data_dir}/raw/sec_api/` mirroring the URL path, so a
rerun reads from disk and makes no requests. Requests are spaced to stay at or below
`MAX_RPS`, well under SEC's 10 requests/second fair-access limit, because other
processes (ingestion) may hit SEC at the same time.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from ledgerlens.config import get_settings
from ledgerlens.logging import get_logger

log = get_logger(__name__)

SEC_DATA_HOST = "https://data.sec.gov"
MAX_RPS = 4.0
MAX_ATTEMPTS = 4
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

_lock = threading.Lock()
_last_request = 0.0


def cache_root() -> Path:
    return get_settings().data_dir / "raw" / "sec_api"


def submissions_url(cik: str) -> str:
    return f"{SEC_DATA_HOST}/submissions/CIK{cik}.json"


def companyfacts_url(cik: str) -> str:
    return f"{SEC_DATA_HOST}/api/xbrl/companyfacts/CIK{cik}.json"


def _cache_path(url: str, root: Path) -> Path:
    rel = url.removeprefix(SEC_DATA_HOST).lstrip("/")
    return root / rel


def _throttle() -> None:
    global _last_request
    with _lock:
        wait = _last_request + 1.0 / MAX_RPS - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()


def fetch_json(url: str, *, root: Path | None = None, client: httpx.Client | None = None) -> Any:
    """Return the JSON body for `url`, from the disk cache when present."""
    path = _cache_path(url, root or cache_root())
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    headers = {"User-Agent": get_settings().require_sec_user_agent(), "Accept": "application/json"}
    owns = client is None
    http = client or httpx.Client(timeout=60.0, follow_redirects=True)
    try:
        resp = _get_with_retries(http, url, headers)
    finally:
        if owns:
            http.close()
    data = resp.json()  # parse before caching so a bad body never reaches the cache
    _atomic_write(path, resp.text)
    log.info("sec_fetched", url=url, bytes=len(resp.content))
    return data


def _get_with_retries(http: httpx.Client, url: str, headers: dict[str, str]) -> httpx.Response:
    """GET with backoff on 429/5xx and on transport errors (timeouts, resets)."""
    for attempt in range(MAX_ATTEMPTS):
        last = attempt == MAX_ATTEMPTS - 1
        _throttle()
        try:
            resp = http.get(url, headers=headers)
        except httpx.TransportError as exc:
            if last:
                raise
            log.warning("sec_retry", url=url, error=type(exc).__name__, attempt=attempt)
            time.sleep(2.0**attempt)
            continue
        if resp.status_code in RETRY_STATUSES and not last:
            log.warning("sec_retry", url=url, status=resp.status_code, attempt=attempt)
            time.sleep(2.0**attempt)
            continue
        resp.raise_for_status()
        return resp
    raise AssertionError("unreachable")  # pragma: no cover


def _atomic_write(path: Path, text: str) -> None:
    """Write via a unique temp file and os.replace, so a crash never leaves partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def submissions(cik: str, *, root: Path | None = None) -> dict[str, Any]:
    data: dict[str, Any] = fetch_json(submissions_url(cik), root=root)
    return data


def companyfacts(cik: str, *, root: Path | None = None) -> dict[str, Any]:
    data: dict[str, Any] = fetch_json(companyfacts_url(cik), root=root)
    return data
