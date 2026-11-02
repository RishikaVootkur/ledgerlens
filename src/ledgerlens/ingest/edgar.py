"""SEC EDGAR client and the filing selection rule.

Every download goes through `HttpCache.fetch`, which stores the raw bytes under
`DATA_DIR/raw/{host}/{path}` so a rerun makes no requests. data.sec.gov JSON
(submissions, companyfacts) is read first from the shared snapshot in
`DATA_DIR/raw/sec_api/{path}` that the eval dataset builder also uses, so both
select the same filings even if a company files something new later. Requests to SEC hosts
carry the configured SEC user agent and are throttled to `SEC_MAX_RPS` (4/s, well
under SEC's 10/s fair-access limit because other processes share it). 429 and 5xx
responses and transport errors are retried with exponential backoff.

Selection rule (shared with the XBRL question generator, keep it exact):
per company, from the submissions `filings.recent` list, take the latest `10-K`
and the two most recent `10-Q`s by `filingDate` (exact form match, so amendments
like `10-K/A` are excluded; a stable sort keeps the `recent` order on ties). Use
the primary document. Fiscal year and period are the most common (`fy`, `fp`)
over companyfacts facts whose `accn` is the filing's accession; when companyfacts
has no fact for that accession they come from `reportDate` and the company's
`fiscalYearEnd` (MMDD) via :func:`fiscal_year_and_period`.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from collections import Counter
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from ledgerlens.config import get_settings
from ledgerlens.logging import get_logger
from ledgerlens.schemas import FilingRef, FiscalPeriod, make_filing_id
from ledgerlens.universe import UNIVERSE

log = get_logger(__name__)

SEC_MAX_RPS = 4.0
OTHER_MAX_RPS = 4.0
MAX_RETRIES = 6
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 60.0
TIMEOUT_S = 60.0
GENERIC_USER_AGENT = "LedgerLens/0.1 (+https://github.com/RishikaVootkur/ledgerlens)"
SEC_HOSTS = ("sec.gov",)
RETRY_STATUS = {429, 500, 502, 503, 504}
FORBIDDEN_BACKOFF_S = 10.0  # base wait after an SEC 403, doubled per attempt
FORBIDDEN_BACKOFF_CAP_S = 300.0
PERMANENT_STATUS = {404, 410}

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SNAPSHOT_DIR = "sec_api"
SNAPSHOT_HOSTS = ("data.sec.gov",)
_PERIODS: tuple[FiscalPeriod, ...] = ("FY", "Q1", "Q2", "Q3", "Q4")
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"


class FetchError(RuntimeError):
    """A download failed for good (404, 403, or retries exhausted)."""

    def __init__(self, url: str, status: int | None, message: str) -> None:
        super().__init__(f"{message} ({status}) for {url}")
        self.url = url
        self.status = status


class RateLimiter:
    """Minimum spacing between request starts. Thread safe."""

    def __init__(
        self,
        max_per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = 1.0 / max_per_second
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if now < self._next:
                self._sleep(self._next - now)
                now = self._next
            self._next = now + self.interval


def _is_sec(host: str) -> bool:
    return any(host == h or host.endswith("." + h) for h in SEC_HOSTS)


class HttpCache:
    """Cached, throttled, retrying HTTP GET for SEC and other document hosts."""

    def __init__(
        self,
        raw_dir: Path,
        *,
        client: httpx.Client | None = None,
        sec_user_agent: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        offline: bool = False,
    ) -> None:
        self.raw_dir = raw_dir
        self._client = client
        self._sec_ua = sec_user_agent
        self._sleep = sleep
        self.offline = offline
        self.sec_limiter = RateLimiter(SEC_MAX_RPS, sleep=sleep)
        self.other_limiter = RateLimiter(OTHER_MAX_RPS, sleep=sleep)
        self.requests_made = 0

    # ------------------------------------------------------------ paths

    def cache_path(self, url: str) -> Path:
        parts = urlsplit(url)
        path = parts.path.lstrip("/") or "index"
        if parts.query:
            digest = hashlib.sha1(parts.query.encode()).hexdigest()[:12]
            path = f"{path}__{digest}"
        return self.raw_dir / parts.netloc / path

    def snapshot_path(self, url: str) -> Path | None:
        """Shared snapshot location for data.sec.gov JSON, mirroring the URL path."""
        parts = urlsplit(url)
        if parts.netloc not in SNAPSHOT_HOSTS or parts.query:
            return None
        return self.raw_dir / SNAPSHOT_DIR / parts.path.lstrip("/")

    def is_cached(self, url: str) -> bool:
        snap = self.snapshot_path(url)
        return (snap is not None and snap.exists()) or self.cache_path(url).exists()

    # ------------------------------------------------------------ http

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=TIMEOUT_S, follow_redirects=True)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def _headers(self, host: str) -> dict[str, str]:
        if _is_sec(host):
            if self._sec_ua is None:
                self._sec_ua = get_settings().require_sec_user_agent()
            return {"User-Agent": self._sec_ua, "Accept-Encoding": "gzip, deflate"}
        return {"User-Agent": GENERIC_USER_AGENT}

    def fetch(self, url: str, *, refresh: bool = False) -> bytes:
        """Return the bytes at `url`, from the snapshot or raw cache when present.

        `refresh` skips both and re-downloads into the raw cache (never the snapshot).
        """
        path = self.cache_path(url)
        snap = self.snapshot_path(url)
        if not refresh:
            if snap is not None and snap.exists():
                return snap.read_bytes()
            if path.exists():
                return path.read_bytes()
        if self.offline:
            raise FetchError(url, None, "offline and not cached")
        data = self._get(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        return data

    def _get(self, url: str) -> bytes:
        host = urlsplit(url).netloc
        limiter = self.sec_limiter if _is_sec(host) else self.other_limiter
        headers = self._headers(host)
        for attempt in range(MAX_RETRIES + 1):
            limiter.wait()
            self.requests_made += 1
            status: int | None = None
            retry_after: float | None = None
            try:
                resp = self.client.get(url, headers=headers)
                status = resp.status_code
                if status == 200:
                    log.debug("fetched", url=url, bytes=len(resp.content))
                    return resp.content
                sec_forbidden = status == 403 and _is_sec(host)
                if status not in RETRY_STATUS and not sec_forbidden:
                    raise FetchError(url, status, "HTTP error")
                retry_after = _retry_after(resp.headers.get("Retry-After"))
                if sec_forbidden:
                    # SEC answers rate-limit breaches and bad user agents with 403.
                    wait = min(FORBIDDEN_BACKOFF_CAP_S, FORBIDDEN_BACKOFF_S * 2**attempt)
                    retry_after = max(retry_after or 0.0, wait)
                    log.warning("sec_forbidden", url=url, body=resp.text[:200])
            except httpx.TransportError as exc:
                log.warning("transport_error", url=url, error=type(exc).__name__)
            if attempt == MAX_RETRIES:
                break
            delay = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2**attempt) + random.uniform(0, 0.5)
            if retry_after is not None:
                delay = max(delay, retry_after)
            log.warning("retrying", url=url, status=status, attempt=attempt + 1, delay=delay)
            self._sleep(delay)
        raise FetchError(url, status, "retries exhausted")

    def fetch_json(self, url: str, *, refresh: bool = False) -> Any:
        return json.loads(self.fetch(url, refresh=refresh))


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return min(BACKOFF_CAP_S, float(value))
    except ValueError:
        return None


# ---------------------------------------------------------------- selection


def _fy_end(year: int, fiscal_year_end: str) -> date:
    month, day = int(fiscal_year_end[:2]), int(fiscal_year_end[2:])
    while True:
        try:
            return date(year, month, day)
        except ValueError:  # e.g. 0229 or 0931
            day -= 1


def fiscal_year_and_period(
    report_date: date, fiscal_year_end: str | None, form: str
) -> tuple[int, FiscalPeriod]:
    """Map a period-end date to (fiscal_year, fiscal_period).

    The fiscal year is named after the calendar year in which it ends (Apple's year
    ending 2025-09-27 is FY2025; Walmart's year ending 2026-01-31 is FY2026). It is
    the first year Y whose fiscal year end (from `fiscalYearEnd`, MMDD) plus a 14 day
    grace for 52/53-week calendars is on or after the report date. The quarter is
    4 minus the number of whole quarters (91.3 days, rounded) left until that year
    end. A 10-K is always FY.

    A year end in the first week of January (Johnson & Johnson's "0103") belongs to a
    52/53-week year that is named for the previous calendar year, so it is treated
    as "1231".
    """
    fye = fiscal_year_end if fiscal_year_end and len(fiscal_year_end) == 4 else "1231"
    if fye[:2] == "01" and int(fye[2:]) <= 7:
        fye = "1231"
    year = report_date.year - 1
    while report_date > _fy_end(year, fye) + timedelta(days=14):
        year += 1
    if form == "10-K":
        return year, "FY"
    quarters_left = round((_fy_end(year, fye) - report_date).days / 91.3)
    quarter = min(4, max(1, 4 - quarters_left))
    period: FiscalPeriod = ("Q1", "Q2", "Q3", "Q4")[quarter - 1]  # type: ignore[assignment]
    return year, period


def _company_for(cik: str, submissions: dict[str, Any]) -> tuple[str, str]:
    for c in UNIVERSE:
        if c.cik == cik:
            return c.ticker, c.name
    tickers = submissions.get("tickers") or []
    ticker = tickers[0] if tickers else f"CIK{int(cik)}"
    return str(ticker).upper(), str(submissions.get("name", ticker))


def fiscal_labels(companyfacts: dict[str, Any] | None, accession: str) -> tuple[int, str] | None:
    """Most common (fy, fp) over companyfacts facts filed with `accession`, if any."""
    if not companyfacts:
        return None
    counts: Counter[tuple[int, str]] = Counter()
    for concepts in companyfacts.get("facts", {}).values():
        for cdata in concepts.values():
            for facts in cdata.get("units", {}).values():
                for f in facts:
                    if f.get("accn") == accession and f.get("fy") and f.get("fp"):
                        counts[(int(f["fy"]), str(f["fp"]))] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _fiscal(
    companyfacts: dict[str, Any] | None,
    accession: str,
    report: date,
    fiscal_year_end: str | None,
    form: str,
) -> tuple[int, FiscalPeriod]:
    labels = fiscal_labels(companyfacts, accession)
    if labels is not None:
        fy, fp = labels
        if form == "10-K":
            return fy, "FY"
        if fp in _PERIODS and fp != "FY":
            return fy, fp  # type: ignore[return-value]
    return fiscal_year_and_period(report, fiscal_year_end, form)


def select_from_submissions(
    submissions: dict[str, Any],
    companyfacts: dict[str, Any] | None = None,
    *,
    ticker: str | None = None,
    company: str | None = None,
) -> list[FilingRef]:
    """Apply the selection rule to submissions (+ companyfacts) JSON. Pure, no network."""
    cik = str(submissions["cik"]).zfill(10)
    default_ticker, default_name = _company_for(cik, submissions)
    ticker = ticker or default_ticker
    company = company or default_name
    fye = submissions.get("fiscalYearEnd")
    recent = submissions["filings"]["recent"]
    rows = [{key: recent[key][i] for key in recent} for i in range(len(recent["accessionNumber"]))]
    picked: list[dict[str, Any]] = []
    for form, keep in (("10-K", 1), ("10-Q", 2)):
        same = sorted(
            (r for r in rows if r["form"] == form), key=lambda r: r["filingDate"], reverse=True
        )
        picked.extend(same[:keep])
    refs = []
    for r in picked:
        report = date.fromisoformat(r["reportDate"])
        acc = r["accessionNumber"]
        fy, fp = _fiscal(companyfacts, acc, report, fye, r["form"])
        refs.append(
            FilingRef(
                filing_id=make_filing_id(ticker, r["form"], fy, fp, acc),
                cik=cik,
                ticker=ticker,
                company=company,
                form=r["form"],
                fiscal_year=fy,
                fiscal_period=fp,
                period_end=report,
                filed=date.fromisoformat(r["filingDate"]),
                accession=acc,
                source_url=ARCHIVES_URL.format(
                    cik_int=int(cik), acc_nodash=acc.replace("-", ""), doc=r["primaryDocument"]
                ),
                source_format="html",
            )
        )
    return refs


def fetch_submissions(cik: str, http: HttpCache, *, refresh: bool = False) -> dict[str, Any]:
    data: dict[str, Any] = http.fetch_json(
        SUBMISSIONS_URL.format(cik=str(cik).zfill(10)), refresh=refresh
    )
    return data


def fetch_companyfacts(
    cik: str, http: HttpCache, *, refresh: bool = False
) -> dict[str, Any] | None:
    """Companyfacts JSON, or None when SEC has none for this CIK.

    A 404 is cached as `{}` so reruns do not ask again.
    """
    url = COMPANYFACTS_URL.format(cik=str(cik).zfill(10))
    try:
        data: dict[str, Any] = http.fetch_json(url, refresh=refresh)
    except FetchError as exc:
        if exc.status != 404:
            raise
        path = http.cache_path(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return None
    return data or None


def select_filings(
    cik: str, http: HttpCache | None = None, *, refresh: bool = False
) -> list[FilingRef]:
    """Latest 10-K and two most recent 10-Qs for a company (see module docstring)."""
    own = http is None
    if http is None:
        http = HttpCache(get_settings().data_dir / "raw")
    try:
        return select_from_submissions(
            fetch_submissions(cik, http, refresh=refresh),
            fetch_companyfacts(cik, http, refresh=refresh),
        )
    finally:
        if own:
            http.close()
