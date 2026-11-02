import json
from datetime import date

import httpx
import pytest
import respx

from ledgerlens.ingest import edgar
from ledgerlens.ingest.edgar import (
    FetchError,
    HttpCache,
    RateLimiter,
    fiscal_year_and_period,
    select_filings,
    select_from_submissions,
)

UA = "Test Agent test@ledgerlens.dev"
SUB_URL = "https://data.sec.gov/submissions/CIK0000320193.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"


@pytest.mark.parametrize(
    ("report", "fye", "form", "expected"),
    [
        (date(2025, 9, 27), "0926", "10-K", (2025, "FY")),  # Apple 52/53-week year
        (date(2025, 12, 27), "0926", "10-Q", (2026, "Q1")),
        (date(2026, 3, 28), "0926", "10-Q", (2026, "Q2")),
        (date(2026, 6, 27), "0926", "10-Q", (2026, "Q3")),
        (date(2026, 7, 31), "0131", "10-Q", (2027, "Q2")),  # Walmart names the year it ends
        (date(2026, 1, 31), "0131", "10-K", (2026, "FY")),
        (date(2026, 3, 31), "1231", "10-Q", (2026, "Q1")),
        (date(2026, 9, 30), "1231", "10-Q", (2026, "Q3")),
        (date(2026, 1, 2), "1231", "10-K", (2025, "FY")),  # year ending just after Dec 31
        (date(2025, 11, 23), "0831", "10-Q", (2026, "Q1")),  # Costco 12-week Q1
        (date(2026, 5, 10), "0831", "10-Q", (2026, "Q3")),
        (date(2026, 6, 30), None, "10-Q", (2026, "Q2")),  # missing FYE defaults to 1231
    ],
)
def test_fiscal_year_and_period(report, fye, form, expected):
    assert fiscal_year_and_period(report, fye, form) == expected


def test_selection_rule_on_canned_submissions(fixtures):
    sub = json.loads((fixtures / "submissions_aapl.json").read_text())
    refs = select_from_submissions(sub)
    assert [r.filing_id for r in refs] == [
        "AAPL-10K-2025FY",
        "AAPL-10Q-2026Q3",
        "AAPL-10Q-2026Q2",
    ]
    k = refs[0]
    assert k.accession == "0000320193-25-000079"
    assert k.source_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
    )
    assert k.period_end == date(2025, 9, 27)
    assert k.filed == date(2025, 10, 31)
    assert k.cik == "0000320193"
    assert k.company == "Apple Inc."
    assert all(r.source_format == "html" for r in refs)


def _facts(*entries: tuple[str, int, str]) -> dict[str, object]:
    units = [
        {"end": "2026-06-27", "val": 1, "accn": accn, "fy": fy, "fp": fp, "form": "10-Q"}
        for accn, fy, fp in entries
    ]
    return {"facts": {"us-gaap": {"Revenues": {"units": {"USD": units}}}}}


def test_companyfacts_labels_win_and_fallback_fills_gaps(fixtures):
    sub = json.loads((fixtures / "submissions_aapl.json").read_text())
    facts = _facts(
        ("0000320193-26-000020", 2026, "Q3"),
        ("0000320193-26-000020", 2026, "Q3"),
        ("0000320193-26-000020", 2025, "Q3"),  # minority label is ignored
        ("0000320193-25-000079", 2025, "FY"),
        ("0000320193-26-000013", 2030, "Q1"),  # deliberately odd: proves facts take precedence
    )
    refs = select_from_submissions(sub, facts)
    assert [r.filing_id for r in refs] == [
        "AAPL-10K-2025FY",
        "AAPL-10Q-2026Q3",
        "AAPL-10Q-2030Q1",
    ]
    # Without facts for an accession, reportDate + fiscalYearEnd decide.
    only_k = select_from_submissions(sub, _facts(("0000320193-25-000079", 2025, "FY")))
    assert [r.filing_id for r in only_k][1:] == ["AAPL-10Q-2026Q3", "AAPL-10Q-2026Q2"]


def test_snapshot_is_read_before_network(fixtures, tmp_path):
    snap = tmp_path / "sec_api" / "submissions" / "CIK0000320193.json"
    snap.parent.mkdir(parents=True)
    snap.write_bytes((fixtures / "submissions_aapl.json").read_bytes())
    facts = tmp_path / "sec_api" / "api" / "xbrl" / "companyfacts" / "CIK0000320193.json"
    facts.parent.mkdir(parents=True)
    facts.write_text(json.dumps(_facts(("0000320193-26-000013", 2026, "Q2"))))
    http = HttpCache(tmp_path, sec_user_agent=UA, offline=True)
    refs = select_filings("320193", http)
    assert len(refs) == 3
    assert http.requests_made == 0


@respx.mock
def test_select_filings_fetches_once_and_caches(fixtures, tmp_path):
    route = respx.get(SUB_URL).mock(
        return_value=httpx.Response(200, content=(fixtures / "submissions_aapl.json").read_bytes())
    )
    facts_route = respx.get(FACTS_URL).mock(return_value=httpx.Response(404))
    http = HttpCache(tmp_path, sec_user_agent=UA, sleep=lambda _: None)
    first = select_filings("320193", http)
    second = select_filings("0000320193", http)
    assert first == second
    assert route.call_count == 1
    assert facts_route.call_count == 1  # the 404 is cached too
    assert route.calls[0].request.headers["User-Agent"] == UA
    assert (tmp_path / "data.sec.gov" / "submissions" / "CIK0000320193.json").exists()


@respx.mock
def test_retries_on_429_and_5xx(tmp_path):
    url = "https://www.sec.gov/Archives/edgar/data/1/2/doc.htm"
    route = respx.get(url).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "2"}),
            httpx.Response(503),
            httpx.Response(200, content=b"<html>ok</html>"),
        ]
    )
    sleeps: list[float] = []
    http = HttpCache(tmp_path, sec_user_agent=UA, sleep=sleeps.append)
    assert http.fetch(url) == b"<html>ok</html>"
    assert route.call_count == 3
    backoffs = [s for s in sleeps if s >= 1.0]
    assert len(backoffs) == 2
    assert backoffs[0] >= 2.0  # honours Retry-After


@respx.mock
def test_404_is_not_retried(tmp_path, monkeypatch):
    url = "https://www.sec.gov/Archives/missing.htm"
    route = respx.get(url).mock(return_value=httpx.Response(404))
    http = HttpCache(tmp_path, sec_user_agent=UA, sleep=lambda _: None)
    with pytest.raises(FetchError) as err:
        http.fetch(url)
    assert err.value.status == 404
    assert route.call_count == 1


@respx.mock
def test_gives_up_after_max_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(edgar, "MAX_RETRIES", 2)
    url = "https://www.sec.gov/Archives/down.htm"
    route = respx.get(url).mock(return_value=httpx.Response(500))
    http = HttpCache(tmp_path, sec_user_agent=UA, sleep=lambda _: None)
    with pytest.raises(FetchError):
        http.fetch(url)
    assert route.call_count == 3


@respx.mock
def test_non_sec_hosts_get_generic_agent(tmp_path):
    url = "https://raw.githubusercontent.com/org/repo/main/pdfs/X.pdf"
    route = respx.get(url).mock(return_value=httpx.Response(200, content=b"%PDF-1.7"))
    http = HttpCache(tmp_path, sec_user_agent=UA, sleep=lambda _: None)
    http.fetch(url)
    assert route.calls[0].request.headers["User-Agent"] == edgar.GENERIC_USER_AGENT


def test_offline_cache_miss_raises(tmp_path):
    http = HttpCache(tmp_path, sec_user_agent=UA, offline=True)
    with pytest.raises(FetchError):
        http.fetch("https://www.sec.gov/x.htm")


def test_cache_path_includes_query_hash(tmp_path):
    http = HttpCache(tmp_path, sec_user_agent=UA)
    a = http.cache_path("https://example.com/pdf?target=a")
    b = http.cache_path("https://example.com/pdf?target=b")
    assert a != b and a.parent == tmp_path / "example.com"


def test_rate_limiter_spaces_requests():
    now = [0.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    limiter = RateLimiter(4.0, clock=lambda: now[0], sleep=sleep)
    for _ in range(5):
        limiter.wait()
    assert now[0] == pytest.approx(1.0)  # 5 requests at 4/s span one second
    assert edgar.SEC_MAX_RPS <= 4.0
