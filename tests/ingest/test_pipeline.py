import json

import httpx
import pytest
import respx

from ledgerlens.ingest.edgar import HttpCache
from ledgerlens.ingest.pipeline import DataPaths, Manifest, run

UA = "Test Agent test@ledgerlens.dev"
SUB_URL = "https://data.sec.gov/submissions/CIK0000320193.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
DOC_PREFIX = "https://www.sec.gov/Archives/edgar/data/320193/"


@pytest.fixture
def sec(fixtures):
    html = (fixtures / "edgar_sample.htm").read_bytes()
    with respx.mock(assert_all_called=False) as mock:
        sub = mock.get(SUB_URL).mock(
            return_value=httpx.Response(
                200, content=(fixtures / "submissions_aapl.json").read_bytes()
            )
        )
        mock.get(FACTS_URL).mock(return_value=httpx.Response(404))
        docs = mock.get(url__startswith=DOC_PREFIX).mock(
            return_value=httpx.Response(200, content=html)
        )
        yield sub, docs, mock


def _run(tmp_path):
    http = HttpCache(tmp_path / "raw", sec_user_agent=UA, sleep=lambda _: None)
    return run(
        tmp_path,
        tickers=["AAPL"],
        include_financebench=False,
        workers=1,
        http=http,
        docs_report=tmp_path / "docs" / "data-quality.md",
    )


def test_second_run_does_no_network_or_parse_work(sec, tmp_path):
    sub, docs, _ = sec
    first = _run(tmp_path)
    assert len(first.filings) == 3
    assert (first.downloaded, first.parsed, first.chunked, first.failed) == (3, 3, 3, {})
    assert first.requests == 5  # submissions, companyfacts, three documents
    paths = DataPaths(tmp_path)
    for ref in first.filings:
        assert paths.parsed(ref.filing_id).exists()
        for s in ("fixed", "section"):
            assert paths.chunks(s, ref.filing_id).exists()
    manifest = json.loads(paths.manifest.read_text())
    steps = manifest["filings"]["AAPL-10K-2025FY"]["steps"]
    assert set(steps) == {"download", "parse", "chunk:fixed", "chunk:section"}
    assert len(steps["parse"]["output_sha256"]) == 64
    assert (tmp_path / "reports" / "quality.json").exists()
    assert (tmp_path / "docs" / "data-quality.md").exists()
    listed = json.loads(paths.filings.read_text())
    assert [f["filing_id"] for f in listed] == [r.filing_id for r in first.filings]
    calls = (sub.call_count, docs.call_count)

    second = _run(tmp_path)
    assert (sub.call_count, docs.call_count) == calls
    assert second.requests == 0
    assert (second.downloaded, second.parsed, second.chunked, second.skipped) == (0, 0, 0, 3)


def test_tampered_output_is_rebuilt_without_download(sec, tmp_path):
    _run(tmp_path)
    paths = DataPaths(tmp_path)
    target = paths.chunks("section", "AAPL-10K-2025FY")
    target.write_text("corrupted\n")
    again = _run(tmp_path)
    assert again.requests == 0
    assert again.parsed == 0
    assert again.chunked == 1
    assert target.read_text() != "corrupted\n"


def test_interrupted_run_resumes(sec, tmp_path):
    _run(tmp_path)
    paths = DataPaths(tmp_path)
    m = Manifest(paths.manifest)
    del m.data["filings"]["AAPL-10Q-2026Q2"]["steps"]["parse"]
    m.save()
    again = _run(tmp_path)
    assert again.requests == 0
    assert again.parsed == 1 and again.skipped == 2


def test_unknown_ticker_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path, tickers=["NOPE"], include_financebench=False)


# ---------------------------------------------------------------- FinanceBench failures

FB_DOC = "ACME_2023_10K"
FB_URL = f"https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{FB_DOC}.pdf"
FB_LINK = "https://investors.example.com/acme-2023-10k.pdf"


def _pdf_bytes() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 80), "Item 7. Management's Discussion and Analysis", fontsize=14)
    page.insert_text((72, 110), "Revenue grew on strong demand.", fontsize=10)
    data = bytes(doc.tobytes())
    doc.close()
    return data


def _fb_dataset(tmp_path):
    path = tmp_path / "raw" / "financebench" / "financebench_merged.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "doc_name": FB_DOC,
        "company": "Acme Widgets",
        "doc_type": "10k",
        "doc_period": 2023,
        "doc_link": FB_LINK,
    }
    path.write_text(json.dumps(row) + "\n")


def _run_fb(tmp_path, **kw):
    http = HttpCache(tmp_path / "raw", sec_user_agent=UA, sleep=lambda _: None)
    return run(tmp_path, tickers=[], include_financebench=True, workers=1, http=http, **kw)


@respx.mock
def test_invalid_download_is_transient_and_recovers(tmp_path):
    _fb_dataset(tmp_path)
    gh = respx.get(FB_URL).mock(
        side_effect=[
            httpx.Response(200, content=b"version https://git-lfs.github.com/spec/v1\n"),
            httpx.Response(200, content=_pdf_bytes()),
        ]
    )
    respx.get(FB_LINK).mock(return_value=httpx.Response(200, content=b"<html>login</html>"))
    first = _run_fb(tmp_path)
    fid = f"FB-{FB_DOC}"
    assert fid in first.failed
    manifest = Manifest(DataPaths(tmp_path).manifest)
    assert manifest.error(fid) is not None
    assert manifest.error(fid)["permanent"] is False
    assert not list((tmp_path / "raw").rglob(f"{FB_DOC}.pdf"))  # nothing bad cached
    second = _run_fb(tmp_path)  # transient errors are retried without --retry-failed
    assert second.failed == {}
    assert gh.call_count == 2
    assert Manifest(DataPaths(tmp_path).manifest).error(fid) is None


@respx.mock
def test_permanent_failure_needs_retry_failed(tmp_path):
    _fb_dataset(tmp_path)
    gh = respx.get(FB_URL).mock(
        side_effect=[httpx.Response(404), httpx.Response(200, content=_pdf_bytes())]
    )
    respx.get(FB_LINK).mock(return_value=httpx.Response(404))
    first = _run_fb(tmp_path)
    fid = f"FB-{FB_DOC}"
    assert Manifest(DataPaths(tmp_path).manifest).error(fid)["permanent"] is True
    assert fid in first.failed
    skipped = _run_fb(tmp_path)
    assert skipped.requests == 0 and fid in skipped.failed
    retried = _run_fb(tmp_path, retry_failed=True)
    assert retried.failed == {}
    assert gh.call_count == 2


# ---------------------------------------------------------------- corpus outputs and lock


def test_partial_run_keeps_the_whole_corpus(sec, tmp_path):
    _fb_dataset(tmp_path)
    mock = sec[2]
    mock.get(FB_URL).mock(return_value=httpx.Response(200, content=_pdf_bytes()))
    http = HttpCache(tmp_path / "raw", sec_user_agent=UA, sleep=lambda _: None)
    run(tmp_path, tickers=["AAPL"], include_financebench=True, workers=1, http=http)
    paths = DataPaths(tmp_path)
    full = [f["filing_id"] for f in json.loads(paths.filings.read_text())]
    assert len(full) == 4 and f"FB-{FB_DOC}" in full
    _run(tmp_path)  # AAPL only, no FinanceBench
    assert [f["filing_id"] for f in json.loads(paths.filings.read_text())] == full
    report = json.loads((paths.reports / "quality.json").read_text())
    assert report["totals"]["filings"] == 4


def test_live_lock_blocks_a_second_run(tmp_path):
    import os

    lock = DataPaths(tmp_path).lock
    lock.write_text(str(os.getpid()))
    with pytest.raises(RuntimeError, match="holds"):
        run(tmp_path, tickers=[], include_financebench=False)
    assert lock.exists()


def test_stale_lock_is_taken_over(tmp_path):
    lock = DataPaths(tmp_path).lock
    lock.write_text("999999999")
    summary = run(tmp_path, tickers=[], include_financebench=False)
    assert summary.failed == {}
    assert not lock.exists()


def test_cli_exits_non_zero_when_filings_fail(fixtures, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from ledgerlens.cli import app
    from ledgerlens.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SEC_USER_AGENT", UA)
    get_settings.cache_clear()
    try:
        with respx.mock(assert_all_called=False) as mock:
            mock.get(SUB_URL).mock(
                return_value=httpx.Response(
                    200, content=(fixtures / "submissions_aapl.json").read_bytes()
                )
            )
            mock.get(FACTS_URL).mock(return_value=httpx.Response(404))
            mock.get(url__startswith=DOC_PREFIX).mock(return_value=httpx.Response(404))
            result = CliRunner().invoke(
                app, ["ingest", "run", "--tickers", "AAPL", "--no-financebench", "--workers", "1"]
            )
    finally:
        get_settings.cache_clear()
    assert result.exit_code == 1
    assert "failed=3" in result.output
