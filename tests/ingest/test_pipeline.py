import json

import httpx
import pytest
import respx

from ledgerlens.ingest.edgar import HttpCache
from ledgerlens.ingest.pipeline import DataPaths, Manifest, run

UA = "Test Agent test@ledgerlens.dev"
SUB_URL = "https://data.sec.gov/submissions/CIK0000320193.json"
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
        docs = mock.get(url__startswith=DOC_PREFIX).mock(
            return_value=httpx.Response(200, content=html)
        )
        yield sub, docs


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
    sub, docs = sec
    first = _run(tmp_path)
    assert len(first.filings) == 3
    assert (first.downloaded, first.parsed, first.chunked, first.failed) == (3, 3, 3, {})
    assert first.requests == 4
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
