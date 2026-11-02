import httpx
import pytest
import respx

from ledgerlens.config import Settings
from ledgerlens.eval.datasets import sec


@pytest.fixture
def settings(monkeypatch, tmp_path):
    s = Settings(
        _env_file=None, sec_user_agent="Test Runner test@ledgerlens.dev", data_dir=tmp_path
    )
    monkeypatch.setattr(sec, "get_settings", lambda: s)
    monkeypatch.setattr(sec, "MAX_RPS", 1000.0)
    return s


@respx.mock
def test_fetch_caches_and_sends_user_agent(settings, tmp_path):
    url = sec.companyfacts_url("0000021344")
    route = respx.get(url).mock(return_value=httpx.Response(200, json={"facts": {}}))
    assert sec.companyfacts("0000021344") == {"facts": {}}
    assert route.calls[0].request.headers["User-Agent"] == "Test Runner test@ledgerlens.dev"
    cached = tmp_path / "raw" / "sec_api" / "api" / "xbrl" / "companyfacts" / "CIK0000021344.json"
    assert cached.exists()
    assert sec.companyfacts("0000021344") == {"facts": {}}
    assert route.call_count == 1  # second read comes from disk


@respx.mock
def test_fetch_retries_on_429(settings, monkeypatch):
    monkeypatch.setattr(sec.time, "sleep", lambda _s: None)
    url = sec.submissions_url("0000021344")
    route = respx.get(url).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json={"ok": True})]
    )
    assert sec.submissions("0000021344") == {"ok": True}
    assert route.call_count == 2


def test_placeholder_user_agent_is_rejected(monkeypatch, tmp_path):
    s = Settings(_env_file=None, sec_user_agent="", data_dir=tmp_path)
    monkeypatch.setattr(sec, "get_settings", lambda: s)
    with pytest.raises(ValueError, match="SEC_USER_AGENT"):
        sec.submissions("0000021344")
