from ledgerlens.config import Settings


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "Test Agent test@example.com")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_dummy")
    s = Settings(_env_file=None)
    assert s.sec_user_agent == "Test Agent test@example.com"
    assert s.groq_api_key is not None
    # Secrets never render in repr or str.
    assert "gsk_dummy" not in repr(s)
    assert "gsk_dummy" not in str(s.groq_api_key)
