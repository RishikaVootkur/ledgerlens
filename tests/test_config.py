import logging

import pytest
from typer.testing import CliRunner

from ledgerlens import __version__
from ledgerlens.cli import app
from ledgerlens.config import Settings, project_root


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "Test Agent test@ledgerlens.dev")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_dummy")
    s = Settings(_env_file=None)
    assert s.sec_user_agent == "Test Agent test@ledgerlens.dev"
    assert s.groq_api_key is not None
    # Secrets never render in repr or str.
    assert "gsk_dummy" not in repr(s)
    assert "gsk_dummy" not in str(s.groq_api_key)


def test_log_level_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "debug")
    assert Settings(_env_file=None).log_level == "DEBUG"
    assert logging.getLevelName("DEBUG") == logging.DEBUG


def test_log_level_rejects_unknown(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "loud")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


@pytest.mark.parametrize("ua", ["", "LedgerLens", "Name contact@example.com"])
def test_sec_user_agent_required(monkeypatch, ua):
    monkeypatch.setenv("SEC_USER_AGENT", ua)
    with pytest.raises(ValueError):
        Settings(_env_file=None).require_sec_user_agent()


def test_project_root_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LEDGERLENS_HOME", str(tmp_path))
    assert project_root() == tmp_path.resolve()


def test_cli_version():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output
