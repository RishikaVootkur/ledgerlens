"""Streamable HTTP: Host/Origin checks and the guard on non-loopback binds. No sockets:
the Starlette app is driven in process."""

from typing import Any

import pytest
import typer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.testclient import TestClient
from typer.testing import CliRunner

from ledgerlens import mcp_server
from ledgerlens.cli import app

INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}
HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


def post(security: TransportSecuritySettings | None, bind: str, **headers: str) -> Any:
    server = mcp_server.build_server(include_ask=False)
    http = server.streamable_http_app(host=bind, transport_security=security, json_response=True)
    with TestClient(http, base_url="http://127.0.0.1:8765") as client:
        return client.post("/mcp", json=INIT, headers={**HEADERS, **headers})


def test_loopback_bind_rejects_a_foreign_host_header():
    assert post(None, "127.0.0.1", Host="127.0.0.1:8765").status_code == 200
    rejected = post(None, "127.0.0.1", Host="evil.example:8765")
    assert rejected.status_code == 421
    assert post(None, "127.0.0.1", Origin="http://evil.example").status_code == 403


def test_remote_bind_accepts_only_the_allowed_hosts():
    include_ask, security = mcp_server.http_settings(
        "0.0.0.0", allow_remote=True, allowed_hosts=["mcp.example.com:8765"], ask=None
    )
    assert include_ask is False
    assert security is not None
    assert post(security, "0.0.0.0", Host="mcp.example.com:8765").status_code == 200
    assert post(security, "0.0.0.0", Host="evil.example:8765").status_code == 421
    allowed = {"Host": "mcp.example.com:8765", "Origin": "https://mcp.example.com:8765"}
    assert post(security, "0.0.0.0", **allowed).status_code == 200
    foreign = {"Host": "mcp.example.com:8765", "Origin": "https://evil.example"}
    assert post(security, "0.0.0.0", **foreign).status_code == 403


def test_http_settings_defaults():
    assert mcp_server.http_settings("127.0.0.1", False, [], None) == (True, None)
    assert mcp_server.http_settings("localhost", False, [], False) == (False, None)
    include_ask, _ = mcp_server.http_settings("0.0.0.0", True, ["h:8765"], ask=True)
    assert include_ask is True
    with pytest.raises(typer.BadParameter, match="--allow-remote"):
        mcp_server.http_settings("0.0.0.0", False, ["h:8765"], None)
    with pytest.raises(typer.BadParameter, match="Host values"):
        mcp_server.http_settings("0.0.0.0", True, [], None)


@pytest.mark.parametrize(
    "args",
    [
        ["--transport", "http", "--host", "0.0.0.0"],
        ["--transport", "http", "--host", "192.168.1.20", "--allowed-host", "x:1"],
        ["--transport", "http", "--host", "0.0.0.0", "--allow-remote"],
        ["--transport", "carrier-pigeon"],
    ],
)
def test_cli_refuses_unsafe_or_unknown_settings_before_serving(args, monkeypatch):
    monkeypatch.setattr(mcp_server, "build_server", lambda **kw: pytest.fail("server built"))
    result = CliRunner().invoke(app, ["mcp", "serve", *args])
    assert result.exit_code == 2, result.output
