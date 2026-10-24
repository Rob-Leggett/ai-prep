import asyncio

import requests

import mcp_server


def test_registers_expected_tools():
    names = sorted(t.name for t in asyncio.run(mcp_server.mcp.list_tools()))
    assert names == ["ask", "health", "predict"]


def test_health_reports_unreachable_api(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(mcp_server.requests, "get", boom)
    result = mcp_server.health()
    assert result["ok"] is False
    assert "refused" in result["error"]
