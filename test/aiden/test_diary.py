"""Tests for AIDEN's mandatory Diary MCP backend."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from aiden.diary import DiaryBridgeError, DiaryBridgeSettings, DiaryMcpBridge, project_slug


def test_settings_parse_command_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIDEN_DIARY_COMMAND", 'python -m "diary server"')
    monkeypatch.setenv("AIDEN_DIARY_TIMEOUT_SECONDS", "12.5")

    settings = DiaryBridgeSettings.from_environment()

    assert settings.command == "python"
    assert settings.args == ("-m", "diary server")
    assert settings.timeout_seconds == 12.5


@pytest.mark.parametrize("value", ["", "zero", "0", "-1"])
def test_settings_reject_invalid_timeout(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("AIDEN_DIARY_TIMEOUT_SECONDS", value)

    with pytest.raises(DiaryBridgeError):
        DiaryBridgeSettings.from_environment()


def test_project_slug_normalizes_directory_name() -> None:
    assert project_slug("/tmp/SE Projects/AIDEN") == "aiden"


def test_bridge_calls_real_mcp_transport(tmp_path: Path) -> None:
    server_script = tmp_path / "fake_diary.py"
    server_script.write_text(
        """
from mcp.server.fastmcp import FastMCP

server = FastMCP("fake-diary")

@server.tool()
def memory_get(path: str) -> str:
    return f"diary:{path}"

if __name__ == "__main__":
    server.run()
""".lstrip(),
        encoding="utf-8",
    )
    bridge = DiaryMcpBridge(DiaryBridgeSettings(command=sys.executable, args=(str(server_script),), timeout_seconds=10))

    try:
        assert bridge.call("memory_get", {"path": "/projects/aiden/status"}) == "diary:/projects/aiden/status"
        assert bridge.call("memory_get", {"path": "/feedback/arbeitsstil"}) == "diary:/feedback/arbeitsstil"
    finally:
        bridge.close()


def test_bridge_has_no_local_fallback(tmp_path: Path) -> None:
    missing = tmp_path / "missing-diary-mcp"
    bridge = DiaryMcpBridge(DiaryBridgeSettings(command=str(missing), timeout_seconds=1))

    with pytest.raises(DiaryBridgeError, match="Diary is required"):
        bridge.call("memory_context")
