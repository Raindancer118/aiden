"""Persistent MCP client for the Diary memory backend.

Codescope deliberately talks to Diary through its public MCP interface. It
does not import Diary internals or access Diary's database directly.
"""

from __future__ import annotations

import atexit
import os
import re
import shlex
import sys
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from threading import Event as ThreadEvent
from threading import RLock
from typing import Any, Optional

import anyio
from anyio.from_thread import BlockingPortal, start_blocking_portal
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent


class DiaryBridgeError(RuntimeError):
    """Failure while communicating with the mandatory Diary backend."""


@dataclass(frozen=True)
class DiaryBridgeSettings:
    """Process and timeout settings for the Diary MCP server."""

    command: str
    args: tuple[str, ...] = ()
    timeout_seconds: float = 30.0

    @classmethod
    def from_environment(cls) -> "DiaryBridgeSettings":
        """Build settings from Codescope's Diary environment variables."""
        command_line = os.environ.get("CODESCOPE_DIARY_COMMAND", "diary-mcp-local")
        parts = shlex.split(command_line)
        if not parts:
            raise DiaryBridgeError("CODESCOPE_DIARY_COMMAND must contain an executable")

        timeout_raw = os.environ.get("CODESCOPE_DIARY_TIMEOUT_SECONDS", "30")
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise DiaryBridgeError("CODESCOPE_DIARY_TIMEOUT_SECONDS must be a number") from exc
        if timeout_seconds <= 0:
            raise DiaryBridgeError("CODESCOPE_DIARY_TIMEOUT_SECONDS must be positive")

        return cls(command=parts[0], args=tuple(parts[1:]), timeout_seconds=timeout_seconds)

    def server_parameters(self) -> StdioServerParameters:
        """Return parameters for starting the configured Diary MCP server."""
        return StdioServerParameters(command=self.command, args=list(self.args))


class DiaryMcpBridge:
    """Long-lived synchronous facade over Diary's asynchronous MCP session."""

    def __init__(self, settings: DiaryBridgeSettings):
        self._settings = settings
        self._lock = RLock()
        self._portal_context: Optional[AbstractContextManager[BlockingPortal]] = None
        self._portal: Optional[BlockingPortal] = None
        self._serve_future: Optional[Future[None]] = None
        self._ready = ThreadEvent()
        self._stop_event: Optional[anyio.Event] = None
        self._startup_error: Optional[BaseException] = None
        self._session: Optional[ClientSession] = None

    async def _serve(self) -> None:
        """Own Diary's context managers for the complete bridge lifetime."""
        try:
            async with stdio_client(self._settings.server_parameters(), errlog=sys.stderr) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    stop_event = anyio.Event()
                    self._session = session
                    self._stop_event = stop_event
                    self._ready.set()
                    await stop_event.wait()
        except BaseException as exc:
            self._startup_error = exc
            raise
        finally:
            self._session = None
            self._stop_event = None
            self._ready.set()

    def _signal_stop(self) -> None:
        """Wake the owner task so it closes Diary in the same async task."""
        stop_event = self._stop_event
        if stop_event is not None:
            stop_event.set()

    def _start_locked(self) -> BlockingPortal:
        """Start the background event loop while holding ``self._lock``."""
        if self._portal is not None:
            return self._portal

        portal_context = start_blocking_portal(name="codescope-diary-mcp")
        portal = portal_context.__enter__()
        self._ready = ThreadEvent()
        self._startup_error = None
        serve_future = portal.start_task_soon(self._serve, name="codescope-diary-session")
        ready = self._ready.wait(timeout=self._settings.timeout_seconds)
        try:
            if not ready:
                raise TimeoutError(f"Diary did not initialize within {self._settings.timeout_seconds:g} seconds")
            if self._startup_error is not None:
                raise self._startup_error
            if self._session is None:
                raise DiaryBridgeError("Diary stopped before its MCP session became ready")
        except BaseException as exc:
            serve_future.cancel()
            portal_context.__exit__(type(exc), exc, exc.__traceback__)
            raise DiaryBridgeError(f"Diary is required but could not be started with {self._settings.command!r}: {exc}") from exc

        self._portal_context = portal_context
        self._portal = portal
        self._serve_future = serve_future
        return portal

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call one Diary tool and normalize its textual result."""
        session = self._session
        if session is None:
            raise DiaryBridgeError("Diary MCP session is not connected")

        result = await session.call_tool(
            tool_name,
            arguments,
            read_timeout_seconds=timedelta(seconds=self._settings.timeout_seconds),
        )
        if result.isError:
            details = "\n".join(item.text for item in result.content if isinstance(item, TextContent))
            raise DiaryBridgeError(details or f"Diary tool {tool_name!r} failed")

        if result.structuredContent is not None:
            structured_result = result.structuredContent.get("result")
            if isinstance(structured_result, str):
                return structured_result

        text_parts = [item.text for item in result.content if isinstance(item, TextContent)]
        if text_parts:
            return "\n".join(text_parts)
        raise DiaryBridgeError(f"Diary tool {tool_name!r} returned no textual result")

    def call(self, tool_name: str, arguments: Optional[dict[str, Any]] = None) -> str:
        """Call a Diary MCP tool from Codescope's synchronous tool runtime."""
        with self._lock:
            portal = self._start_locked()
            try:
                return portal.call(self._call_tool, tool_name, arguments or {})
            except DiaryBridgeError:
                raise
            except BaseException as exc:
                self._close_locked()
                raise DiaryBridgeError(f"Diary MCP call {tool_name!r} failed: {exc}") from exc

    def _close_locked(self) -> None:
        """Close resources while holding ``self._lock``."""
        portal = self._portal
        portal_context = self._portal_context
        serve_future = self._serve_future
        self._portal = None
        self._portal_context = None
        self._serve_future = None

        if portal is not None:
            try:
                portal.call(self._signal_stop)
            except BaseException:
                pass
        if serve_future is not None:
            try:
                serve_future.result(timeout=self._settings.timeout_seconds)
            except FutureTimeoutError:
                serve_future.cancel()
            except BaseException:
                pass
        if portal_context is not None:
            portal_context.__exit__(None, None, None)

    def close(self) -> None:
        """Close the persistent Diary MCP session, if it was started."""
        with self._lock:
            self._close_locked()


class DiaryBridgeRegistry:
    """Process-wide owner of the persistent Diary bridge."""

    _bridge: Optional[DiaryMcpBridge] = None
    _lock = RLock()

    @classmethod
    def get(cls) -> DiaryMcpBridge:
        """Return the bridge, creating it lazily."""
        with cls._lock:
            if cls._bridge is None:
                cls._bridge = DiaryMcpBridge(DiaryBridgeSettings.from_environment())
            return cls._bridge

    @classmethod
    def close(cls) -> None:
        """Close and forget the bridge."""
        with cls._lock:
            bridge = cls._bridge
            cls._bridge = None
        if bridge is not None:
            bridge.close()


def get_diary_bridge() -> DiaryMcpBridge:
    """Return Codescope's process-wide Diary MCP bridge."""
    return DiaryBridgeRegistry.get()


def close_diary_bridge() -> None:
    """Close and forget the process-wide Diary MCP bridge."""
    DiaryBridgeRegistry.close()


def project_slug(project_root: str | Path) -> str:
    """Derive a Diary-compatible project slug from a project directory."""
    slug = re.sub(r"[^a-z0-9]+", "-", Path(project_root).name.lower()).strip("-")
    if not slug:
        raise DiaryBridgeError(f"Cannot derive a Diary project slug from {str(project_root)!r}")
    return slug


atexit.register(close_diary_bridge)
