"""Telling an agent what the user did in the explorer.

The explorer can rebuild an index or start a watcher while an agent is
working in the same repository. The agent needs to know that happened -- its
last search may now be stale -- but most of the time the right response is
none at all, so the notice says so explicitly rather than reading like a task.

Delivery rides along with whatever tool the agent calls next: unseen events
are appended to that tool's result. There is no polling and no interruption,
and each event is handed over exactly once.
"""

from __future__ import annotations

import time

_HEADER = "[codescope explorer] The user ran the following in the web UI while you were working:"
_FOOTER = (
    "This is a notification, not a request. React only if it changes what you are doing "
    "(for example, results you read before a reindex may be stale). Otherwise ignore it and carry on."
)


def format_events(events: list) -> str:  # type: ignore[type-arg]
    """Render explorer events as a notice appended to a tool result."""
    if not events:
        return ""
    lines = [_HEADER]
    for event in events:
        when = time.strftime("%H:%M:%S", time.localtime(event.at))
        outcome = f" - {event.detail}" if event.detail else ""
        failed = " (FAILED)" if event.status == "failed" else ""
        lines.append(f"  - {when} {event.action} on {event.project}{failed}{outcome}")
    lines.append(_FOOTER)
    return "\n".join(lines)


def pending_notice() -> str:
    """The notice for events not yet handed to the agent, or ``""``.

    Reads only from an explorer hosted by *this* process. When the explorer
    runs in another process (a second MCP server, or the CLI), that process
    delivers the notices to its own agent, which is the one that shares the
    session with the user's browser.
    """
    try:
        from codescope.web.explorer import hosted_registry

        registry = hosted_registry()
        if registry is None:
            return ""
        return format_events(registry.actions.take_unseen())
    except Exception:  # pragma: no cover - a notice must never break a tool
        return ""


def append_notice(result: str) -> str:
    """Append any pending notice to a tool result."""
    notice = pending_notice()
    if not notice:
        return result
    return f"{result}\n\n{notice}"
