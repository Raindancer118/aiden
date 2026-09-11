"""Live progress for long-running index work.

Indexing a large repository takes minutes, and the only signal the web UI had
was "running". That is indistinguishable from "hung", so a run publishes its
phase and counts here while it works and any reader -- the explorer, the
``index_status`` tool, the CLI -- can poll it.

State is per process and per project root. The explorer runs the actions it
starts in its own process, so its own runs are always visible; runs started
elsewhere (an MCP tool call in the agent's process) are visible to that
process. Finished runs are kept for a short grace period so a UI that polls
every second still gets to show the result.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

#: how long a finished run stays readable, in seconds
RETAIN_FINISHED_S = 90.0

_LOCK = threading.Lock()
_RUNS: dict[str, "IndexProgress"] = {}


def _key(root: str | Path) -> str:
    return str(Path(root).resolve())


@dataclass
class IndexProgress:
    """One index run, as it happens."""

    root: str
    operation: str
    phase: str = "starting"
    detail: str = ""
    done: int = 0
    total: int | None = None
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None

    @property
    def running(self) -> bool:
        return self.finished_at is None

    def set_phase(self, phase: str, *, total: int | None = None, detail: str = "") -> None:
        with _LOCK:
            self.phase = phase
            self.total = total
            self.done = 0
            self.detail = detail
            self.updated_at = time.time()

    def advance(self, by: int = 1, *, detail: str | None = None) -> None:
        with _LOCK:
            self.done += by
            if detail is not None:
                self.detail = detail
            self.updated_at = time.time()

    def finish(self, detail: str = "", error: str | None = None) -> None:
        with _LOCK:
            self.phase = "failed" if error else "done"
            self.detail = error or detail
            self.error = error
            self.finished_at = time.time()
            self.updated_at = self.finished_at

    def snapshot(self) -> dict:
        with _LOCK:
            elapsed = (self.finished_at or time.time()) - self.started_at
            percent: float | None = None
            eta_s: float | None = None
            if self.total:
                percent = round(min(100.0, 100.0 * self.done / self.total), 1)
                if self.done and self.running:
                    eta_s = round(max(0.0, elapsed * (self.total - self.done) / self.done), 1)
            elif not self.running:
                percent = 100.0
            return {
                "root": self.root,
                "operation": self.operation,
                "phase": self.phase,
                "detail": self.detail,
                "done": self.done,
                "total": self.total,
                "percent": percent,
                "running": self.running,
                "elapsed_s": round(elapsed, 1),
                "eta_s": eta_s,
                "started_at": self.started_at,
                "updated_at": self.updated_at,
                "finished_at": self.finished_at,
                "error": self.error,
            }


def current(root: str | Path) -> IndexProgress | None:
    """The run for ``root``, if one is running or recently finished."""
    key = _key(root)
    with _LOCK:
        run = _RUNS.get(key)
        if run is None:
            return None
        stale = run.finished_at is not None and time.time() - run.finished_at > RETAIN_FINISHED_S
    if stale:
        with _LOCK:
            if _RUNS.get(key) is run:
                del _RUNS[key]
        return None
    return run


def snapshot(root: str | Path) -> dict | None:
    run = current(root)
    return run.snapshot() if run is not None else None


def is_running(root: str | Path) -> bool:
    run = current(root)
    return run is not None and run.running


def clear(root: str | Path) -> None:
    """Drop the recorded run (used by tests)."""
    with _LOCK:
        _RUNS.pop(_key(root), None)


@contextmanager
def track(root: str | Path, operation: str) -> Iterator[IndexProgress]:
    """Publish progress for one run, and always mark it finished.

    A run already in flight for the same root is *not* replaced: nested calls
    (the watcher reindexing inside its own run, say) report into the outer run
    rather than resetting it.
    """
    key = _key(root)
    with _LOCK:
        existing = _RUNS.get(key)
        nested = existing is not None and existing.running
        if nested:
            assert existing is not None
            run = existing
        else:
            run = IndexProgress(root=key, operation=operation)
            _RUNS[key] = run
    if nested:
        yield run
        return
    try:
        yield run
    except BaseException as e:
        run.finish(error=f"{type(e).__name__}: {e}"[:300])
        raise
    else:
        if run.running:
            run.finish()
