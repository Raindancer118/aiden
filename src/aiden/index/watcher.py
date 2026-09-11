"""Background file watcher that keeps the index fresh in real time.

Uses ``watchfiles`` (Rust notify backend, debounced) in a daemon thread. On
each debounced batch of changes it reindexes only the affected source files via
``Indexer.reindex_paths``. Watchers are tracked in a per-database registry so
the MCP tools can start, stop, and query them across tool calls.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from aiden.index.indexer import _DEFAULT_IGNORE_DIRS as IGNORE_DIRS
from aiden.index.indexer import Indexer
from aiden.index.languages import spec_for_path

if TYPE_CHECKING:
    from watchfiles.filters import DefaultFilter

log = logging.getLogger(__name__)

_WATCHERS: dict[str, "IndexWatcher"] = {}
_REGISTRY_LOCK = threading.Lock()


class IndexWatcher:
    def __init__(self, project_root: str | Path, *, embeddings: bool = True, db_path: str | Path | None = None):
        self.indexer = Indexer(project_root, db_path=db_path)
        self.root = self.indexer.root
        self.embeddings = embeddings
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.batches = 0
        self.files_reindexed = 0
        self.last_event_at: float | None = None
        self.started_at: float | None = None
        self.error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"aiden-watch:{self.root.name}", daemon=True)
        self.started_at = time.time()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    def _build_filter(self) -> "DefaultFilter":
        from watchfiles.filters import DefaultFilter

        extra = tuple(d for d in IGNORE_DIRS if d not in DefaultFilter.ignore_dirs)

        class _Filter(DefaultFilter):
            ignore_dirs = (*DefaultFilter.ignore_dirs, *extra)

        return _Filter()

    def _run(self) -> None:
        try:
            from watchfiles import watch

            for batch in watch(self.root, watch_filter=self._build_filter(), stop_event=self._stop):
                self.last_event_at = time.time()
                self.batches += 1
                rels: list[str] = []
                refresh_all = False
                for _change, raw_path in batch:
                    p = Path(raw_path)
                    try:
                        rel = p.resolve().relative_to(self.root).as_posix()
                    except ValueError:
                        continue
                    if rel == ".gitignore":
                        refresh_all = True
                        continue
                    # Keep deletions (path may be gone) and supported source files.
                    if not p.exists() or spec_for_path(rel) is not None:
                        rels.append(rel)
                if not rels and not refresh_all:
                    continue
                try:
                    if refresh_all:
                        report = self.indexer.reindex(force=False, embeddings=self.embeddings)
                    else:
                        report = self.indexer.reindex_paths(rels, force=False, embeddings=self.embeddings)
                    self.files_reindexed += report.indexed + report.removed
                    self.error = None
                except Exception as e:  # pragma: no cover - defensive
                    self.error = str(e)
                    log.warning("Watcher reindex failed: %s", e)
        except Exception as e:  # pragma: no cover
            self.error = str(e)
            log.warning("Watcher stopped with error: %s", e)

    def status(self) -> dict:
        return {
            "running": self.running,
            "root": str(self.root),
            "embeddings": self.embeddings,
            "batches": self.batches,
            "files_reindexed": self.files_reindexed,
            "uptime_s": round(time.time() - self.started_at, 1) if self.started_at else None,
            "last_event_at": self.last_event_at,
            "error": self.error,
        }


def _key(project_root: str | Path) -> str:
    return str(Path(project_root).resolve())


def start_watcher(project_root: str | Path, *, embeddings: bool = True) -> dict:
    with _REGISTRY_LOCK:
        key = _key(project_root)
        existing = _WATCHERS.get(key)
        if existing and existing.running:
            return existing.status()
        watcher = IndexWatcher(project_root, embeddings=embeddings)
        watcher.start()
        _WATCHERS[key] = watcher
        return watcher.status()


def stop_watcher(project_root: str | Path) -> dict:
    with _REGISTRY_LOCK:
        key = _key(project_root)
        watcher = _WATCHERS.get(key)
        if watcher is None:
            return {"running": False, "root": key, "note": "no watcher registered"}
        watcher.stop()
        status = watcher.status()
        _WATCHERS.pop(key, None)
        return status


def watcher_status(project_root: str | Path) -> dict:
    with _REGISTRY_LOCK:
        watcher = _WATCHERS.get(_key(project_root))
        if watcher is None:
            return {"running": False, "root": _key(project_root), "note": "no watcher registered"}
        return watcher.status()
