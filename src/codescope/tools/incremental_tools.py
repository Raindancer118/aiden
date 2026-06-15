"""Incremental indexing tools: git-scoped sync and a live file watcher."""

from __future__ import annotations

from dataclasses import asdict

from codescope.index.incremental import sync_incremental
from codescope.index.indexer import Indexer
from codescope.index.watcher import start_watcher, stop_watcher, watcher_status
from serena.tools import Tool


class SyncIndexTool(Tool):
    """
    Quickly bring the index up to date with the current working tree.

    Uses git to find changed/added/deleted files and reindexes only those
    (falling back to a fast hash-based full pass for non-git projects). Much
    cheaper than a full reindex; run it after editing files when no live
    watcher is active.
    """

    def apply(self, embeddings: bool = True) -> str:
        """
        :param embeddings: also (re)compute embeddings for changed symbols.
        :return: JSON summary with indexed/skipped/removed counts and totals.
        """
        report = sync_incremental(Indexer(self.get_project_root()), embeddings=embeddings)
        return self._to_json(
            {
                "indexed": report.indexed,
                "skipped_unchanged": report.skipped_unchanged,
                "removed": report.removed,
                "errors": report.errors,
                "duration_s": report.duration_s,
                "totals": asdict(report.stats),
            }
        )


class WatchStartTool(Tool):
    """
    Start a background file watcher that keeps the index fresh in real time.

    The watcher runs in a daemon thread, debounces rapid edits, and reindexes
    only the affected source files. Safe to call repeatedly (idempotent).
    """

    def apply(self, embeddings: bool = True) -> str:
        """
        :param embeddings: keep embeddings updated for changed symbols.
        :return: JSON watcher status.
        """
        return self._to_json(start_watcher(self.get_project_root(), embeddings=embeddings))


class WatchStopTool(Tool):
    """Stop the background file watcher for the active project, if running."""

    def apply(self) -> str:
        """
        :return: JSON final watcher status.
        """
        return self._to_json(stop_watcher(self.get_project_root()))


class WatchStatusTool(Tool):
    """Report the status of the background file watcher for the active project."""

    def apply(self) -> str:
        """
        :return: JSON watcher status (running, batches processed, uptime, ...).
        """
        return self._to_json(watcher_status(self.get_project_root()))
