"""Tests for incremental indexing: reindex_paths, git sync, and the watcher."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from codescope.index.incremental import git_changes, sync_incremental
from codescope.index.indexer import Indexer
from codescope.index.store import IndexStore


def _symbol_names(db: Path) -> set[str]:
    store = IndexStore(db)
    try:
        return {r[0] for r in store.conn.execute("SELECT name FROM symbols")}
    finally:
        store.close()


def test_reindex_paths_updates_and_prunes(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def alpha():\n    pass\n")
    (tmp_path / "b.py").write_text("def beta():\n    pass\n")
    db = tmp_path / "idx" / "index.db"
    idx = Indexer(tmp_path, db_path=db)
    idx.reindex(embeddings=False)
    assert _symbol_names(db) == {"alpha", "beta"}

    # Modify a.py, add c.py, delete b.py, then reindex just those paths.
    (tmp_path / "a.py").write_text("def alpha():\n    pass\n\ndef alpha2():\n    pass\n")
    (tmp_path / "c.py").write_text("def gamma():\n    pass\n")
    (tmp_path / "b.py").unlink()
    report = idx.reindex_paths(["a.py", "b.py", "c.py"], embeddings=False)
    assert report.removed == 1
    assert _symbol_names(db) == {"alpha", "alpha2", "gamma"}


def test_git_changes_detects_untracked(tmp_path: Path) -> None:
    pygit2 = pytest.importorskip("pygit2")
    pygit2.init_repository(str(tmp_path))
    (tmp_path / "mod.py").write_text("def thing():\n    pass\n")
    changes = git_changes(tmp_path)
    assert changes is not None
    changed, _deleted = changes
    assert "mod.py" in changed


def test_sync_incremental_non_git_falls_back(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("def ex():\n    pass\n")
    db = tmp_path / "idx" / "index.db"
    report = sync_incremental(Indexer(tmp_path, db_path=db), embeddings=False)
    assert report.errors == 0
    assert "ex" in _symbol_names(db)


def test_sync_incremental_git_scoped(tmp_path: Path) -> None:
    pygit2 = pytest.importorskip("pygit2")
    pygit2.init_repository(str(tmp_path))
    (tmp_path / "svc.py").write_text("def serve():\n    pass\n")
    db = tmp_path / "idx" / "index.db"
    report = sync_incremental(Indexer(tmp_path, db_path=db), embeddings=False)
    assert report.indexed == 1
    assert "serve" in _symbol_names(db)


def test_watcher_picks_up_changes(tmp_path: Path) -> None:
    from codescope.index.watcher import start_watcher, stop_watcher, watcher_status

    (tmp_path / "seed.py").write_text("def seed():\n    pass\n")
    db = tmp_path / ".serena" / "codescope" / "index.db"
    try:
        status = start_watcher(tmp_path, embeddings=False)
        assert status["running"] is True

        time.sleep(0.5)
        (tmp_path / "live.py").write_text("def live_symbol():\n    pass\n")

        deadline = time.time() + 12
        found = False
        while time.time() < deadline:
            if db.exists() and "live_symbol" in _symbol_names(db):
                found = True
                break
            time.sleep(0.3)
        assert found, "watcher did not index the new file in time"
        assert watcher_status(tmp_path)["running"] is True
    finally:
        stop_watcher(tmp_path)
    assert watcher_status(tmp_path)["running"] is False


def test_incremental_tools_registered() -> None:
    from codescope.cli import register_codescope_tools

    register_codescope_tools()
    from serena.tools import ToolRegistry

    names = ToolRegistry().get_tool_names()
    for t in ("sync_index", "watch_start", "watch_stop", "watch_status"):
        assert t in names
