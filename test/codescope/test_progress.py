"""Index runs must be observable while they run.

"Running" for four minutes is indistinguishable from "hung", so a run publishes
its phase and counts and the explorer reports them.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from codescope.index import progress
from codescope.index.embed import HashingEmbedder
from codescope.index.indexer import Indexer


@pytest.fixture(autouse=True)
def _clean_registry(tmp_path: Path):
    yield
    progress.clear(tmp_path)


def _project(root: Path, files: int = 6) -> None:
    for i in range(files):
        (root / f"m{i}.py").write_text(f"def f{i}():\n    return {i}\n", encoding="utf-8")


def test_no_run_reported_before_anything_happens(tmp_path: Path) -> None:
    assert progress.snapshot(tmp_path) is None
    assert progress.is_running(tmp_path) is False


def test_reindex_publishes_a_finished_run(tmp_path: Path) -> None:
    _project(tmp_path)
    Indexer(tmp_path).reindex(embedder=HashingEmbedder(dim=32))

    snap = progress.snapshot(tmp_path)
    assert snap is not None
    assert snap["operation"] == "reindex"
    assert snap["phase"] == "done"
    assert snap["running"] is False
    assert snap["error"] is None
    assert snap["elapsed_s"] >= 0


def test_progress_is_visible_while_the_run_is_in_flight(tmp_path: Path) -> None:
    """The whole point: a reader sees counts climb, not just "running"."""
    _project(tmp_path, files=40)
    seen: list[dict] = []
    started = threading.Event()

    indexer = Indexer(tmp_path)
    original = indexer._index_batch

    def watched(*args, **kwargs):  # type: ignore[no-untyped-def]
        started.set()
        snap = progress.snapshot(tmp_path)
        if snap:
            seen.append(snap)
        return original(*args, **kwargs)

    indexer._index_batch = watched  # type: ignore[method-assign]
    indexer.reindex(embedder=HashingEmbedder(dim=32))

    assert started.is_set()
    assert seen, "no progress was published during the run"
    assert seen[0]["running"] is True
    assert seen[0]["phase"] == "parsing"
    assert seen[0]["total"] == 40


def test_a_failing_run_is_reported_as_failed(tmp_path: Path) -> None:
    _project(tmp_path)
    indexer = Indexer(tmp_path)

    def boom(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("disk on fire")

    indexer._index_batch = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        indexer.reindex(embedder=HashingEmbedder(dim=32))

    snap = progress.snapshot(tmp_path)
    assert snap is not None
    assert snap["phase"] == "failed"
    assert snap["running"] is False
    assert "disk on fire" in (snap["error"] or "")


def test_percent_and_eta_are_derived_from_the_counts() -> None:
    run = progress.IndexProgress(root="/tmp/x", operation="reindex")
    run.set_phase("parsing", total=100)
    run.advance(25)

    snap = run.snapshot()
    assert snap["percent"] == 25.0
    assert snap["eta_s"] is not None and snap["eta_s"] >= 0


def test_nested_tracking_does_not_reset_the_outer_run(tmp_path: Path) -> None:
    with progress.track(tmp_path, "reindex") as outer:
        outer.set_phase("parsing", total=10)
        outer.advance(4)
        with progress.track(tmp_path, "sync") as inner:
            assert inner is outer
        assert progress.snapshot(tmp_path)["done"] == 4  # type: ignore[index]
    assert progress.snapshot(tmp_path)["phase"] == "done"  # type: ignore[index]
