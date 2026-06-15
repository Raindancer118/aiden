"""Git-scoped incremental sync.

Uses the git working-tree status to find changed/untracked/deleted files
quickly, then reindexes only those (the per-file content hash still prevents
redundant work). Falls back to a full hash-based reindex when the project is
not a git repository.
"""

from __future__ import annotations

import logging
from pathlib import Path

from codescope.index.indexer import Indexer, ReindexReport
from codescope.index.languages import spec_for_path

log = logging.getLogger(__name__)


def git_changes(root: Path) -> tuple[set[str], set[str]] | None:
    """Return (changed_paths, deleted_paths) as project-relative posix strings.

    ``None`` if ``root`` is not inside a git repository.
    """
    try:
        import pygit2
        from pygit2.enums import FileStatus
    except Exception:  # pragma: no cover
        return None

    repo_path = pygit2.discover_repository(str(root))
    if repo_path is None:
        return None

    repo = pygit2.Repository(repo_path)
    workdir = Path(repo.workdir).resolve() if repo.workdir else root
    root = root.resolve()

    deleted_mask = FileStatus.WT_DELETED | FileStatus.INDEX_DELETED
    ignore_mask = FileStatus.IGNORED | FileStatus.CURRENT

    changed: set[str] = set()
    deleted: set[str] = set()
    for rel_to_repo, flags in repo.status().items():
        if flags & ignore_mask:
            continue
        abs_path = (workdir / rel_to_repo).resolve()
        try:
            rel = abs_path.relative_to(root).as_posix()
        except ValueError:
            continue  # outside the indexed root
        if flags & deleted_mask:
            deleted.add(rel)
        else:
            changed.add(rel)
    return changed, deleted


def sync_incremental(indexer: Indexer, *, embeddings: bool = True) -> ReindexReport:
    """Fast incremental update scoped to git working-tree changes.

    Falls back to a full (hash-based) reindex for non-git projects.
    """
    changes = git_changes(indexer.root)
    if changes is None:
        return indexer.reindex(force=False, embeddings=embeddings)
    changed, deleted = changes
    paths = [p for p in changed if spec_for_path(p) is not None]
    paths.extend(deleted)
    if not paths:
        return indexer.reindex_paths([], embeddings=embeddings)
    return indexer.reindex_paths(paths, force=False, embeddings=embeddings)
