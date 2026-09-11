"""Git-scoped incremental sync.

Uses the git working-tree status to find changed/untracked/deleted files
quickly, then reindexes only those (the per-file content hash still prevents
redundant work). Falls back to a full hash-based reindex when the project is
not a git repository.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from aiden.index.indexer import Indexer, ReindexReport
from aiden.index.languages import spec_for_path
from aiden.index.store import IndexStore

log = logging.getLogger(__name__)

_GIT_HEAD_META_KEY = "git_head"
_GIT_DIRTY_PATHS_META_KEY = "git_dirty_paths"


@dataclass(frozen=True, slots=True)
class _GitSyncState:
    head: str | None
    dirty_paths: frozenset[str] | None


def _git_head(root: Path) -> str | None:
    """Return the current commit id, or None for an unborn/non-git repository."""
    try:
        import pygit2
    except Exception:  # pragma: no cover
        return None

    repo_path = pygit2.discover_repository(str(root))
    if repo_path is None:
        return None
    repo = pygit2.Repository(repo_path)
    try:
        return str(repo.head.target)
    except (KeyError, pygit2.GitError):
        return None


def _indexed_git_state(indexer: Indexer) -> _GitSyncState:
    store = IndexStore(indexer.db_path)
    try:
        head = store.get_meta(_GIT_HEAD_META_KEY)
        encoded_paths = store.get_meta(_GIT_DIRTY_PATHS_META_KEY)
        if encoded_paths is None:
            return _GitSyncState(head=head, dirty_paths=None)
        try:
            decoded_paths = json.loads(encoded_paths)
        except (TypeError, json.JSONDecodeError):
            log.warning("Ignoring invalid git dirty-path metadata in %s", indexer.db_path)
            return _GitSyncState(head=head, dirty_paths=None)
        if not isinstance(decoded_paths, list) or not all(isinstance(path, str) for path in decoded_paths):
            log.warning("Ignoring invalid git dirty-path metadata in %s", indexer.db_path)
            return _GitSyncState(head=head, dirty_paths=None)
        return _GitSyncState(head=head, dirty_paths=frozenset(decoded_paths))
    finally:
        store.close()


def _record_git_state(indexer: Indexer, head: str | None, dirty_paths: frozenset[str]) -> None:
    store = IndexStore(indexer.db_path)
    try:
        if head is None:
            store.conn.execute("DELETE FROM meta WHERE key=?", (_GIT_HEAD_META_KEY,))
        else:
            store.set_meta(_GIT_HEAD_META_KEY, head)
        store.set_meta(_GIT_DIRTY_PATHS_META_KEY, json.dumps(sorted(dirty_paths)))
        store.commit()
    finally:
        store.close()


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
    current_dirty_paths = frozenset(changed | deleted)
    current_head = _git_head(indexer.root)
    indexed_state = _indexed_git_state(indexer)
    previous_dirty_paths = indexed_state.dirty_paths or frozenset()
    head_changed = current_head != indexed_state.head and (current_head is not None or indexed_state.head is not None)
    gitignore_changed = ".gitignore" in current_dirty_paths or ".gitignore" in previous_dirty_paths

    if head_changed or indexed_state.dirty_paths is None or gitignore_changed:
        report = indexer.reindex(force=False, embeddings=embeddings)
    else:
        paths = [p for p in changed if spec_for_path(p) is not None]
        paths.extend(deleted)
        # A path that was dirty during the previous sync but is clean now was
        # restored/stashed without changing HEAD. Re-read it from the worktree.
        paths.extend(previous_dirty_paths - current_dirty_paths)
        report = indexer.reindex_paths(paths, force=False, embeddings=embeddings)

    if report.errors == 0:
        _record_git_state(indexer, current_head, current_dirty_paths)
    return report
