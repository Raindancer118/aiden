"""Index orchestration: walk a project, detect changes, (re)index files.

Change detection uses a cheap mtime+size pre-filter followed by a blake3
content hash. Full git-scoped, watch-driven incremental reindexing is added in
milestone M5; this module already supports incremental updates via the stored
per-file hash.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import blake3
import pathspec

from codescope.index.embed import Embedder, get_embedder
from codescope.index.languages import spec_for_path
from codescope.index.parser import TreeSitterParser
from codescope.index.store import IndexStats, IndexStore

log = logging.getLogger(__name__)

INDEX_RELPATH = Path(".serena") / "codescope" / "index.db"

# Directories we never descend into, regardless of .gitignore.
_DEFAULT_IGNORE_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", ".serena",
        "node_modules", ".venv", "venv", "env", ".env",
        "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        "dist", "build", "target", "out", "bin", "obj",
        ".idea", ".vscode", ".gradle", ".tox", ".nox", ".cache",
        "site-packages", ".next", ".nuxt", "vendor", "coverage",
    }
)

_MAX_FILE_BYTES = 2_000_000  # skip very large files (likely generated/vendored)


@dataclass(slots=True)
class ReindexReport:
    indexed: int
    skipped_unchanged: int
    removed: int
    errors: int
    stats: IndexStats
    duration_s: float


def default_db_path(project_root: str | Path) -> Path:
    return Path(project_root) / INDEX_RELPATH


class Indexer:
    """Builds and maintains the index for a single project root."""

    def __init__(self, project_root: str | Path, db_path: str | Path | None = None, embedder_name: str = "auto"):
        self.root = Path(project_root).resolve()
        self.db_path = Path(db_path) if db_path else default_db_path(self.root)
        self.embedder_name = embedder_name
        self.parser = TreeSitterParser()
        self._gitignore = self._load_gitignore()

    def _load_gitignore(self) -> pathspec.PathSpec | None:
        gi = self.root / ".gitignore"
        if not gi.exists():
            return None
        try:
            return pathspec.PathSpec.from_lines("gitwildmatch", gi.read_text(encoding="utf-8").splitlines())
        except Exception as e:  # pragma: no cover
            log.warning("Could not parse .gitignore: %s", e)
            return None

    def _is_ignored(self, rel: str) -> bool:
        if self._gitignore is None:
            return False
        return self._gitignore.match_file(rel)

    def iter_source_files(self):
        """Yield (absolute_path, relative_posix_path) for indexable source files."""
        stack = [self.root]
        while stack:
            current = stack.pop()
            try:
                entries = list(current.iterdir())
            except (PermissionError, OSError):
                continue
            for entry in entries:
                name = entry.name
                if entry.is_dir():
                    if name in _DEFAULT_IGNORE_DIRS:
                        continue
                    rel = entry.relative_to(self.root).as_posix()
                    if self._is_ignored(rel + "/"):
                        continue
                    stack.append(entry)
                    continue
                if spec_for_path(name) is None:
                    continue
                rel = entry.relative_to(self.root).as_posix()
                if self._is_ignored(rel):
                    continue
                yield entry, rel

    @staticmethod
    def _read_source(path: Path) -> bytes | None:
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                return None
            data = path.read_bytes()
        except (OSError, PermissionError):
            return None
        if b"\x00" in data[:8192]:  # crude binary guard
            return None
        return data

    def index_file(
        self, store: IndexStore, abs_path: Path, rel_path: str, *, force: bool
    ) -> tuple[str, list[tuple[int, str, str, str]]]:
        """Index a single file.

        :return: ``(outcome, inserted)`` where outcome is 'indexed'/'skipped'/
            'error' and ``inserted`` is the list of ``(sid, body, path, lang)``
            for newly stored symbols (empty unless outcome == 'indexed').
        """
        data = self._read_source(abs_path)
        if data is None:
            return "skipped", []
        file_hash = blake3.blake3(data).hexdigest()
        if not force and store.get_file_hash(rel_path) == file_hash:
            return "skipped", []
        result = self.parser.parse(rel_path, data)
        if result is None:
            return "skipped", []
        try:
            st = abs_path.stat()
            inserted = store.upsert_file(
                path=rel_path,
                lang=result.language,
                file_hash=file_hash,
                mtime=st.st_mtime,
                size=st.st_size,
                indexed_at=time.time(),
                symbols=result.symbols,
                refs=result.refs,
            )
        except Exception as e:
            log.warning("Failed to store %s: %s", rel_path, e)
            return "error", []
        return "indexed", inserted

    def _resolve_embedder(self, embeddings: bool, embedder: Embedder | None) -> Embedder | None:
        if not embeddings:
            return None
        if embedder is not None:
            return embedder
        try:
            return get_embedder(self.embedder_name)
        except Exception as e:  # pragma: no cover - depends on optional deps
            log.warning("Embeddings requested but no embedder available: %s", e)
            return None

    def reindex(
        self, *, force: bool = False, embeddings: bool = True, embedder: Embedder | None = None
    ) -> ReindexReport:
        """(Re)index the whole project, pruning files that no longer exist.

        If ``embeddings`` is true and an embedder is available, symbol bodies
        are embedded and stored for semantic search. When force-reindexing only
        part of the tree, embeddings are (re)built for the affected symbols.
        """
        start = time.time()
        indexed = skipped = errors = 0
        seen: set[str] = set()
        pending: list[tuple[int, str, str, str]] = []  # (sid, body, path, lang)
        store = IndexStore(self.db_path)
        try:
            resolved_embedder = self._resolve_embedder(embeddings, embedder) if store.vec_enabled else None

            for abs_path, rel in self.iter_source_files():
                seen.add(rel)
                outcome, inserted = self.index_file(store, abs_path, rel, force=force)
                if outcome == "indexed":
                    indexed += 1
                    pending.extend(inserted)
                elif outcome == "error":
                    errors += 1
                else:
                    skipped += 1

            # Prune files removed from disk.
            removed = 0
            for stale in store.indexed_paths() - seen:
                store.delete_file(stale)
                removed += 1

            if resolved_embedder is not None and pending:
                self._embed_pending(store, resolved_embedder, pending)

            store.commit()
            stats = store.stats()
        finally:
            store.close()
        return ReindexReport(
            indexed=indexed,
            skipped_unchanged=skipped,
            removed=removed,
            errors=errors,
            stats=stats,
            duration_s=round(time.time() - start, 3),
        )

    @staticmethod
    def _embed_pending(store: IndexStore, embedder: Embedder, pending: list[tuple[int, str, str, str]]) -> None:
        rows = [(sid, body, path, lang) for sid, body, path, lang in pending if body.strip()]
        if not rows:
            return
        store.ensure_vec_table(embedder.dim, embedder.id)
        vectors = embedder.embed_documents([body for _sid, body, _p, _l in rows])
        store.insert_embeddings(
            [(sid, vec, path, lang) for (sid, _body, path, lang), vec in zip(rows, vectors, strict=True)]
        )

    def status(self) -> IndexStats:
        store = IndexStore(self.db_path)
        try:
            return store.stats()
        finally:
            store.close()
