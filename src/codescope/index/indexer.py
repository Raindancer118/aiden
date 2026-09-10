"""Index orchestration: walk a project, detect changes, (re)index files.

Change detection uses a cheap mtime+size pre-filter followed by a blake3
content hash. Full git-scoped, watch-driven incremental reindexing is added in
milestone M5; this module already supports incremental updates via the stored
per-file hash.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import blake3
import pathspec

from codescope.index.embed import Embedder, HashingEmbedder, build_embed_text, get_embedder
from codescope.index.languages import spec_for_path
from codescope.index.parser import TreeSitterParser
from codescope.index.store import IndexStats, IndexStore

log = logging.getLogger(__name__)

INDEX_RELPATH = Path(".serena") / "codescope" / "index.db"

# Directories we never descend into, regardless of .gitignore.
_DEFAULT_IGNORE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".serena",
        "node_modules",
        ".venv",
        "venv",
        "env",
        ".env",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        "target",
        "out",
        "bin",
        "obj",
        ".idea",
        ".vscode",
        ".gradle",
        ".tox",
        ".nox",
        ".cache",
        "site-packages",
        ".next",
        ".nuxt",
        "vendor",
        "coverage",
    }
)

_MAX_FILE_BYTES = 2_000_000  # skip very large files (likely generated/vendored)

#: Files per write transaction during a full reindex. Bounded so a long run
#: keeps releasing the SQLite writer lock instead of blocking searches.
_COMMIT_EVERY_FILES = 200


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
        if any(part in _DEFAULT_IGNORE_DIRS for part in PurePosixPath(rel).parts):
            return True
        if self._gitignore is None:
            return False
        return self._gitignore.match_file(rel)

    def iter_source_files(self) -> Iterator[tuple[Path, str]]:
        """Yield (absolute_path, relative_posix_path) for indexable source files."""
        stack = [self.root]
        while stack:
            current = stack.pop()
            try:
                entries = list(current.iterdir())
            except (PermissionError, OSError):
                continue
            for entry in entries:
                if entry.is_symlink():
                    continue
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

    def index_file(self, store: IndexStore, abs_path: Path, rel_path: str, *, force: bool) -> tuple[str, list[tuple[int, str, str, str]]]:
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

    def _resolve_embedder(self, embeddings: bool, embedder: Embedder | None) -> tuple[Embedder | None, bool]:
        """Resolve the embedder to use.

        :return: ``(embedder, explicit)``. ``explicit`` is True when the
            caller named the backend, and False when it came from ``auto``
            resolution -- which may silently have degraded to the hashing
            fallback and must therefore never trigger a destructive rebuild
            of a real semantic index (see :meth:`_update_embeddings`).
        """
        if not embeddings:
            return None, False
        if embedder is not None:
            return embedder, True
        try:
            return get_embedder(self.embedder_name), self.embedder_name.lower() not in ("", "auto")
        except Exception as e:  # pragma: no cover - depends on optional deps
            log.warning("Embeddings requested but no embedder available: %s", e)
            return None, False

    def reindex(self, *, force: bool = False, embeddings: bool = True, embedder: Embedder | None = None) -> ReindexReport:
        """(Re)index the whole project, pruning files that no longer exist.

        If ``embeddings`` is true and an embedder is available, symbol bodies
        are embedded and stored for semantic search. When force-reindexing only
        part of the tree, embeddings are (re)built for the affected symbols.
        """
        start = time.time()
        indexed = skipped = errors = 0
        seen: set[str] = set()
        store = IndexStore(self.db_path)
        try:
            self._gitignore = self._load_gitignore()

            batch: list[tuple[Path, str]] = []
            for abs_path, rel in self.iter_source_files():
                seen.add(rel)
                batch.append((abs_path, rel))
                if len(batch) < _COMMIT_EVERY_FILES:
                    continue
                counts = self._index_batch(store, batch, force=force)
                indexed, errors, skipped = indexed + counts[0], errors + counts[1], skipped + counts[2]
                batch.clear()
            if batch:
                counts = self._index_batch(store, batch, force=force)
                indexed, errors, skipped = indexed + counts[0], errors + counts[1], skipped + counts[2]

            # Prune files removed from disk.
            removed = 0
            with store.transaction():
                for stale in store.indexed_paths() - seen:
                    store.delete_file(stale)
                    removed += 1

            # persist the lexical index before model loading or vectorization,
            # both of which may be slow or depend on optional external assets.
            store.commit()

            if store.vec_enabled:
                resolved_embedder, explicit = self._resolve_embedder(embeddings, embedder)
                if resolved_embedder is not None:
                    self._update_embeddings(store, resolved_embedder, explicit=explicit)

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

    def _index_batch(self, store: IndexStore, batch: list[tuple[Path, str]], *, force: bool) -> tuple[int, int, int]:
        """Index a group of files in one transaction.

        Grouping matters: each file's savepoint would otherwise commit on its
        own, so a large reindex paid one durable write per file. The group is
        bounded so a long reindex still releases the writer lock regularly.
        """
        indexed = errors = skipped = 0
        with store.transaction():
            for abs_path, rel in batch:
                outcome, _inserted = self.index_file(store, abs_path, rel, force=force)
                if outcome == "indexed":
                    indexed += 1
                elif outcome == "error":
                    errors += 1
                else:
                    skipped += 1
        return indexed, errors, skipped

    def reindex_paths(
        self,
        rel_paths: list[str],
        *,
        force: bool = True,
        embeddings: bool = True,
        embedder: Embedder | None = None,
    ) -> ReindexReport:
        """Incrementally (re)index a specific set of files.

        Missing files are pruned. Used by the git-scoped sync and the file
        watcher. ``force`` defaults to True since callers already know the
        files changed, but the per-file hash still prevents redundant writes
        when ``force`` is False.
        """
        start = time.time()
        indexed = skipped = errors = removed = 0
        store = IndexStore(self.db_path)
        try:
            self._gitignore = self._load_gitignore()
            indexed_paths = store.indexed_paths()
            for raw_rel in dict.fromkeys(rel_paths):  # de-dup, preserve order
                rel_path = PurePosixPath(raw_rel)
                windows_path = PureWindowsPath(raw_rel)
                if (
                    not raw_rel
                    or "\\" in raw_rel
                    or "\x00" in raw_rel
                    or rel_path.is_absolute()
                    or rel_path == PurePosixPath(".")
                    or ".." in rel_path.parts
                    or windows_path.anchor
                    or ".." in windows_path.parts
                ):
                    log.warning("Refusing to index path outside the project: %s", raw_rel)
                    errors += 1
                    continue
                rel = rel_path.as_posix()
                abs_path = self.root.joinpath(*rel_path.parts)
                try:
                    resolved_path = abs_path.resolve()
                    resolved_path.relative_to(self.root)
                except (OSError, RuntimeError, ValueError):
                    log.warning("Refusing to index path outside the project: %s", raw_rel)
                    if rel in indexed_paths:
                        store.delete_file(rel)
                        removed += 1
                    errors += 1
                    continue
                if not abs_path.is_file() or abs_path.is_symlink() or self._is_ignored(rel):
                    if rel in indexed_paths:
                        store.delete_file(rel)
                        removed += 1
                    continue
                if spec_for_path(rel) is None:
                    continue
                outcome, _inserted = self.index_file(store, abs_path, rel, force=force)
                if outcome == "indexed":
                    indexed += 1
                elif outcome == "error":
                    errors += 1
                else:
                    skipped += 1

            store.commit()

            if store.vec_enabled:
                resolved_embedder, explicit = self._resolve_embedder(embeddings, embedder)
                if resolved_embedder is not None:
                    self._update_embeddings(store, resolved_embedder, explicit=explicit)

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

    def _update_embeddings(self, store: IndexStore, embedder: Embedder, *, explicit: bool = True) -> None:
        """Backfill vectors while preserving the last complete vector index on failure.

        An embedder change requires re-embedding every symbol, since the
        existing vectors are no longer comparable to newly computed ones. That
        rebuild is staged (see ``IndexStore.rebuild_vec_table``): embeddings
        are computed first, and the previous vector table is only replaced
        once the new one is fully built, so a failing embedder backend leaves
        the last working index untouched instead of a `DROP TABLE` (which
        auto-commits and can't be undone by a transaction rollback) destroying
        it up front.

        The backfill path (no embedder change) is the common one and writes
        incrementally: vectors are persisted batch by batch, so peak memory
        stays flat and an interrupted run keeps everything already written.
        """
        if store.embedder_changed(embedder.dim, embedder.id):
            if not explicit and isinstance(embedder, HashingEmbedder) and not (store.get_meta("embedder_id") or "").startswith("hashing-"):
                # ``auto`` degrades to the hashing fallback when the real
                # backend fails to load. Treating that as an intentional
                # embedder change would silently replace a semantic index
                # with lexical hash vectors, so leave the index alone.
                log.warning(
                    "Semantic backend unavailable; keeping existing %s vectors instead of rebuilding with the hashing fallback.",
                    store.get_meta("embedder_id"),
                )
                return
            rows = [row for row in store.all_symbol_rows_for_embedding() if row[6].strip()]
            embedded = self._embed_rows(embedder, rows)
            store.rebuild_vec_table(embedder.dim, embedder.id, embedded)
            store.commit()
            return

        store.ensure_vec_table(embedder.dim, embedder.id)
        pending = store.symbols_without_embeddings()
        if pending:
            self._embed_pending(store, embedder, pending)
        store.commit()

    @staticmethod
    def _embed_texts(rows: list[tuple[int, str, str, str, str, str, str]]) -> list[str]:
        """Compose the context-enriched text embedded for each symbol row."""
        return [build_embed_text(name, kind, signature, body) for _sid, _path, _lang, name, kind, signature, body in rows]

    @classmethod
    def _embed_rows(
        cls, embedder: Embedder, rows: list[tuple[int, str, str, str, str, str, str]]
    ) -> list[tuple[int, list[float], str, str]]:
        """Embed symbol rows, buffering the result.

        Used only for the staged full rebuild, which must hold every vector
        before it may replace the previous table. Vectors are ~3 KB each, so
        buffering them is cheap; it is the *model activations* that were the
        memory problem, and ``embed_batched`` bounds those.
        """
        out: list[tuple[int, list[float], str, str]] = []
        for batch in embedder.embed_batched(cls._embed_texts(rows)):
            for index, vector in batch:
                out.append((rows[index][0], vector, rows[index][1], rows[index][2]))
        return out

    @classmethod
    def _embed_pending(cls, store: IndexStore, embedder: Embedder, pending: list[tuple[int, str, str, str, str, str, str]]) -> None:
        """Embed and persist missing vectors batch by batch (interruptible)."""
        rows = [row for row in pending if row[6].strip()]
        if not rows:
            return
        done = 0
        for batch in embedder.embed_batched(cls._embed_texts(rows)):
            store.insert_embeddings([(rows[i][0], vec, rows[i][1], rows[i][2]) for i, vec in batch])
            store.commit()
            done += len(batch)
            if done % 512 < len(batch):
                log.info("Embedded %d/%d symbols", done, len(rows))

    def status(self) -> IndexStats:
        store = IndexStore(self.db_path)
        try:
            return store.stats()
        finally:
            store.close()
