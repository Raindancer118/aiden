"""SQLite persistence for the Codescope index.

A single database file holds:
  meta          - schema version + index configuration
  files         - per-file change-detection metadata (hash/mtime/size)
  symbols       - extracted symbol definitions (direct, indexed lookups)
  refs          - extracted references (backs the call/usage graph, M4)
  symbols_fts   - FTS5 BM25 index over (name, path, body)
  symbols_trgm  - FTS5 trigram index over body (substring/regex acceleration)

The vector table (sqlite-vec) is added in milestone M3.
``symbols_fts`` / ``symbols_trgm`` rows share their rowid with ``symbols.id``
so a file's rows can be removed precisely on reindex.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

from codescope.index.parser import RefHit, SymbolDef

log = logging.getLogger(__name__)


def _encode_vector(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _decode_vector(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS files (
    path        TEXT PRIMARY KEY,
    lang        TEXT,
    hash        TEXT NOT NULL,
    mtime       REAL,
    size        INTEGER,
    indexed_at  REAL
);

CREATE TABLE IF NOT EXISTS symbols (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    start_line  INTEGER NOT NULL,
    start_col   INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    end_col     INTEGER NOT NULL,
    signature   TEXT
);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_path ON symbols(path);
CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind);

CREATE TABLE IF NOT EXISTS refs (
    id    INTEGER PRIMARY KEY,
    path  TEXT NOT NULL,
    name  TEXT NOT NULL,
    kind  TEXT NOT NULL,
    line  INTEGER NOT NULL,
    col   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_refs_name ON refs(name);
CREATE INDEX IF NOT EXISTS idx_refs_path ON refs(path);

CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    name, path, body, kind UNINDEXED,
    tokenize = 'unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS symbols_trgm USING fts5(
    body,
    tokenize = 'trigram'
);
"""


@dataclass(slots=True)
class IndexStats:
    files: int
    symbols: int
    refs: int
    languages: dict[str, int]
    vectors: int
    embedder: str | None


class IndexStore:
    """Thin wrapper around the SQLite index database."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.vec_enabled = self._load_sqlite_vec()
        self._init_schema()

    def _load_sqlite_vec(self) -> bool:
        try:
            import sqlite_vec

            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            return True
        except Exception as e:  # pragma: no cover - platform dependent
            log.info("sqlite-vec not available; vector search disabled: %s", e)
            return False

    def _init_schema(self) -> None:
        self.conn.executescript(_SCHEMA)
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    # -- meta -------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    # -- vector table -----------------------------------------------------

    def ensure_vec_table(self, dim: int, embedder_id: str) -> bool:
        """Create the sqlite-vec table for ``dim`` if needed. Returns success."""
        if not self.vec_enabled:
            return False
        existing = self.get_meta("embedder_dim")
        if existing is not None and int(existing) != dim:
            # Embedder changed: drop stale vectors so dimensions stay consistent.
            self.conn.execute("DROP TABLE IF EXISTS chunks_vec")
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
            f"sid integer primary key, embedding float[{dim}], +path text, +lang text)"
        )
        self.set_meta("embedder_dim", str(dim))
        self.set_meta("embedder_id", embedder_id)
        return True

    def has_vectors(self) -> bool:
        if not self.vec_enabled:
            return False
        row = self.conn.execute("SELECT name FROM sqlite_master WHERE name='chunks_vec'").fetchone()
        if row is None:
            return False
        return self.conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] > 0

    def insert_embeddings(self, rows: list[tuple[int, list[float], str, str]]) -> None:
        if not self.vec_enabled or not rows:
            return
        self.conn.executemany(
            "INSERT INTO chunks_vec(sid, embedding, path, lang) VALUES(?,?,?,?)",
            [(sid, _encode_vector(vec), path, lang) for sid, vec, path, lang in rows],
        )

    def vector_search(self, query_vec: list[float], k: int) -> list[tuple[int, float]]:
        """Return [(symbol_id, distance)] for the k nearest symbols."""
        if not self.has_vectors():
            return []
        return self.conn.execute(
            "SELECT sid, distance FROM chunks_vec WHERE embedding MATCH ? AND k=? ORDER BY distance",
            (_encode_vector(query_vec), k),
        ).fetchall()

    def get_embedding(self, sid: int) -> list[float] | None:
        """Return the stored embedding for a symbol id, or None."""
        if not self.has_vectors():
            return None
        row = self.conn.execute("SELECT embedding FROM chunks_vec WHERE sid=?", (sid,)).fetchone()
        return _decode_vector(row[0]) if row else None

    def symbols_with_min_lines(self, min_lines: int) -> list[tuple[int, str, str, str, int, int, str]]:
        """Symbols whose body spans at least ``min_lines`` lines.

        :return: ``(id, name, kind, path, start_line, end_line, signature)`` rows.
        """
        return self.conn.execute(
            "SELECT id, name, kind, path, start_line, end_line, signature FROM symbols "
            "WHERE (end_line - start_line + 1) >= ?",
            (min_lines,),
        ).fetchall()

    def _vec_table_exists(self) -> bool:
        if not self.vec_enabled:
            return False
        return self.conn.execute("SELECT name FROM sqlite_master WHERE name='chunks_vec'").fetchone() is not None

    # -- change detection -------------------------------------------------

    def get_file_hash(self, path: str) -> str | None:
        row = self.conn.execute("SELECT hash FROM files WHERE path=?", (path,)).fetchone()
        return row[0] if row else None

    def indexed_paths(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT path FROM files")}

    # -- mutation ---------------------------------------------------------

    def delete_file(self, path: str) -> None:
        cur = self.conn.cursor()
        ids = [r[0] for r in cur.execute("SELECT id FROM symbols WHERE path=?", (path,))]
        if ids:
            placeholders = ",".join("?" * len(ids))
            cur.execute(f"DELETE FROM symbols_fts WHERE rowid IN ({placeholders})", ids)
            cur.execute(f"DELETE FROM symbols_trgm WHERE rowid IN ({placeholders})", ids)
            if self._vec_table_exists():
                cur.execute(f"DELETE FROM chunks_vec WHERE sid IN ({placeholders})", ids)
        cur.execute("DELETE FROM symbols WHERE path=?", (path,))
        cur.execute("DELETE FROM refs WHERE path=?", (path,))
        cur.execute("DELETE FROM files WHERE path=?", (path,))

    def upsert_file(
        self,
        path: str,
        lang: str,
        file_hash: str,
        mtime: float,
        size: int,
        indexed_at: float,
        symbols: list[SymbolDef],
        refs: list[RefHit],
    ) -> list[tuple[int, str, str, str]]:
        """Replace all rows for ``path`` with freshly parsed data (single txn).

        :return: list of ``(symbol_id, body, path, lang)`` for the inserted
            symbols, so the caller can batch-embed bodies afterwards.
        """
        cur = self.conn.cursor()
        self.delete_file(path)
        cur.execute(
            "INSERT INTO files(path, lang, hash, mtime, size, indexed_at) VALUES(?,?,?,?,?,?)",
            (path, lang, file_hash, mtime, size, indexed_at),
        )
        inserted: list[tuple[int, str, str, str]] = []
        for s in symbols:
            cur.execute(
                "INSERT INTO symbols(path, name, kind, start_line, start_col, end_line, end_col, signature) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (path, s.name, s.kind, s.start_line, s.start_col, s.end_line, s.end_col, s.signature),
            )
            sid = cur.lastrowid
            assert sid is not None
            cur.execute(
                "INSERT INTO symbols_fts(rowid, name, path, body, kind) VALUES(?,?,?,?,?)",
                (sid, s.name, path, s.body, s.kind),
            )
            cur.execute("INSERT INTO symbols_trgm(rowid, body) VALUES(?,?)", (sid, s.body))
            inserted.append((sid, s.body, path, lang))
        if refs:
            cur.executemany(
                "INSERT INTO refs(path, name, kind, line, col) VALUES(?,?,?,?,?)",
                [(path, r.name, r.kind, r.line, r.col) for r in refs],
            )
        return inserted

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # -- stats ------------------------------------------------------------

    def stats(self) -> IndexStats:
        c = self.conn
        n_files = c.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        n_syms = c.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        n_refs = c.execute("SELECT COUNT(*) FROM refs").fetchone()[0]
        langs = dict(c.execute("SELECT lang, COUNT(*) FROM files GROUP BY lang ORDER BY 2 DESC").fetchall())
        n_vec = 0
        if self._vec_table_exists():
            n_vec = c.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
        return IndexStats(
            files=n_files,
            symbols=n_syms,
            refs=n_refs,
            languages=langs,
            vectors=n_vec,
            embedder=self.get_meta("embedder_id"),
        )

    def __enter__(self) -> "IndexStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
