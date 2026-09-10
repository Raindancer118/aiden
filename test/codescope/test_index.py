"""Tests for the Codescope index engine (parser + store + indexer)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from codescope.index.embed import HashingEmbedder
from codescope.index.indexer import Indexer
from codescope.index.parser import SymbolDef, TreeSitterParser
from codescope.index.store import IndexStore


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "app.py").write_text(
        "class Greeter:\n    def greet(self, name):\n        return shout(name)\n\ndef shout(text):\n    return text.upper()\n"
    )
    (tmp_path / "util.go").write_text("package util\n\nfunc Add(a int, b int) int {\n\treturn a + b\n}\n\ntype Pair struct{}\n")
    (tmp_path / "lib.rs").write_text("fn main() {\n    helper();\n}\n\nstruct Config { n: i32 }\n")
    # Ignored content
    (tmp_path / ".gitignore").write_text("ignored/\n")
    ignored = tmp_path / "ignored"
    ignored.mkdir()
    (ignored / "secret.py").write_text("def should_not_be_indexed():\n    pass\n")
    return tmp_path


def _db(tmp_path: Path) -> Path:
    return tmp_path / "idx" / "index.db"


def test_parser_extracts_python_symbols() -> None:
    parser = TreeSitterParser()
    src = b"class Foo:\n    def bar(self):\n        return helper()\n\ndef helper():\n    pass\n"
    result = parser.parse("x.py", src)
    assert result is not None
    names = {(s.name, s.kind) for s in result.symbols}
    assert ("Foo", "class") in names
    assert ("bar", "method") in names or ("bar", "function") in names
    assert ("helper", "function") in names
    # References (used by the graph layer) include the call to helper().
    assert any(r.name == "helper" for r in result.refs)


def test_parser_extracts_typescript_symbols() -> None:
    parser = TreeSitterParser()
    src = (
        b"export function add(a: number): number { return mul(a); }\n"
        b"export const mul = (a: number) => a * 2;\n"
        b"export class Service { run(): void { helper(); } }\n"
        b"interface Repo { find(id: string): void; }\n"
        b"type ID = string;\n"
        b"enum Color { Red, Green }\n"
    )
    result = parser.parse("svc.ts", src)
    assert result is not None
    kinds = {(s.name, s.kind) for s in result.symbols}
    assert ("add", "function") in kinds
    assert ("Service", "class") in kinds
    assert ("Repo", "interface") in kinds
    assert ("ID", "type") in kinds
    assert ("Color", "enum") in kinds
    assert any(r.name == "helper" for r in result.refs)


def test_parser_unsupported_extension_returns_none() -> None:
    assert TreeSitterParser().parse("notes.unknownext", b"hello") is None


def test_reindex_extracts_multiple_languages(project: Path) -> None:
    idx = Indexer(project, db_path=_db(project))
    report = idx.reindex(embeddings=False)
    assert report.errors == 0
    assert report.indexed == 3  # app.py, util.go, lib.rs (ignored/ excluded)
    assert {"python", "go", "rust"} <= set(report.stats.languages)
    assert report.stats.symbols >= 6


def test_gitignore_is_respected(project: Path) -> None:
    idx = Indexer(project, db_path=_db(project))
    idx.reindex(embeddings=False)
    store = IndexStore(_db(project))
    try:
        rows = store.conn.execute("SELECT COUNT(*) FROM symbols WHERE name='should_not_be_indexed'").fetchone()
        assert rows[0] == 0
    finally:
        store.close()


def test_index_walk_does_not_follow_directory_symlinks(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    (project / "local.py").write_text("def local():\n    pass\n")
    (external / "secret.py").write_text("def external_secret():\n    pass\n")
    try:
        (project / "linked").symlink_to(external, target_is_directory=True)
    except OSError as e:  # pragma: no cover - platform policy
        pytest.skip(f"directory symlinks unavailable: {e}")

    paths = {rel for _absolute, rel in Indexer(project).iter_source_files()}

    assert paths == {"local.py"}


def test_reindex_paths_rejects_paths_through_external_symlink(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    (external / "secret.py").write_text("def external_secret():\n    pass\n")
    try:
        (project / "linked").symlink_to(external, target_is_directory=True)
    except OSError as e:  # pragma: no cover - platform policy
        pytest.skip(f"directory symlinks unavailable: {e}")

    db = _db(project)
    report = Indexer(project, db_path=db).reindex_paths(["linked/secret.py"], embeddings=False)

    assert report.indexed == 0
    assert report.errors == 1
    with IndexStore(db) as store:
        assert store.indexed_paths() == set()


@pytest.mark.parametrize("unsafe_path", [r"..\outside.py", r"C:\outside.py", r"\outside.py"])
def test_reindex_paths_rejects_windows_path_escapes(tmp_path: Path, unsafe_path: str) -> None:
    report = Indexer(tmp_path, db_path=_db(tmp_path)).reindex_paths([unsafe_path], embeddings=False)

    assert report.indexed == 0
    assert report.errors == 1


def test_failed_file_upsert_keeps_previous_index_rows(tmp_path: Path) -> None:
    store = IndexStore(_db(tmp_path))
    stable = SymbolDef(
        name="stable",
        kind="function",
        start_line=1,
        start_col=1,
        end_line=2,
        end_col=9,
        signature="def stable():",
        body="def stable():\n    pass",
    )
    broken = SymbolDef(
        name="broken",
        kind="function",
        start_line=1,
        start_col=1,
        end_line=2,
        end_col=9,
        signature="def broken():",
        body="def broken():\n    pass",
    )
    try:
        store.upsert_file("app.py", "python", "old-hash", 1.0, 20, 1.0, [stable], [])
        store.commit()
        store.conn.execute(
            "CREATE TRIGGER reject_broken BEFORE INSERT ON symbols WHEN NEW.name='broken' BEGIN SELECT RAISE(ABORT, 'broken symbol'); END"
        )

        with pytest.raises(sqlite3.IntegrityError, match="broken symbol"):
            store.upsert_file("app.py", "python", "new-hash", 2.0, 20, 2.0, [broken], [])
        store.commit()

        assert store.get_file_hash("app.py") == "old-hash"
        assert store.conn.execute("SELECT name FROM symbols WHERE path='app.py'").fetchall() == [("stable",)]
        assert store.conn.execute("SELECT name FROM symbols_fts WHERE path='app.py'").fetchall() == [("stable",)]
    finally:
        store.close()


def test_incremental_skips_unchanged(project: Path) -> None:
    idx = Indexer(project, db_path=_db(project))
    idx.reindex(embeddings=False)
    second = idx.reindex(embeddings=False)
    assert second.indexed == 0
    assert second.skipped_unchanged == 3


def test_later_reindex_backfills_vectors_for_unchanged_symbols(project: Path) -> None:
    idx = Indexer(project, db_path=_db(project))
    lexical = idx.reindex(embeddings=False)
    assert lexical.stats.vectors == 0

    semantic = idx.reindex(embedder=HashingEmbedder())

    assert semantic.indexed == 0
    assert semantic.stats.vectors == semantic.stats.symbols


def test_embedder_change_rebuilds_all_vectors_even_with_same_dimension(project: Path) -> None:
    idx = Indexer(project, db_path=_db(project))
    initial = HashingEmbedder(dim=32)
    first = idx.reindex(embedder=initial)
    assert first.stats.vectors == first.stats.symbols

    replacement = HashingEmbedder(dim=32)
    replacement.id = "replacement-32"
    second = idx.reindex(embedder=replacement)

    assert second.indexed == 0
    assert second.stats.embedder == "replacement-32"
    assert second.stats.vectors == second.stats.symbols


def test_failed_embedder_change_keeps_previous_vectors(project: Path) -> None:
    idx = Indexer(project, db_path=_db(project))
    initial = HashingEmbedder(dim=32)
    first = idx.reindex(embedder=initial)

    class FailingEmbedder(HashingEmbedder):
        def __init__(self) -> None:
            super().__init__(dim=32)
            self.id = "failing-32"

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding backend unavailable")

    with pytest.raises(RuntimeError, match="backend unavailable"):
        idx.reindex(embedder=FailingEmbedder())

    with IndexStore(_db(project)) as store:
        stats = store.stats()
    assert stats.embedder == initial.id
    assert stats.vectors == first.stats.vectors == first.stats.symbols


def test_reindex_prunes_deleted_files(project: Path) -> None:
    idx = Indexer(project, db_path=_db(project))
    idx.reindex(embeddings=False)
    (project / "lib.rs").unlink()
    report = idx.reindex(embeddings=False)
    assert report.removed == 1
    assert "rust" not in report.stats.languages


def test_bm25_search_returns_relevant_symbol(project: Path) -> None:
    idx = Indexer(project, db_path=_db(project))
    idx.reindex(embeddings=False)
    store = IndexStore(_db(project))
    try:
        rows = store.conn.execute(
            "SELECT name FROM symbols_fts WHERE symbols_fts MATCH 'shout' ORDER BY bm25(symbols_fts) LIMIT 3"
        ).fetchall()
        assert any(r[0] == "shout" for r in rows)
    finally:
        store.close()


def test_index_tools_registered() -> None:
    from codescope.cli import register_codescope_tools

    register_codescope_tools()
    from serena.tools import ToolRegistry

    names = ToolRegistry().get_tool_names()
    assert "reindex" in names
    assert "index_status" in names


def test_failed_vector_swap_rolls_back_to_the_previous_vectors(project: Path) -> None:
    """A crash mid-swap must not leave the project without a vector index."""
    db = _db(project)
    Indexer(project, db_path=db).reindex(embedder=HashingEmbedder(dim=16))

    with IndexStore(db) as store:
        before = store.stats().vectors
        assert before > 0

        good = [(sid, [0.0] * 16, "app.py", "python") for sid, *_ in [(r[0],) for r in store.all_symbol_rows_for_embedding()]]
        broken = [*good, (good[0][0], [0.0] * 16, "app.py", "python")]  # duplicate primary key
        with pytest.raises(sqlite3.Error):
            store.rebuild_vec_table(16, "hashing-16", broken)

    with IndexStore(db) as store:
        assert store.stats().vectors == before
        assert store.get_meta("embedder_id") == "hashing-16"


def test_health_reports_a_missing_index_instead_of_zeroes(tmp_path: Path) -> None:
    health = Indexer(tmp_path, db_path=_db(tmp_path)).health()
    assert health["indexed"] is False
    assert any("reindex" in a for a in health["advice"])


def test_health_flags_the_lexical_fallback_as_not_semantic(project: Path) -> None:
    """A hash-vector index answers every query - lexically. Say so."""
    db = _db(project)
    Indexer(project, db_path=db).reindex(embedder=HashingEmbedder())

    health = Indexer(project, db_path=db).health()
    assert health["indexed"] is True
    assert health["vectors"] > 0
    assert health["symbols_missing_vectors"] == 0
    assert health["semantic_search_ready"] is False
    assert health["clone_detection_ready"] is False
    assert any("hashing fallback" in a for a in health["advice"])


def test_health_reports_an_incomplete_vector_backfill(project: Path) -> None:
    db = _db(project)
    Indexer(project, db_path=db).reindex(embeddings=False)

    health = Indexer(project, db_path=db).health()
    assert health["vectors"] == 0
    assert health["symbols_missing_vectors"] == health["symbols"] > 0
    assert any("No vectors indexed" in a for a in health["advice"])
