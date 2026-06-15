"""Tests for the Codescope index engine (parser + store + indexer)."""

from __future__ import annotations

from pathlib import Path

import pytest

from codescope.index.indexer import Indexer
from codescope.index.parser import TreeSitterParser
from codescope.index.store import IndexStore


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "app.py").write_text(
        "class Greeter:\n"
        "    def greet(self, name):\n"
        "        return shout(name)\n"
        "\n"
        "def shout(text):\n"
        "    return text.upper()\n"
    )
    (tmp_path / "util.go").write_text(
        "package util\n\nfunc Add(a int, b int) int {\n\treturn a + b\n}\n\ntype Pair struct{}\n"
    )
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
    report = idx.reindex()
    assert report.errors == 0
    assert report.indexed == 3  # app.py, util.go, lib.rs (ignored/ excluded)
    assert {"python", "go", "rust"} <= set(report.stats.languages)
    assert report.stats.symbols >= 6


def test_gitignore_is_respected(project: Path) -> None:
    idx = Indexer(project, db_path=_db(project))
    idx.reindex()
    store = IndexStore(_db(project))
    try:
        rows = store.conn.execute("SELECT COUNT(*) FROM symbols WHERE name='should_not_be_indexed'").fetchone()
        assert rows[0] == 0
    finally:
        store.close()


def test_incremental_skips_unchanged(project: Path) -> None:
    idx = Indexer(project, db_path=_db(project))
    idx.reindex()
    second = idx.reindex()
    assert second.indexed == 0
    assert second.skipped_unchanged == 3


def test_reindex_prunes_deleted_files(project: Path) -> None:
    idx = Indexer(project, db_path=_db(project))
    idx.reindex()
    (project / "lib.rs").unlink()
    report = idx.reindex()
    assert report.removed == 1
    assert "rust" not in report.stats.languages


def test_bm25_search_returns_relevant_symbol(project: Path) -> None:
    idx = Indexer(project, db_path=_db(project))
    idx.reindex()
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
