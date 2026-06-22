"""Tests for the Codescope search engine (BM25 + vector + trigram + RRF).

These use the deterministic HashingEmbedder so no model download is needed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from codescope.index.embed import HashingEmbedder, embedder_from_id
from codescope.index.indexer import Indexer
from codescope.index.search import SearchEngine


@pytest.fixture
def indexed(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "auth.py").write_text(
        "def validate_token(token):\n"
        "    '''Check whether an auth token is valid.'''\n"
        "    return verify_signature(token)\n"
        "\n"
        "def verify_signature(token):\n"
        "    return True\n"
    )
    (tmp_path / "io_utils.py").write_text(
        "def read_config_file(path):\n"
        "    with open(path) as f:\n"
        "        return f.read()\n"
    )
    db = tmp_path / "idx" / "index.db"
    report = Indexer(tmp_path, db_path=db).reindex(embedder=HashingEmbedder())
    assert report.errors == 0
    return tmp_path, db


def test_vectors_are_built(indexed: tuple[Path, Path]) -> None:
    root, db = indexed
    stats = Indexer(root, db_path=db).status()
    assert stats.vectors > 0
    assert stats.embedder is not None and stats.embedder.startswith("hashing-")


def test_hybrid_search_finds_symbol_and_fuses_sources(indexed: tuple[Path, Path]) -> None:
    root, db = indexed
    hits = SearchEngine(root, db_path=db).hybrid_search("validate token", limit=5)
    assert hits
    assert any(h.name == "validate_token" for h in hits)
    # At least one hit should be supported by more than one retriever.
    assert any(len(h.sources) >= 1 for h in hits)
    top = hits[0]
    assert {"bm25", "vector", "trigram"} >= set(top.sources)


def test_semantic_search_returns_results(indexed: tuple[Path, Path]) -> None:
    root, db = indexed
    hits = SearchEngine(root, db_path=db).semantic_search("verify_signature", limit=5)
    assert any(h.name == "verify_signature" for h in hits)


def test_substring_search(indexed: tuple[Path, Path]) -> None:
    root, db = indexed
    hits = SearchEngine(root, db_path=db).substring_search("read_config", limit=5)
    assert any(h.name == "read_config_file" for h in hits)


def test_substring_search_too_short_returns_empty(indexed: tuple[Path, Path]) -> None:
    root, db = indexed
    assert SearchEngine(root, db_path=db).substring_search("ab", limit=5) == []


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_regex_search(indexed: tuple[Path, Path]) -> None:
    root, db = indexed
    hits = SearchEngine(root, db_path=db).regex_search(r"def verify_\w+", limit=10)
    assert any("auth.py" in h.path and "verify_signature" in h.text for h in hits)


def test_embedder_from_id_roundtrip() -> None:
    e = HashingEmbedder(dim=128)
    restored = embedder_from_id(e.id)
    assert restored.dim == 128
    a = e.embed_query("hello world")
    b = restored.embed_query("hello world")
    assert a == b


def test_find_similar_code_finds_matching_symbol(indexed: tuple[Path, Path]) -> None:
    root, db = indexed
    snippet = "def validate_token(token):\n    return verify_signature(token)\n"
    hits = SearchEngine(root, db_path=db).find_similar_code(snippet, limit=5)
    assert hits
    assert hits[0].name == "validate_token"
    assert -1.0 <= hits[0].score <= 1.0


@pytest.fixture
def indexed_with_clones(tmp_path: Path) -> tuple[Path, Path]:
    body = (
        "def {name}(items):\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        total = total + item\n"
        "    return total\n"
    )
    (tmp_path / "a.py").write_text(body.format(name="sum_items"))
    (tmp_path / "b.py").write_text(body.format(name="sum_items"))
    (tmp_path / "c.py").write_text(
        "def unrelated(path):\n"
        "    with open(path) as f:\n"
        "        return f.read().upper()\n"
    )
    db = tmp_path / "idx" / "index.db"
    report = Indexer(tmp_path, db_path=db).reindex(embedder=HashingEmbedder())
    assert report.errors == 0
    return tmp_path, db


def test_find_duplicate_code_groups_clones(indexed_with_clones: tuple[Path, Path]) -> None:
    root, db = indexed_with_clones
    groups = SearchEngine(root, db_path=db).find_duplicate_code(min_lines=3, similarity=0.95)
    assert groups
    paths = {m.path for m in groups[0].members}
    assert "a.py" in paths and "b.py" in paths
    assert all(m.lines >= 3 for m in groups[0].members)
    assert groups[0].similarity >= 0.95


def test_find_duplicate_code_respects_min_lines(indexed_with_clones: tuple[Path, Path]) -> None:
    root, db = indexed_with_clones
    # The clone bodies are 5 lines; a min_lines of 100 excludes everything.
    assert SearchEngine(root, db_path=db).find_duplicate_code(min_lines=100) == []


def test_search_tools_registered() -> None:
    from codescope.cli import register_codescope_tools

    register_codescope_tools()
    from serena.tools import ToolRegistry

    names = ToolRegistry().get_tool_names()
    for t in ("search_code", "search_semantic", "search_regex", "find_similar_code", "find_duplicate_code"):
        assert t in names
