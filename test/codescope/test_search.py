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
    (tmp_path / "io_utils.py").write_text("def read_config_file(path):\n    with open(path) as f:\n        return f.read()\n")
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


def test_blank_searches_return_no_arbitrary_vector_matches(indexed: tuple[Path, Path]) -> None:
    root, db = indexed
    engine = SearchEngine(root, db_path=db)
    assert engine.hybrid_search("   ") == []
    assert engine.semantic_search("   ") == []
    assert engine.substring_search("   ") == []
    assert engine.find_similar_code("   ") == []


def test_substring_search_escapes_embedded_quotes(tmp_path: Path) -> None:
    (tmp_path / "greeting.py").write_text("def greeting():\n    return 'say \"hello\" now'\n")
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embeddings=False)

    hits = SearchEngine(tmp_path, db_path=db).substring_search('hello" now')
    assert [hit.name for hit in hits] == ["greeting"]


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_regex_search(indexed: tuple[Path, Path]) -> None:
    root, db = indexed
    hits = SearchEngine(root, db_path=db).regex_search(r"def verify_\w+", limit=10)
    assert any("auth.py" in h.path and "verify_signature" in h.text for h in hits)


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_regex_search_handles_colon_in_path_and_validates_failures(tmp_path: Path) -> None:
    (tmp_path / "module:variant.py").write_text("needle = True\n")
    engine = SearchEngine(tmp_path)

    hits = engine.regex_search("needle", limit=5)
    assert [(hit.path, hit.line) for hit in hits] == [("module:variant.py", 1)]
    assert engine.regex_search("needle", limit=0) == []
    with pytest.raises(RuntimeError, match="ripgrep search failed"):
        engine.regex_search("[")


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_regex_search_honors_limits_above_fifty(tmp_path: Path) -> None:
    (tmp_path / "many.txt").write_text("".join(f"needle {line}\n" for line in range(60)))
    assert len(SearchEngine(tmp_path).regex_search("needle", limit=55)) == 55


def test_embedder_from_id_roundtrip() -> None:
    e = HashingEmbedder(dim=128)
    restored = embedder_from_id(e.id)
    assert restored.dim == 128
    a = e.embed_query("hello world")
    b = restored.embed_query("hello world")
    assert a == b


def test_hashing_embedder_supports_unicode_and_rejects_invalid_dimensions() -> None:
    assert any(HashingEmbedder(dim=32).embed_query("π"))
    with pytest.raises(ValueError, match="positive"):
        HashingEmbedder(dim=0)
    with pytest.raises(ValueError, match="positive"):
        embedder_from_id("hashing--1")


def test_semantic_search_falls_back_when_query_embedding_fails(indexed: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    root, db = indexed

    class FailingEmbedder:
        def embed_query(self, _query: str) -> list[float]:
            raise RuntimeError("backend unavailable")

    monkeypatch.setattr("codescope.index.search.embedder_from_id", lambda _embedder_id: FailingEmbedder())
    hits = SearchEngine(root, db_path=db).semantic_search("validate token", limit=5)
    assert any(hit.name == "validate_token" and hit.sources == ["bm25"] for hit in hits)


def test_find_similar_code_finds_matching_symbol(indexed: tuple[Path, Path]) -> None:
    root, db = indexed
    snippet = "def validate_token(token):\n    return verify_signature(token)\n"
    hits = SearchEngine(root, db_path=db).find_similar_code(snippet, limit=5)
    assert hits
    assert hits[0].name == "validate_token"
    assert -1.0 <= hits[0].score <= 1.0


@pytest.fixture
def indexed_with_clones(tmp_path: Path) -> tuple[Path, Path]:
    body = "def {name}(items):\n    total = 0\n    for item in items:\n        total = total + item\n    return total\n"
    (tmp_path / "a.py").write_text(body.format(name="sum_items"))
    (tmp_path / "b.py").write_text(body.format(name="sum_items"))
    (tmp_path / "c.py").write_text("def unrelated(path):\n    with open(path) as f:\n        return f.read().upper()\n")
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


def test_clone_checks_fail_loudly_without_embeddings(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("def example():\n    return 1\n")
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embeddings=False)
    engine = SearchEngine(tmp_path, db_path=db)

    with pytest.raises(RuntimeError, match="requires embeddings"):
        engine.find_duplicate_code()
    with pytest.raises(RuntimeError, match="requires embeddings"):
        engine.detect_clones_in_diff()


def test_added_blocks_parses_new_file() -> None:
    from codescope.devops.vcs import _parse_added_blocks

    diff_text = (
        "diff --git a/new.py b/new.py\n"
        "new file mode 100644\n"
        "index 0000000..1111111\n"
        "--- /dev/null\n"
        "+++ b/new.py\n"
        "@@ -0,0 +1,3 @@\n"
        "+def f():\n"
        "+    return 1\n"
        "+\n"
    )
    blocks = _parse_added_blocks(diff_text, min_lines=1)
    assert len(blocks) == 1
    assert blocks[0].path == "new.py"
    assert blocks[0].start_line == 1
    assert blocks[0].end_line == 3
    assert "def f():" in blocks[0].text


def test_detect_clones_in_diff_flags_duplicate(tmp_path: Path) -> None:
    pygit2 = pytest.importorskip("pygit2")
    body = "def sum_items(items):\n    total = 0\n    for item in items:\n        total = total + item\n    return total\n"
    pygit2.init_repository(str(tmp_path))
    (tmp_path / "orig.py").write_text(body)
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embedder=HashingEmbedder())

    # A brand-new file re-implementing the same function (intent-to-add so it
    # shows up in `git diff`); it is NOT in the index, so no self-match.
    (tmp_path / "copy.py").write_text(body)
    import subprocess

    subprocess.run(["git", "add", "-N", "copy.py"], cwd=tmp_path, check=True)

    findings = SearchEngine(tmp_path, db_path=db).detect_clones_in_diff(min_lines=3, similarity=0.95)
    assert findings
    assert findings[0].added_path == "copy.py"
    assert findings[0].matches.name == "sum_items"
    assert findings[0].matches.path == "orig.py"
    assert findings[0].similarity >= 0.95


def test_detect_clones_in_diff_clean_when_no_duplication(tmp_path: Path) -> None:
    pygit2 = pytest.importorskip("pygit2")
    pygit2.init_repository(str(tmp_path))
    (tmp_path / "orig.py").write_text("def sum_items(items):\n    return sum(items)\n")
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embedder=HashingEmbedder())

    (tmp_path / "fresh.py").write_text(
        "def render_html_table(rows):\n"
        "    cells = [str(value).upper() for row in rows for value in row]\n"
        "    return '<table>' + ''.join(cells) + '</table>'\n"
    )
    import subprocess

    subprocess.run(["git", "add", "-N", "fresh.py"], cwd=tmp_path, check=True)

    findings = SearchEngine(tmp_path, db_path=db).detect_clones_in_diff(min_lines=3, similarity=0.95)
    assert findings == []


def test_search_tools_registered() -> None:
    from codescope.cli import register_codescope_tools

    register_codescope_tools()
    from serena.tools import ToolRegistry

    names = ToolRegistry().get_tool_names()
    for t in (
        "search_code",
        "search_semantic",
        "search_regex",
        "find_similar_code",
        "find_duplicate_code",
        "detect_clones_in_diff",
    ):
        assert t in names
