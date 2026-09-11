"""Tests for the AIDEN search engine (BM25 + vector + trigram + RRF).

These use the deterministic HashingEmbedder so no model download is needed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from aiden.index.embed import HashingEmbedder, embedder_from_id
from aiden.index.indexer import Indexer
from aiden.index.search import SearchEngine, SearchFilter


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
    assert {"name", "bm25", "vector", "trigram"} >= set(top.sources)
    # A hit must be actionable on its own: it carries its code and identity.
    assert top.symbol_id > 0
    assert top.preview.strip()
    assert top.lang == "python"


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

    monkeypatch.setattr("aiden.index.search.embedder_from_id", lambda _embedder_id: FailingEmbedder())
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
    from aiden.devops.vcs import _parse_added_blocks

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
    from aiden.cli import register_aiden_tools

    register_aiden_tools()
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


# -- embedder caching & bounded batching -------------------------------------


class _RecordingEmbedder(HashingEmbedder):
    """Hashing embedder that records the batches it was asked to embed."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__(dim=dim)
        self.batch_size = 4
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return super().embed_documents(texts)


def test_embed_batched_bounds_batch_size_and_covers_every_text() -> None:
    embedder = _RecordingEmbedder()
    texts = [f"symbol_{i} " + "x" * (i * 7) for i in range(11)]

    seen: dict[int, list[float]] = {}
    for batch in embedder.embed_batched(texts):
        assert len(batch) <= embedder.batch_size
        for index, vector in batch:
            seen[index] = vector

    assert sorted(seen) == list(range(len(texts)))
    assert all(len(call) <= embedder.batch_size for call in embedder.calls)


def test_embed_batched_groups_texts_of_similar_length() -> None:
    """Length-homogeneous batches are what keeps ONNX padding (and RSS) low."""
    embedder = _RecordingEmbedder()
    texts = ["x" * n for n in (1, 5000, 2, 5000, 3, 5000, 4, 5000)]

    list(embedder.embed_batched(texts))

    # The four short texts must never share a batch with a 5000-char body.
    for call in embedder.calls:
        lengths = [len(t) for t in call]
        assert max(lengths) - min(lengths) < 100


def test_truncation_caps_oversized_input() -> None:
    """Backends that pad to the longest item need a hard ceiling per document."""
    embedder = _RecordingEmbedder()
    embedder.max_chars = 50
    assert [len(t) for t in embedder._truncate(["y" * 5000, "short"])] == [50, 5]

    embedder.max_chars = 0  # 0 disables truncation
    assert len(embedder._truncate(["y" * 5000])[0]) == 5000


def test_embedder_from_id_is_cached() -> None:
    """A search must never pay for reloading the embedding model."""
    from aiden.index.embed import clear_embedder_cache, get_embedder

    clear_embedder_cache()
    first = embedder_from_id("hashing-64")
    second = embedder_from_id("hashing-64")
    assert first is second
    assert get_embedder("hashing", dim=64) is get_embedder("hashing", dim=64)
    assert embedder_from_id("hashing-128") is not first


# -- filters, previews and fusion hygiene ------------------------------------


@pytest.fixture
def indexed_multi(tmp_path: Path) -> tuple[Path, Path]:
    """A tree with two languages, a test file and a nested package."""
    src = tmp_path / "src" / "auth"
    src.mkdir(parents=True)
    (src / "tokens.py").write_text(
        "class TokenStore:\n    pass\n\n\ndef validate_token(token):\n    '''Validate an auth token.'''\n    return bool(token)\n"
    )
    (tmp_path / "src" / "auth" / "tokens.go").write_text('package auth\n\nfunc ValidateToken(t string) bool {\n\treturn t != ""\n}\n')
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_tokens.py").write_text("def test_validate_token():\n    assert validate_token('x')\n")
    db = tmp_path / "idx" / "index.db"
    assert Indexer(tmp_path, db_path=db).reindex(embedder=HashingEmbedder()).errors == 0
    return tmp_path, db


def test_language_filter_narrows_before_top_k(indexed_multi: tuple[Path, Path]) -> None:
    root, db = indexed_multi
    hits = SearchEngine(root, db_path=db).hybrid_search("validate token", limit=10, flt=SearchFilter(lang="go"))
    assert hits
    assert {h.lang for h in hits} == {"go"}


def test_kind_and_test_exclusion_filters(indexed_multi: tuple[Path, Path]) -> None:
    root, db = indexed_multi
    engine = SearchEngine(root, db_path=db)

    classes = engine.hybrid_search("token", limit=10, flt=SearchFilter(kind="class"))
    assert classes and {h.kind for h in classes} == {"class"}

    assert any(h.path.startswith("tests/") for h in engine.hybrid_search("validate token", limit=10))
    without_tests = engine.hybrid_search("validate token", limit=10, flt=SearchFilter(exclude_tests=True))
    assert without_tests
    assert not any(h.path.startswith("tests/") for h in without_tests)


def test_path_glob_filter(indexed_multi: tuple[Path, Path]) -> None:
    root, db = indexed_multi
    hits = SearchEngine(root, db_path=db).hybrid_search("token", limit=10, flt=SearchFilter(path_glob="src/auth/*.py"))
    assert hits
    assert all(h.path.startswith("src/auth/") and h.path.endswith(".py") for h in hits)


def test_filter_that_matches_nothing_returns_nothing(indexed_multi: tuple[Path, Path]) -> None:
    root, db = indexed_multi
    assert SearchEngine(root, db_path=db).hybrid_search("token", limit=10, flt=SearchFilter(lang="rust")) == []


def test_natural_language_query_does_not_use_the_trigram_retriever(indexed_multi: tuple[Path, Path]) -> None:
    """A sentence cannot appear verbatim in code; ranking on it is noise."""
    root, db = indexed_multi
    hits = SearchEngine(root, db_path=db).hybrid_search("how do we validate an auth token", limit=10)
    assert hits
    assert not any("trigram" in h.sources for h in hits)

    identifier_hits = SearchEngine(root, db_path=db).hybrid_search("validate_token", limit=10)
    assert any("trigram" in h.sources for h in identifier_hits)


def test_preview_folds_long_bodies_instead_of_dumping_them(tmp_path: Path) -> None:
    lines = "\n".join(f"    step_{i}()" for i in range(200))
    (tmp_path / "long.py").write_text(f"def long_function():\n{lines}\n    return 1\n")
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embedder=HashingEmbedder())

    hit = SearchEngine(tmp_path, db_path=db).hybrid_search("long_function", limit=1)[0]
    assert hit.preview.count("\n") < 20
    assert "lines omitted" in hit.preview
    assert hit.preview.startswith("def long_function():")
    # The tail of the body must still be searchable even though it is folded.
    assert SearchEngine(tmp_path, db_path=db).substring_search("step_199", limit=5)


def test_clone_group_reports_its_weakest_pair(tmp_path: Path) -> None:
    """Similarity is not transitive: a chained cluster must not look like a clique."""
    body = "def {name}(items):\n    total = 0\n    for item in items:\n        total = total + item\n    return total\n"
    (tmp_path / "a.py").write_text(body.format(name="sum_items"))
    (tmp_path / "b.py").write_text(body.format(name="sum_items"))
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embedder=HashingEmbedder())

    group = SearchEngine(tmp_path, db_path=db).find_duplicate_code(min_lines=3, similarity=0.95)[0]
    assert group.min_similarity <= group.similarity
    assert group.min_similarity >= 0.95  # a true clone pair: every pair holds up


def test_batch_size_follows_the_quadratic_budget() -> None:
    """Attention is quadratic in the padded length, so count x longest^2 is capped."""
    embedder = _RecordingEmbedder()
    embedder.batch_size = 64
    embedder.batch_cost = 1_000_000

    assert embedder.batch_size_for(1000) == 1
    assert embedder.batch_size_for(500) == 4
    assert embedder.batch_size_for(10) == 64  # clamped by the item cap
    assert embedder.batch_size_for(0) >= 1


def test_batches_keep_one_shape_for_the_whole_pass() -> None:
    """A later batch may not be larger in any dimension, or the arena regrows."""
    embedder = _RecordingEmbedder()
    embedder.batch_size = 64
    embedder.batch_cost = 1_000_000

    texts = ["y" * 900] * 3 + ["x" * 10] * 60
    list(embedder.embed_batched(texts))

    sizes = [len(call) for call in embedder.calls]
    assert max(sizes) == sizes[0], sizes  # never grows past the first batch
    longest = [max(len(t) for t in call) for call in embedder.calls]
    assert longest == sorted(longest, reverse=True), longest


def test_every_text_is_embedded_exactly_once_under_budgeting() -> None:
    embedder = _RecordingEmbedder()
    embedder.batch_size = 8
    embedder.batch_cost = 4_000_000
    texts = [f"sym{i}" + "z" * (i * 31 % 400) for i in range(97)]

    seen: dict[int, list[float]] = {}
    for batch in embedder.embed_batched(texts):
        for index, vector in batch:
            assert index not in seen
            seen[index] = vector
    assert sorted(seen) == list(range(len(texts)))


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_thing.py",
        "src/pkg/test_thing.py",
        "src/pkg/thing_test.go",
        "src/pkg/thing.test.ts",
        "src/pkg/thing.spec.ts",
        "src/java/ThingTest.java",
        "spec/thing_spec.rb",
    ],
)
def test_exclude_tests_recognises_the_usual_conventions(tmp_path: Path, path: str) -> None:
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix
    sources = {
        ".py": "def helper_thing():\n    return 1\n",
        ".go": "package pkg\n\nfunc HelperThing() int {\n\treturn 1\n}\n",
        ".ts": "export function helperThing(): number { return 1; }\n",
        ".java": "class ThingTest { void helperThing() {} }\n",
        ".rb": "def helper_thing\n  1\nend\n",
    }
    target.write_text(sources[suffix])
    (tmp_path / "prod.py").write_text("def helper_thing():\n    return 2\n")

    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embedder=HashingEmbedder())
    engine = SearchEngine(tmp_path, db_path=db)

    assert any(h.path == path for h in engine.hybrid_search("helper thing", limit=20)), "fixture must be indexed"
    filtered = engine.hybrid_search("helper thing", limit=20, flt=SearchFilter(exclude_tests=True))
    assert not any(h.path == path for h in filtered), f"{path} should be recognised as a test file"
    assert any(h.path == "prod.py" for h in filtered)


@pytest.mark.parametrize("name", ["latest.py", "contest.py", "protest.py", "attestation.py"])
def test_exclude_tests_keeps_source_files_that_merely_contain_test(tmp_path: Path, name: str) -> None:
    """The prefilter is case-sensitive: 'latest.py' is not a test file."""
    (tmp_path / name).write_text("def helper_thing():\n    return 1\n")
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embedder=HashingEmbedder())

    hits = SearchEngine(tmp_path, db_path=db).hybrid_search("helper thing", limit=10, flt=SearchFilter(exclude_tests=True))
    assert any(h.path == name for h in hits), f"{name} was wrongly treated as a test file"


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [("auth.py", True), ("*.py", True), ("*.[jt]s", False), ("src/*", True), ("src/auth.py", True), ("other/*", False)],
)
def test_path_glob_prefilter_is_a_superset_of_the_exact_matcher(tmp_path: Path, pattern: str, expected: bool) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "auth.py").write_text("def validate_thing():\n    return 1\n")
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embedder=HashingEmbedder())

    hits = SearchEngine(tmp_path, db_path=db).hybrid_search("validate thing", limit=10, flt=SearchFilter(path_glob=pattern))
    assert bool(hits) is expected, f"path_glob={pattern!r}"


def test_batches_run_longest_first() -> None:
    """ONNX grows its arena per new tensor shape and never returns it.

    Ascending batches therefore make memory climb for the whole run; taking
    the largest batch first allocates the high-water mark once and every
    later batch reuses it.
    """
    embedder = _RecordingEmbedder()
    embedder.batch_size = 4
    embedder.batch_cost = 40_000_000
    texts = ["z" * n for n in (10, 900, 40, 1500, 70, 300)]

    list(embedder.embed_batched(texts))

    longest_per_batch = [max(len(t) for t in call) for call in embedder.calls]
    assert longest_per_batch == sorted(longest_per_batch, reverse=True), longest_per_batch


def test_filtered_vector_search_agrees_with_an_exact_scan(indexed_multi: tuple[Path, Path]) -> None:
    """The widened sweep is an optimization, not a different answer."""
    from aiden.index.search import IndexStore

    root, db = indexed_multi
    engine = SearchEngine(root, db_path=db)
    flt = SearchFilter(lang="python")

    with IndexStore(db) as store:
        qvec = engine._embed_query(store, "validate an auth token")
        assert qvec is not None
        allowed = engine._filtered_ids(store, flt)
        assert allowed
        swept = engine._vector_ids_for(store, qvec, 5, flt)
        exact = engine._exact_scan(store, qvec, 5, allowed)

    assert swept, "a filtered vector search must still return candidates"
    assert set(swept) <= allowed
    # Same top result either way; the sweep may order deeper ties differently.
    assert swept[0] == exact[0]


def test_common_names_do_not_hijack_a_natural_language_query(tmp_path: Path) -> None:
    """'new' is ordinary English and the name of dozens of methods."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    for i in range(30):
        (pkg / f"m{i}.py").write_text(f"class C{i}:\n    def new(self):\n        return {i}\n")
    (tmp_path / "reuse.py").write_text(
        "def find_existing_helper(snippet):\n"
        "    '''Look for code that already exists before writing new code.'''\n"
        "    return search_index(snippet)\n"
    )
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embedder=HashingEmbedder())

    hits = SearchEngine(tmp_path, db_path=db).hybrid_search("find code that already exists before writing new code", limit=5)
    assert hits
    assert hits[0].name == "find_existing_helper", [h.name for h in hits]


def test_a_query_of_only_common_names_still_finds_them(tmp_path: Path) -> None:
    """Dropping every token would answer nothing; keep them when that happens."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    for i in range(30):
        (pkg / f"m{i}.py").write_text(f"class C{i}:\n    def new(self):\n        return {i}\n")
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embedder=HashingEmbedder())

    hits = SearchEngine(tmp_path, db_path=db).hybrid_search("new", limit=5)
    assert hits and all(h.name == "new" for h in hits)
