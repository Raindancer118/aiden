"""Tests for the one-call code-context bundle."""

from __future__ import annotations

from pathlib import Path

import pytest

from codescope.index.context import ContextEngine
from codescope.index.embed import HashingEmbedder
from codescope.index.indexer import Indexer


@pytest.fixture
def project(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "auth.py").write_text(
        "def validate_token(token):\n"
        "    '''Validate an auth token.'''\n"
        "    return verify_signature(token)\n"
        "\n"
        "def verify_signature(token):\n"
        "    return bool(token)\n"
        "\n"
        "def login(token):\n"
        "    return validate_token(token)\n"
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_auth.py").write_text("def test_validate_token():\n    assert validate_token('x')\n")
    db = tmp_path / "idx" / "index.db"
    assert Indexer(tmp_path, db_path=db).reindex(embedder=HashingEmbedder()).errors == 0
    return tmp_path, db


def test_context_bundles_code_callers_callees_and_tests(project: tuple[Path, Path]) -> None:
    root, db = project
    context = ContextEngine(root, db_path=db).build("validate_token")

    assert context.symbols
    focus = next(s for s in context.symbols if s.name == "validate_token")
    assert focus.path == "auth.py"
    assert "return verify_signature(token)" in focus.code
    assert focus.lang == "python"
    assert focus.symbol_id > 0
    assert not focus.truncated

    assert any(c.name == "login" for c in focus.called_by)
    assert any(c.name == "verify_signature" for c in focus.calls)
    assert any(t.path.startswith("tests/") for t in focus.tests)
    # Callers and tests are disjoint views of the same reference set.
    assert not any(c.path.startswith("tests/") for c in focus.called_by)


def test_context_can_omit_tests_and_bound_its_size(project: tuple[Path, Path]) -> None:
    root, db = project
    context = ContextEngine(root, db_path=db).build("validate_token", include_tests=False, body_lines=2, related=1)
    focus = next(s for s in context.symbols if s.name == "validate_token")
    assert focus.tests == []
    assert focus.truncated
    assert len(focus.called_by) <= 1


def test_context_marks_ambiguous_edges_speculative(tmp_path: Path) -> None:
    """A name defined twice cannot be resolved by name alone -- say so."""
    (tmp_path / "a.py").write_text("def save():\n    return 1\n\n\ndef writer():\n    return save()\n")
    (tmp_path / "b.py").write_text("def save():\n    return 2\n")
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embedder=HashingEmbedder())

    context = ContextEngine(tmp_path, db_path=db).build("writer")
    focus = next(s for s in context.symbols if s.name == "writer")
    assert focus.calls
    assert all(c.speculative for c in focus.calls if c.name == "save")


def test_context_warns_instead_of_returning_a_bare_empty_list(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def only_thing():\n    return 1\n")
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embeddings=False)

    context = ContextEngine(tmp_path, db_path=db).build("something that does not exist at all")
    assert context.warnings


def test_context_tool_is_registered() -> None:
    from codescope.cli import register_codescope_tools

    register_codescope_tools()
    from serena.tools import ToolRegistry

    assert "get_code_context" in ToolRegistry().get_tool_names()
