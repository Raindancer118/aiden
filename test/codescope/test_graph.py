"""Tests for the Codescope code-graph engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from codescope.index.graph import GraphEngine
from codescope.index.indexer import Indexer


@pytest.fixture
def graph(tmp_path: Path) -> GraphEngine:
    (tmp_path / "app.py").write_text(
        "def main():\n"
        "    return process(load())\n"
        "\n"
        "def process(data):\n"
        "    return transform(data)\n"
        "\n"
        "def transform(data):\n"
        "    return data\n"
        "\n"
        "def load():\n"
        "    return read_source()\n"
        "\n"
        "def read_source():\n"
        "    return 'x'\n"
    )
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embeddings=False)
    return GraphEngine(tmp_path, db_path=db)


def test_dependencies(graph: GraphEngine) -> None:
    names = {d.name for d in graph.dependencies("main")}
    assert {"process", "load"} <= names


def test_dependents(graph: GraphEngine) -> None:
    assert {d.name for d in graph.dependents("transform")} == {"process"}
    assert {d.name for d in graph.dependents("load")} == {"main"}


def test_call_chain(graph: GraphEngine) -> None:
    tree = graph.call_chain("main", depth=3)
    assert tree.name == "main"
    first_level = {c.name for c in tree.children}
    assert {"process", "load"} <= first_level
    process_node = next(c for c in tree.children if c.name == "process")
    assert any(gc.name == "transform" for gc in process_node.children)


def test_call_chain_expands_shared_dependency_on_each_branch(tmp_path: Path) -> None:
    (tmp_path / "diamond.py").write_text(
        "def root():\n"
        "    left()\n"
        "    right()\n"
        "\n"
        "def left():\n"
        "    shared()\n"
        "\n"
        "def right():\n"
        "    shared()\n"
        "\n"
        "def shared():\n"
        "    leaf()\n"
        "\n"
        "def leaf():\n"
        "    return 1\n"
    )
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embeddings=False)

    tree = GraphEngine(tmp_path, db_path=db).call_chain("root", depth=4)
    for branch_name in ("left", "right"):
        branch = next(child for child in tree.children if child.name == branch_name)
        shared = next(child for child in branch.children if child.name == "shared")
        assert [child.name for child in shared.children] == ["leaf"]


def test_call_chain_does_not_merge_same_named_symbol_bodies(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def run():\n    alpha()\n\ndef alpha():\n    pass\n")
    (tmp_path / "b.py").write_text("def run():\n    beta()\n\ndef beta():\n    pass\n")
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embeddings=False)

    tree = GraphEngine(tmp_path, db_path=db).call_chain("run", depth=2)
    assert tree.path == "a.py"
    assert {child.name for child in tree.children} == {"alpha"}


def test_change_impact(graph: GraphEngine) -> None:
    tree = graph.change_impact("read_source", depth=4)
    # read_source <- load <- main
    load_node = next((c for c in tree.children if c.name == "load"), None)
    assert load_node is not None
    assert any(gc.name == "main" for gc in load_node.children)


def test_file_summary(graph: GraphEngine) -> None:
    summary = graph.file_summary("app.py")
    assert summary is not None
    assert summary.language == "python"
    assert {s.name for s in summary.symbols} >= {"main", "process", "transform", "load", "read_source"}


def test_project_map(graph: GraphEngine) -> None:
    pmap = graph.project_map()
    assert len(pmap) == 1
    entry = pmap[0]
    assert entry["path"] == "app.py"
    assert entry["symbol_count"] >= 5


def test_graph_tools_registered() -> None:
    from codescope.cli import register_codescope_tools

    register_codescope_tools()
    from serena.tools import ToolRegistry

    names = ToolRegistry().get_tool_names()
    for t in ("get_dependencies", "get_dependents", "get_call_chain", "get_change_impact", "get_project_map", "get_file_summary"):
        assert t in names
