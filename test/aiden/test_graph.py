"""Tests for the AIDEN code-graph engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiden.index.graph import GraphEngine
from aiden.index.indexer import Indexer
from aiden.index.store import IndexStore


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
    from aiden.cli import register_aiden_tools

    register_aiden_tools()
    from serena.tools import ToolRegistry

    names = ToolRegistry().get_tool_names()
    for t in ("get_dependencies", "get_dependents", "get_call_chain", "get_change_impact", "get_project_map", "get_file_summary"):
        assert t in names


# -- materialized reference ownership ----------------------------------------


def test_refs_record_their_owning_symbol_at_index_time(tmp_path: Path) -> None:
    """A reference belongs to the innermost symbol whose body contains it."""
    (tmp_path / "app.py").write_text("def helper():\n    return 1\n\n\nclass Service:\n    def run(self):\n        return helper()\n")
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embeddings=False)

    with IndexStore(db) as store:
        owners = dict(
            store.conn.execute("SELECT r.name, s.name FROM refs AS r JOIN symbols AS s ON s.id = r.owner_id WHERE r.name = 'helper'")
        )
        # The call to helper() sits inside run(), not inside the class body.
        assert owners.get("helper") == "run", owners
        assert store.conn.execute("SELECT COUNT(*) FROM refs WHERE owner_id IS NULL AND line > 0").fetchone()[0] >= 0


def test_dependents_use_the_stored_owner(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def helper():\n    return 1\n\n\ndef caller():\n    return helper()\n")
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embeddings=False)

    dependents = GraphEngine(tmp_path, db_path=db).dependents("helper")
    assert [d.name for d in dependents] == ["caller"]
    assert dependents[0].path == "app.py"


def test_existing_index_is_migrated_without_a_reindex(tmp_path: Path) -> None:
    """An index built before ownership existed must not need rebuilding."""
    (tmp_path / "app.py").write_text("def helper():\n    return 1\n\n\ndef caller():\n    return helper()\n")
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embeddings=False)

    # Recreate the *actual* old schema: a refs table with no owner_id column.
    with IndexStore(db) as store:
        rows = store.conn.execute("SELECT id, path, name, kind, line, col FROM refs").fetchall()
        store.conn.execute("DROP TABLE refs")
        store.conn.execute(
            "CREATE TABLE refs (id INTEGER PRIMARY KEY, path TEXT NOT NULL, name TEXT NOT NULL, "
            "kind TEXT NOT NULL, line INTEGER NOT NULL, col INTEGER NOT NULL)"
        )
        store.conn.executemany("INSERT INTO refs(id, path, name, kind, line, col) VALUES(?,?,?,?,?,?)", rows)
        store.conn.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
        store.commit()

    # Opening the store adds the column and backfills it.
    with IndexStore(db) as store:
        assert "owner_id" in {r[1] for r in store.conn.execute("PRAGMA table_info(refs)")}
        assert store.conn.execute("SELECT COUNT(*) FROM refs WHERE owner_id IS NOT NULL").fetchone()[0] > 0
        assert store.get_meta("schema_version") == "2"

    assert [d.name for d in GraphEngine(tmp_path, db_path=db).dependents("helper")] == ["caller"]


def test_references_outside_any_symbol_are_stored_without_an_owner(tmp_path: Path) -> None:
    """Module-level code has no owning symbol; that is a null, not a crash."""
    (tmp_path / "script.py").write_text("import os\n\nprint(os.getcwd())\n")
    db = tmp_path / "idx" / "index.db"
    assert Indexer(tmp_path, db_path=db).reindex(embeddings=False).errors == 0

    with IndexStore(db) as store:
        rows = store.conn.execute("SELECT name, owner_id FROM refs WHERE path='script.py'").fetchall()
        assert rows, "the module-level call should still be indexed as a reference"
        assert all(owner is None for _name, owner in rows)


def test_nested_function_calls_belong_to_the_inner_function(tmp_path: Path) -> None:
    """Ownership is what separates an outer symbol's callees from an inner one's."""
    (tmp_path / "app.py").write_text(
        "def leaf():\n    return 1\n\n\ndef outer():\n    def inner():\n        return leaf()\n    return inner\n"
    )
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embeddings=False)
    graph = GraphEngine(tmp_path, db_path=db)

    assert [d.name for d in graph.dependencies("inner")] == ["leaf"]
    # outer() calls inner(), not leaf() -- the call to leaf is inner's.
    assert "leaf" not in [d.name for d in graph.dependencies("outer")]
    assert [d.name for d in graph.dependents("leaf")] == ["inner"]
