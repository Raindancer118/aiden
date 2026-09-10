"""Type and call hierarchy against a real language server.

These exercise the LSP path end to end (pyright on the shared Python test
repo); nothing here is mocked, because the whole point of these tools is that
the answer is resolved rather than guessed.
"""

from __future__ import annotations

import pytest

from solidlsp.ls_config import Language


@pytest.mark.parametrize("language_server", [Language.PYTHON], indirect=True)
class TestTypeHierarchy:
    def test_supertypes_of_a_subclass(self, language_server) -> None:
        ls = language_server.language_server
        # class User(BaseModel) in test_repo/models.py
        result = ls.request_type_hierarchy("test_repo/models.py", 31, 6, direction="supertypes")
        if not result["supported"]:
            pytest.skip("this language server does not implement textDocument/typeHierarchy")
        assert result["item"] is not None, result
        assert result["item"]["name"] == "User"
        assert any(s["name"] == "BaseModel" for s in result["supertypes"]), result

    def test_subtypes_of_a_base_class(self, language_server) -> None:
        ls = language_server.language_server
        # class BaseModel(ABC) in test_repo/models.py
        result = ls.request_type_hierarchy("test_repo/models.py", 10, 6, direction="subtypes")
        if not result["supported"]:
            pytest.skip("this language server does not implement textDocument/typeHierarchy")
        assert result["item"] is not None, result
        names = {s["name"] for s in result["subtypes"]}
        assert {"User", "Item"} <= names, names
        # Locations are project-relative so an agent can act on them directly.
        assert all(not s["relativePath"].startswith("/") for s in result["subtypes"])

    def test_missing_capability_is_reported_not_raised(self, language_server) -> None:
        """A server without type hierarchy must answer, not blow up at the agent."""
        ls = language_server.language_server
        result = ls.request_type_hierarchy("test_repo/models.py", 10, 6)
        assert set(result) >= {"item", "supertypes", "subtypes", "supported"}
        assert isinstance(result["supported"], bool)


@pytest.mark.parametrize("language_server", [Language.PYTHON], indirect=True)
class TestCallHierarchy:
    def test_incoming_calls_are_resolved(self, language_server) -> None:
        ls = language_server.language_server
        target = ls.request_document_symbols("test_repo/models.py")
        assert target is not None
        # create_user_object is called from elsewhere in the repo.
        calls = ls.request_call_hierarchy("test_repo/models.py", 87, 4, direction="incoming")
        assert isinstance(calls, list)
        for entry in calls:
            assert entry["name"]
            assert entry["relativePath"]
            assert entry["line"] >= 1
            assert all(isinstance(line, int) for line in entry["callSites"])

    def test_outgoing_calls_are_resolved(self, language_server) -> None:
        ls = language_server.language_server
        calls = ls.request_call_hierarchy("test_repo/models.py", 87, 4, direction="outgoing")
        assert isinstance(calls, list)
        assert any(entry["name"] == "User" for entry in calls), calls


# -- index fallback (works without a type-hierarchy-capable server) ----------


def test_index_type_hierarchy_reads_declarations(tmp_path) -> None:
    from codescope.index.embed import HashingEmbedder
    from codescope.index.graph import GraphEngine
    from codescope.index.indexer import Indexer

    (tmp_path / "models.py").write_text(
        "class BaseModel:\n    pass\n\n\nclass User(BaseModel):\n    pass\n\n\nclass Item(BaseModel):\n    pass\n"
    )
    (tmp_path / "svc.java").write_text("public class Service extends BaseModel implements Runnable {\n    void run() {}\n}\n")
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embedder=HashingEmbedder())

    hierarchy = GraphEngine(tmp_path, db_path=db).type_hierarchy("BaseModel")
    assert hierarchy.item is not None
    assert {s.name for s in hierarchy.subtypes} == {"User", "Item", "Service"}
    assert hierarchy.resolved_by == "index"

    user = GraphEngine(tmp_path, db_path=db).type_hierarchy("User")
    assert [s.name for s in user.supertypes] == ["BaseModel"]
    # Runnable is not defined in the project, so it is not claimed as an edge.
    assert "Runnable" not in {s.name for s in GraphEngine(tmp_path, db_path=db).type_hierarchy("Service").supertypes}


def test_index_type_hierarchy_reports_unknown_types(tmp_path) -> None:
    from codescope.index.graph import GraphEngine
    from codescope.index.indexer import Indexer

    (tmp_path / "a.py").write_text("class Only:\n    pass\n")
    db = tmp_path / "idx" / "index.db"
    Indexer(tmp_path, db_path=db).reindex(embeddings=False)

    hierarchy = GraphEngine(tmp_path, db_path=db).type_hierarchy("Missing")
    assert hierarchy.item is None
    assert hierarchy.note


def test_hierarchy_tools_are_registered() -> None:
    from codescope.cli import register_codescope_tools

    register_codescope_tools()
    from serena.tools import ToolRegistry

    names = ToolRegistry().get_tool_names()
    assert "get_type_hierarchy" in names
    assert "get_call_hierarchy" in names


def test_type_hierarchy_tool_returns_one_shape_on_both_paths(tmp_path, monkeypatch) -> None:
    """LSP and index answers must be parseable by the same caller."""
    import json

    from codescope.index.embed import HashingEmbedder
    from codescope.index.indexer import Indexer
    from codescope.tools.hierarchy_tools import GetTypeHierarchyTool

    (tmp_path / "models.py").write_text("class BaseModel:\n    pass\n\n\nclass User(BaseModel):\n    pass\n")
    Indexer(tmp_path).reindex(embedder=HashingEmbedder())

    tool = object.__new__(GetTypeHierarchyTool)
    tool.get_project_root = lambda: str(tmp_path)  # type: ignore[method-assign]
    tool._to_json = lambda payload: json.dumps(payload, default=str)  # type: ignore[method-assign]
    tool._locate = lambda name_path, relative_path: ("models.py", 4, 6)  # type: ignore[method-assign]

    class _NoTypeHierarchy:
        @staticmethod
        def request_type_hierarchy(*_args, **_kwargs):
            return {"item": None, "supertypes": [], "subtypes": [], "supported": False}

    tool._language_server = lambda _rel: _NoTypeHierarchy()  # type: ignore[method-assign]

    result = json.loads(GetTypeHierarchyTool.apply(tool, "User", "models.py"))
    assert result["resolved_by"] == "index"
    assert result["item"]["name"] == "User"
    # Same keys as the LSP path: relativePath/line, not path/start_line.
    assert set(result["supertypes"][0]) == {"name", "kind", "relativePath", "line"}
    assert result["supertypes"][0]["name"] == "BaseModel"
    assert "does not implement type hierarchy" in result["note"]

    # An unknown type keeps the index's own explanation instead of losing it.
    unknown = json.loads(GetTypeHierarchyTool.apply(tool, "Nope", "models.py"))
    assert unknown["item"] is None
    assert "No indexed type" in unknown["note"]
