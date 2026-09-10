"""Start-up / smoke tests for the Codescope tool layer.

These verify that Codescope is importable and that its tools register cleanly
into Serena's ToolRegistry via the runtime hook in ``codescope.cli`` - i.e.
that the fork's core integration mechanism works without editing any Serena
source file.
"""

import subprocess
import sys


def test_codescope_importable() -> None:
    import codescope

    assert codescope.__version__
    assert codescope.PRODUCT_NAME == "Volantic Codescope"


def test_info_tool_name() -> None:
    from codescope.tools.info_tools import CodescopeInfoTool

    # Serena derives the MCP tool name from the class name minus "Tool", snake_cased.
    assert CodescopeInfoTool.get_name_from_cls() == "codescope_info"


def test_registration_appends_tool_package() -> None:
    from codescope.cli import LOCAL_MEMORY_TOOLS, register_codescope_tools
    from serena.tools import tools_base

    register_codescope_tools()
    assert "codescope.tools" in tools_base.tool_packages
    assert tools_base.tool_names_excluded_from_registry >= LOCAL_MEMORY_TOOLS


def test_info_tool_registered_in_registry() -> None:
    from codescope.cli import register_codescope_tools

    register_codescope_tools()

    from serena.tools import ToolRegistry

    registry = ToolRegistry()
    tool_names = registry.get_tool_names()
    assert "codescope_info" in tool_names, f"codescope_info not registered; available tools: {sorted(tool_names)}"
    assert "memory_project_context" in tool_names
    assert "memory_upsert" in tool_names
    assert "read_memory" not in tool_names
    assert "write_memory" not in tool_names
    assert "onboarding" not in tool_names


def test_info_tool_apply_runs() -> None:
    """The apply() body should run without an active project (no LSP needed)."""
    from codescope import __version__
    from codescope.tools.info_tools import CodescopeInfoTool

    # apply() only reads module-level constants, so we can invoke it unbound.
    result = CodescopeInfoTool.apply(object.__new__(CodescopeInfoTool))  # type: ignore[arg-type]
    assert "Volantic Codescope" in result
    assert __version__ in result


def test_codescope_cli_does_not_expose_local_memory_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "codescope.cli", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "memories" not in result.stdout


def test_index_tool_bodies_execute(tmp_path) -> None:
    """Run the index tools' apply() bodies for real.

    Registration tests only prove a tool exists; they do not execute a single
    line of its body, so a name that is only resolved at call time (a dropped
    import, say) stays invisible until an agent hits it.
    """
    import json

    from codescope.tools.index_tools import IndexStatusTool, ReindexTool

    (tmp_path / "app.py").write_text("def greet(name):\n    return name.upper()\n")

    def _stub(cls):
        tool = object.__new__(cls)
        tool.get_project_root = lambda: str(tmp_path)  # type: ignore[method-assign]
        tool._to_json = lambda payload: json.dumps(payload, default=str)  # type: ignore[method-assign]
        return tool

    report = json.loads(ReindexTool.apply(_stub(ReindexTool), embeddings=False))
    assert report["indexed"] == 1
    assert report["totals"]["symbols"] >= 1

    health = json.loads(IndexStatusTool.apply(_stub(IndexStatusTool)))
    assert health["indexed"] is True
    assert health["symbols"] >= 1
    assert "advice" in health
