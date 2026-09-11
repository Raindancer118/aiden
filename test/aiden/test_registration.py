"""Start-up / smoke tests for the AIDEN tool layer.

These verify that AIDEN is importable and that its tools register cleanly
into Serena's ToolRegistry via the runtime hook in ``aiden.cli`` - i.e.
that the fork's core integration mechanism works without editing any Serena
source file.
"""

import subprocess
import sys


def test_aiden_importable() -> None:
    import aiden

    assert aiden.__version__
    assert aiden.PRODUCT_NAME == "AIDEN"


def test_info_tool_name() -> None:
    from aiden.tools.info_tools import AidenInfoTool

    # Serena derives the MCP tool name from the class name minus "Tool", snake_cased.
    assert AidenInfoTool.get_name_from_cls() == "aiden_info"


def test_registration_appends_tool_package() -> None:
    from aiden.cli import LOCAL_MEMORY_TOOLS, register_aiden_tools
    from serena.tools import tools_base

    register_aiden_tools()
    assert "aiden.tools" in tools_base.tool_packages
    assert tools_base.tool_names_excluded_from_registry >= LOCAL_MEMORY_TOOLS


def test_info_tool_registered_in_registry() -> None:
    from aiden.cli import register_aiden_tools

    register_aiden_tools()

    from serena.tools import ToolRegistry

    registry = ToolRegistry()
    tool_names = registry.get_tool_names()
    assert "aiden_info" in tool_names, f"aiden_info not registered; available tools: {sorted(tool_names)}"
    assert "memory_project_context" in tool_names
    assert "memory_upsert" in tool_names
    assert "read_memory" not in tool_names
    assert "write_memory" not in tool_names
    assert "onboarding" not in tool_names


def test_info_tool_apply_runs() -> None:
    """The apply() body should run without an active project (no LSP needed)."""
    from aiden import __version__
    from aiden.tools.info_tools import AidenInfoTool

    # apply() only reads module-level constants, so we can invoke it unbound.
    result = AidenInfoTool.apply(object.__new__(AidenInfoTool))  # type: ignore[arg-type]
    assert "AIDEN" in result
    assert __version__ in result


def test_aiden_cli_does_not_expose_local_memory_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "aiden.cli", "--help"],
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

    from aiden.tools.index_tools import IndexStatusTool, ReindexTool

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


def test_hooked_tools_still_expose_their_mcp_schema() -> None:
    """The explorer hooks wrap apply(); the MCP schema is built from it.

    Serena derives every tool's description and parameter schema from the
    apply method's docstring and signature. A wrapper that does not carry
    those through does not degrade the schema -- it aborts server startup,
    which is how this was found: the MCP server refused to boot at all.
    """
    from aiden.cli import _hook_explorer_notices, _hook_project_activation, register_aiden_tools

    register_aiden_tools()
    _hook_project_activation()
    _hook_explorer_notices()

    from serena.tools import ToolRegistry

    registry = ToolRegistry()
    checked = 0
    for name in registry.get_tool_names():
        tool_cls = registry.get_tool_class_by_name(name)
        if not tool_cls.__module__.startswith(("aiden.tools", "serena.tools.config_tools")):
            continue
        assert tool_cls.get_apply_docstring_from_cls().strip(), f"{name} lost its docstring"
        parameters = tool_cls.get_apply_fn_metadata_from_cls().arg_model.model_json_schema()
        assert "properties" in parameters, f"{name} lost its parameter schema"
        checked += 1
    assert checked > 20, f"expected the aiden tools to be covered, checked only {checked}"


def test_mcp_server_boots_with_every_tool() -> None:
    """A start-up test: build the tool set the way the server does.

    This is the exact step that failed when the hooks stripped apply()'s
    metadata, so it is worth paying for on every run.
    """
    from aiden.cli import _hook_explorer_notices, _hook_project_activation, register_aiden_tools

    register_aiden_tools()
    _hook_project_activation()
    _hook_explorer_notices()

    from serena.mcp import SerenaMCPFactory
    from serena.tools import ToolRegistry

    class _Context:
        tool_description_overrides: dict[str, str] = {}

    class _Agent:
        @staticmethod
        def get_context() -> "_Context":
            return _Context()

    registry = ToolRegistry()
    built = 0
    for name in registry.get_tool_names():
        tool_cls = registry.get_tool_class_by_name(name)
        instance = object.__new__(tool_cls)
        instance.agent = _Agent()  # type: ignore[attr-defined]
        mcp_tool = SerenaMCPFactory.make_mcp_tool(instance, openai_tool_compatible=False)
        assert mcp_tool.description, f"{name} would be exposed without a description"
        built += 1
    assert built > 80, f"expected the full tool set, built only {built}"


def test_reported_version_matches_the_project() -> None:
    """A hardcoded second copy drifted from pyproject for three releases."""
    import tomllib
    from pathlib import Path

    import aiden

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    assert aiden.__version__ == declared
