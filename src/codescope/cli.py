"""Command-line entrypoint for Volantic Codescope.

This wrapper registers Codescope's tools before Serena's registry singleton is
created. It also removes Serena's local Markdown memory workflow: Codescope
uses Diary through MCP as its only memory backend.
"""

from __future__ import annotations

LOCAL_MEMORY_TOOLS = frozenset(
    {
        "delete_memory",
        "edit_memory",
        "list_memories",
        "onboarding",
        "read_memory",
        "rename_memory",
        "write_memory",
    }
)

DIARY_MEMORY_TOOLS = frozenset(
    {
        "memory_context",
        "memory_delete",
        "memory_get",
        "memory_project_context",
        "memory_search",
        "memory_search_semantic",
        "memory_tree",
        "memory_upsert",
    }
)

LEGACY_MEMORY_TOOL_EXCLUSION_ALIASES = {
    "delete_memory": {"memory_delete"},
    "edit_memory": {"memory_upsert"},
    "list_memories": {"memory_tree"},
    "read_memory": {
        "memory_context",
        "memory_get",
        "memory_project_context",
        "memory_search",
        "memory_search_semantic",
    },
    "write_memory": {"memory_upsert"},
}


def register_codescope_tools() -> None:
    """Make Codescope's tools discoverable by Serena's ToolRegistry.

    Serena's ``ToolRegistry`` only registers ``Tool`` subclasses whose module
    starts with one of the package prefixes in
    ``serena.tools.tools_base.tool_packages``. We append ``"codescope.tools"``
    to that list and import the package so the subclasses exist by the time the
    registry (a singleton) is constructed.
    """
    from serena.tools import tools_base

    tools_base.tool_names_excluded_from_registry.update(LOCAL_MEMORY_TOOLS)
    for legacy_name, diary_names in LEGACY_MEMORY_TOOL_EXCLUSION_ALIASES.items():
        tools_base.tool_exclusion_aliases.setdefault(legacy_name, set()).update(diary_names)
    if "codescope.tools" not in tools_base.tool_packages:
        tools_base.tool_packages.append("codescope.tools")

    # Importing the package forces all Codescope Tool subclasses to be defined,
    # which is required for ToolRegistry's subclass discovery to find them.
    import codescope.tools  # noqa: F401


def main() -> None:
    """Entrypoint for the ``codescope`` console script."""
    register_codescope_tools()

    from serena.cli import top_level

    # Codescope has no file-backed memory CLI. Diary is the sole backend.
    top_level.commands.pop("memories", None)
    top_level()


if __name__ == "__main__":
    main()
