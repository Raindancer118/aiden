"""Command-line entrypoint for Volantic Codescope.

This wrapper registers Codescope's tools before Serena's registry singleton is
created. It also removes Serena's local Markdown memory workflow: Codescope
uses Diary through MCP as its only memory backend.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

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


def _explorer_enabled() -> bool:
    """Whether to bring the graph explorer up with the MCP server."""
    return os.environ.get("CODESCOPE_EXPLORER", "1").strip().lower() not in ("0", "false", "no", "off")


def start_explorer_for(project_root: str) -> None:
    """Attach ``project_root`` to the shared explorer, starting it if needed.

    Failure here must never take the MCP server down with it: the explorer is
    a convenience, and the server has to keep serving tools without it.
    """
    if not _explorer_enabled():
        return
    try:
        from codescope.web.explorer import ensure_explorer

        ensure_explorer(project_root)
    except Exception as e:  # pragma: no cover - defensive
        log.warning("Could not start the Codescope explorer: %s", e)


def _hook_project_activation() -> None:
    """Register each activated project with the explorer.

    Serena has no post-activation callback, so the tool's ``apply`` is
    wrapped. Kept to this one small patch, in the same spirit as the tool
    exclusions above.
    """
    from serena.tools.config_tools import ActivateProjectTool

    if getattr(ActivateProjectTool, "_codescope_explorer_hooked", False):
        return
    original = ActivateProjectTool.apply

    def apply(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = original(self, *args, **kwargs)
        try:
            start_explorer_for(self.get_project_root())
        except Exception as e:  # pragma: no cover - defensive
            log.warning("Explorer registration failed after project activation: %s", e)
        return result

    ActivateProjectTool.apply = apply  # type: ignore[method-assign]
    ActivateProjectTool._codescope_explorer_hooked = True  # type: ignore[attr-defined]


def _hook_explorer_notices() -> None:
    """Deliver explorer notices with the next tool result.

    Wrapping every Codescope tool once here beats threading a notice check
    through each of them, and keeps the whole mechanism in one reviewable
    place next to the other hooks.
    """
    import codescope.tools  # noqa: F401
    from codescope.web.notices import append_notice
    from serena.tools import Tool

    def wrap(tool_cls: type) -> None:
        # Abstract bases in the package define no apply(); only real tools do.
        if getattr(tool_cls, "_codescope_notice_hooked", False) or not hasattr(tool_cls, "apply"):
            return
        original = tool_cls.apply

        def apply(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            result = original(self, *args, **kwargs)
            return append_notice(result) if isinstance(result, str) else result

        tool_cls.apply = apply  # type: ignore[method-assign]
        tool_cls._codescope_notice_hooked = True  # type: ignore[attr-defined]

    def walk(cls: type) -> None:
        for subclass in cls.__subclasses__():
            if subclass.__module__.startswith("codescope.tools"):
                wrap(subclass)
            walk(subclass)

    walk(Tool)


def main() -> None:
    """Entrypoint for the ``codescope`` console script."""
    register_codescope_tools()
    if _explorer_enabled():
        _hook_project_activation()
        _hook_explorer_notices()

    from codescope.index_cli import index_group
    from serena.cli import top_level

    # Codescope has no file-backed memory CLI. Diary is the sole backend.
    top_level.commands.pop("memories", None)
    # Indexing is the one operation worth running outside the MCP server, under
    # a resource budget (see codescope.index_cli).
    top_level.add_command(index_group)
    top_level()


if __name__ == "__main__":
    main()
