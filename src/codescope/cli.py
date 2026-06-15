"""Command-line entrypoint for Volantic Codescope.

This is a thin wrapper around Serena's CLI. Its job is to make sure the
Codescope tool package is registered with Serena's :class:`ToolRegistry`
*before* the registry singleton is first instantiated, and then to delegate to
Serena's existing command group (so ``codescope start-mcp-server`` and every
other Serena subcommand work unchanged).

We deliberately register Codescope at runtime rather than editing any file
under ``serena/`` or ``solidlsp/`` - this keeps the fork mergeable with
upstream Serena.
"""

from __future__ import annotations


def register_codescope_tools() -> None:
    """Make Codescope's tools discoverable by Serena's ToolRegistry.

    Serena's ``ToolRegistry`` only registers ``Tool`` subclasses whose module
    starts with one of the package prefixes in
    ``serena.tools.tools_base.tool_packages``. We append ``"codescope.tools"``
    to that list and import the package so the subclasses exist by the time the
    registry (a singleton) is constructed.
    """
    from serena.tools import tools_base

    if "codescope.tools" not in tools_base.tool_packages:
        tools_base.tool_packages.append("codescope.tools")

    # Importing the package forces all Codescope Tool subclasses to be defined,
    # which is required for ToolRegistry's subclass discovery to find them.
    import codescope.tools  # noqa: F401


def main() -> None:
    """Entrypoint for the ``codescope`` console script."""
    register_codescope_tools()

    from serena.cli import top_level

    top_level()


if __name__ == "__main__":
    main()
