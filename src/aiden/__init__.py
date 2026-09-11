"""AIDEN.

An all-in-one codebase-intelligence MCP server. Built on top of Serena
(LSP-accurate symbol navigation & structural editing) and extended with a
persistent hybrid index (tree-sitter + SQLite FTS5/BM25 + vectors + trigram),
code graphs, incremental reindexing, and dev-ops tooling (test runner,
project scaffolding, GitHub).

The Serena internals (``serena`` and ``solidlsp`` packages) are kept
unmodified so upstream updates remain mergeable; all AIDEN-specific value
lives in this ``aiden`` package and is registered into Serena's existing
tool registry.
"""


def _installed_version() -> str:
    """Read the version from package metadata rather than repeating it here.

    A second copy drifts: this said 0.1.0 while the project was on 0.3.x, so
    ``aiden_info`` reported a version that had not existed for a while.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("aiden")
    except PackageNotFoundError:  # pragma: no cover - running from a source tree
        return "0+unknown"


__version__ = _installed_version()

PRODUCT_NAME = "AIDEN"
PRODUCT_TAGLINE = "Agent Intelligence for Development, Exploration & Navigation"
