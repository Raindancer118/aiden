"""Volantic Codescope.

An all-in-one codebase-intelligence MCP server. Built on top of Serena
(LSP-accurate symbol navigation & structural editing) and extended with a
persistent hybrid index (tree-sitter + SQLite FTS5/BM25 + vectors + trigram),
code graphs, incremental reindexing, and dev-ops tooling (test runner,
project scaffolding, GitHub).

The Serena internals (``serena`` and ``solidlsp`` packages) are kept
unmodified so upstream updates remain mergeable; all Codescope-specific value
lives in this ``codescope`` package and is registered into Serena's existing
tool registry.
"""

__version__ = "0.1.0"

PRODUCT_NAME = "Volantic Codescope"
