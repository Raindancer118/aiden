"""Codescope hybrid index engine.

A persistent, per-project index built with tree-sitter (symbol extraction) and
stored in a single SQLite file. It backs the search and graph tools that
Serena's LSP layer does not provide.

Modules:
    languages  - file-extension -> tree-sitter language resolution
    parser     - tree-sitter parsing + symbol/reference extraction
    store      - SQLite schema and persistence
    indexer    - orchestration: walk files, detect changes, (re)index
"""
