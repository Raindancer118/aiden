"""Code-graph tools: dependencies, dependents, call chains, change impact."""

from __future__ import annotations

from dataclasses import asdict

from serena.tools import Tool, ToolMarkerSymbolicRead

from codescope.index.graph import GraphEngine, GraphNode


def _node_to_dict(node: GraphNode) -> dict:
    return {
        "name": node.name,
        "kind": node.kind,
        "path": node.path,
        "line": node.line,
        "children": [_node_to_dict(c) for c in node.children],
    }


class GetDependenciesTool(Tool, ToolMarkerSymbolicRead):
    """
    List the symbols that a given symbol depends on (its direct callees /
    referenced definitions), derived from the Codescope index. Fast and
    language-agnostic; for precise type-aware resolution prefer the LSP-based
    find_references / find_symbol tools.
    """

    def apply(self, symbol: str) -> str:
        """
        :param symbol: the name of the symbol (function/class/method/...).
        :return: JSON list of symbols referenced from within ``symbol``'s body.
        """
        deps = GraphEngine(self.get_project_root()).dependencies(symbol)
        return self._to_json([asdict(d) for d in deps])


class GetDependentsTool(Tool, ToolMarkerSymbolicRead):
    """
    List the symbols that depend on a given symbol (its callers / users):
    every defined symbol whose body references ``symbol``.
    """

    def apply(self, symbol: str) -> str:
        """
        :param symbol: the name of the symbol to find users of.
        :return: JSON list of symbols that reference ``symbol``.
        """
        deps = GraphEngine(self.get_project_root()).dependents(symbol)
        return self._to_json([asdict(d) for d in deps])


class GetCallChainTool(Tool, ToolMarkerSymbolicRead):
    """
    Return the outgoing call tree from a symbol (what it depends on,
    transitively) up to a given depth. Useful for understanding how a function
    is implemented across the codebase.
    """

    def apply(self, symbol: str, depth: int = 3) -> str:
        """
        :param symbol: the root symbol name.
        :param depth: maximum traversal depth (1-6 is typical).
        :return: JSON tree of {name, kind, path, line, children}.
        """
        tree = GraphEngine(self.get_project_root()).call_chain(symbol, depth=depth)
        return self._to_json(_node_to_dict(tree))


class GetChangeImpactTool(Tool, ToolMarkerSymbolicRead):
    """
    Estimate the blast radius of changing a symbol: the reverse call tree of
    everything that (transitively) depends on it, up to a given depth. Use this
    before refactoring or renaming to see what might be affected.
    """

    def apply(self, symbol: str, depth: int = 3) -> str:
        """
        :param symbol: the symbol you intend to change.
        :param depth: maximum traversal depth of the dependents tree.
        :return: JSON tree of affected symbols {name, kind, path, line, children}.
        """
        tree = GraphEngine(self.get_project_root()).change_impact(symbol, depth=depth)
        return self._to_json(_node_to_dict(tree))


class GetProjectMapTool(Tool, ToolMarkerSymbolicRead):
    """
    Return a high-level structural map of the project: per-file language,
    symbol count, and the top-level symbols in each file. A fast way to orient
    yourself in an unfamiliar codebase. Requires the index to be built.
    """

    def apply(self, max_files: int = 200) -> str:
        """
        :param max_files: maximum number of files to include (busiest first).
        :return: JSON list of {path, language, symbol_count, symbols[]}.
        """
        return self._to_json(GraphEngine(self.get_project_root()).project_map(max_files=max_files))


class GetFileSummaryTool(Tool, ToolMarkerSymbolicRead):
    """
    Summarize a single file: the symbols it defines (with kinds and line
    numbers) and the indexed symbols it depends on. Requires the index.
    """

    def apply(self, path: str) -> str:
        """
        :param path: project-relative path of the file (e.g. ``src/app/main.py``).
        :return: JSON {path, language, symbols[], depends_on[]} or an error note.
        """
        summary = GraphEngine(self.get_project_root()).file_summary(path)
        if summary is None:
            return self._to_json({"error": f"File not indexed: {path}"})
        return self._to_json(asdict(summary))
