"""Type and call hierarchy - the IDE views the index cannot approximate.

``get_dependencies`` / ``get_call_chain`` resolve edges by identifier name,
which is fast and works for every indexed language but cannot tell two
``save`` methods apart or see inheritance at all. These two tools ask the
language server instead, so the answer is type-aware: the same information an
IDE shows in its Type Hierarchy and Call Hierarchy panels.

Use these when correctness matters (before a refactor, when reasoning about an
override, when the name-based graph reported a ``speculative`` edge); use the
index graph when you want a fast structural overview of a whole subtree.
"""

from __future__ import annotations

from dataclasses import asdict

from codescope.index.graph import GraphEngine
from serena.tools import Tool, ToolMarkerSymbolicRead


class _HierarchyToolBase(Tool):
    """Shared name_path -> (file, line, column) resolution."""

    def _locate(self, name_path: str, relative_path: str) -> tuple[str, int, int]:
        retriever = self.create_language_server_symbol_retriever()
        symbol = retriever.find_unique(name_path, within_relative_path=relative_path)
        if symbol.relative_path is None or symbol.line is None or symbol.column is None:
            raise ValueError(f"Symbol '{name_path}' has no source location the language server can query.")
        return symbol.relative_path, symbol.line, symbol.column

    def _language_server(self, relative_path: str):  # type: ignore[no-untyped-def]
        return self.project.get_language_server_manager_or_raise().get_language_server(relative_path)


class GetTypeHierarchyTool(_HierarchyToolBase, ToolMarkerSymbolicRead):
    """
    Show what a type extends and what extends it (type-aware, via the language server).

    The IDE's Type Hierarchy view: supertypes (base classes / implemented
    interfaces) and subtypes (implementations and subclasses) of a class,
    interface or trait. The index's name-based graph cannot see inheritance at
    all, so this is the tool to use before changing a base class, adding an
    abstract method, or deciding where an override belongs.

    Returns empty lists when the project's language server does not implement
    type hierarchy (a protocol capability, not an error); find_implementations
    is the partial fallback in that case.
    """

    def apply(self, name_path: str, relative_path: str, direction: str = "both") -> str:
        """
        Resolve the type hierarchy around a symbol.

        :param name_path: name path of the class/interface (as used by find_symbol).
        :param relative_path: file containing that symbol.
        :param direction: "supertypes", "subtypes", or "both".
        :return: JSON with the resolved item and its supertypes/subtypes, each
            with name, kind, relativePath and line.
        """
        if direction not in ("supertypes", "subtypes", "both"):
            raise ValueError("direction must be 'supertypes', 'subtypes' or 'both'")
        rel, line, column = self._locate(name_path, relative_path)
        result = self._language_server(rel).request_type_hierarchy(rel, line, column, direction=direction)
        if result.get("supported") and result.get("item") is not None:
            result["resolved_by"] = "lsp"
            return self._to_json(result)

        # Several widely used servers (pyright among them) implement call
        # hierarchy but not type hierarchy. Rather than answering "not
        # supported", fall back to the index, which reads base types out of
        # the stored declaration lines.
        fallback = asdict(GraphEngine(self.get_project_root()).type_hierarchy(name_path.rsplit("/", 1)[-1]))
        if direction == "supertypes":
            fallback["subtypes"] = []
        elif direction == "subtypes":
            fallback["supertypes"] = []
        fallback["note"] = (
            "The language server for this file does not implement type hierarchy; "
            "these edges were read from indexed declarations and are not type-resolved. "
            "Confirm important ones with find_implementations."
        ).strip()
        return self._to_json(fallback)


class GetCallHierarchyTool(_HierarchyToolBase, ToolMarkerSymbolicRead):
    """
    Show the resolved callers or callees of a function (type-aware, via the language server).

    The IDE's Call Hierarchy view. Unlike get_call_chain, which matches bare
    identifier names and therefore conflates same-named methods on different
    types, this is resolved by the language server: overloads, methods on
    different classes, and shadowed names are kept apart. Use it to answer
    "who actually calls this?" before changing a signature.

    Returns an empty list when the language server does not implement call
    hierarchy; find_referencing_symbols is the fallback.
    """

    def apply(self, name_path: str, relative_path: str, direction: str = "incoming") -> str:
        """
        Resolve one level of the call hierarchy around a symbol.

        :param name_path: name path of the function/method.
        :param relative_path: file containing that symbol.
        :param direction: "incoming" for callers, "outgoing" for callees.
        :return: JSON list of related symbols, each with name, kind,
            relativePath, line, and the lines of the individual call sites.
        """
        if direction not in ("incoming", "outgoing"):
            raise ValueError("direction must be 'incoming' or 'outgoing'")
        rel, line, column = self._locate(name_path, relative_path)
        ls = self._language_server(rel)
        entries = ls.request_call_hierarchy(rel, line, column, direction=direction)
        return self._to_json(
            {"direction": direction, "of": {"name_path": name_path, "relative_path": rel, "line": line + 1}, "calls": entries}
        )
