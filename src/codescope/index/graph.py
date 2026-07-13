"""Code-structure graph queries built on the symbol/reference index.

Edges are derived on demand from the ``symbols`` and ``refs`` tables:

  * a *reference* at ``(path, line)`` is attributed to the innermost symbol
    whose line range contains it (the "caller");
  * the reference name is resolved to defined symbol(s) of the same name
    (the "callee").

This is fast, language-agnostic, and works immediately after indexing - but
name-based resolution is approximate (it does not disambiguate overloads or
scopes). For precise, type-aware resolution use Serena's LSP tools
(find_references / find_symbol); use these graph tools for fast structural
overviews and blast-radius estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from codescope.index.indexer import default_db_path
from codescope.index.store import IndexStore


@dataclass(slots=True)
class SymbolRef:
    name: str
    kind: str
    path: str
    start_line: int
    end_line: int = 0


@dataclass(slots=True)
class FileSummary:
    path: str
    language: str
    symbols: list[SymbolRef]
    depends_on: list[str]


@dataclass(slots=True)
class GraphNode:
    name: str
    kind: str
    path: str
    line: int
    children: list["GraphNode"] = field(default_factory=list)


class GraphEngine:
    def __init__(self, project_root: str | Path, db_path: str | Path | None = None):
        self.root = Path(project_root).resolve()
        self.db_path = Path(db_path) if db_path else default_db_path(self.root)

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _containing_symbol(store: IndexStore, path: str, line: int) -> tuple[str, str, str, int] | None:
        row = store.conn.execute(
            "SELECT name, kind, path, start_line FROM symbols "
            "WHERE path=? AND start_line<=? AND end_line>=? "
            "ORDER BY (end_line - start_line) ASC LIMIT 1",
            (path, line, line),
        ).fetchone()
        return tuple(row) if row else None  # type: ignore[return-value]

    @staticmethod
    def _defined_names(store: IndexStore) -> set[str]:
        return {r[0] for r in store.conn.execute("SELECT DISTINCT name FROM symbols")}

    def _dependents(self, store: IndexStore, name: str) -> list[SymbolRef]:
        """Symbols that reference ``name`` (callers / users)."""
        rows = store.conn.execute("SELECT path, line FROM refs WHERE name=?", (name,)).fetchall()
        seen: set[tuple[str, str, str]] = set()
        out: list[SymbolRef] = []
        for path, line in rows:
            caller = self._containing_symbol(store, path, line)
            if caller is None:
                continue
            cname, ckind, cpath, cline = caller
            if cname == name:  # ignore self-references inside the symbol's own body
                continue
            key = (cname, ckind, cpath)
            if key in seen:
                continue
            seen.add(key)
            out.append(SymbolRef(name=cname, kind=ckind, path=cpath, start_line=cline))
        return out

    def _dependencies(
        self,
        store: IndexStore,
        name: str,
        defined: set[str],
        *,
        path: str | None = None,
        start_line: int | None = None,
    ) -> list[SymbolRef]:
        """Defined symbols referenced from within ``name``'s body (callees)."""
        if path is None or start_line is None:
            defs = store.conn.execute(
                "SELECT path, start_line, end_line FROM symbols WHERE name=? ORDER BY path, start_line", (name,)
            ).fetchall()
        else:
            defs = store.conn.execute(
                "SELECT path, start_line, end_line FROM symbols WHERE name=? AND path=? AND start_line=?",
                (name, path, start_line),
            ).fetchall()
        out: list[SymbolRef] = []
        seen: set[str] = set()
        for path, start, end in defs:
            refs = store.conn.execute(
                "SELECT DISTINCT name FROM refs WHERE path=? AND line BETWEEN ? AND ?",
                (path, start, end),
            ).fetchall()
            for (refname,) in refs:
                if refname == name or refname not in defined or refname in seen:
                    continue
                seen.add(refname)
                target = store.conn.execute(
                    "SELECT name, kind, path, start_line, end_line FROM symbols WHERE name=? "
                    "ORDER BY CASE WHEN path=? THEN 0 ELSE 1 END, path, start_line LIMIT 1",
                    (refname, path),
                ).fetchone()
                if target:
                    out.append(SymbolRef(name=target[0], kind=target[1], path=target[2], start_line=target[3], end_line=target[4]))
        return out

    # -- public API -------------------------------------------------------

    def dependents(self, name: str) -> list[SymbolRef]:
        with IndexStore(self.db_path) as store:
            return self._dependents(store, name)

    def dependencies(self, name: str) -> list[SymbolRef]:
        with IndexStore(self.db_path) as store:
            return self._dependencies(store, name, self._defined_names(store))

    def call_chain(self, name: str, depth: int = 3) -> GraphNode:
        """Outgoing call tree from ``name`` (what it depends on), to ``depth``."""
        with IndexStore(self.db_path) as store:
            defined = self._defined_names(store)
            root_sym = store.conn.execute(
                "SELECT name, kind, path, start_line FROM symbols WHERE name=? ORDER BY path, start_line LIMIT 1", (name,)
            ).fetchone()
            root = GraphNode(
                name=name, kind=root_sym[1] if root_sym else "?", path=root_sym[2] if root_sym else "", line=root_sym[3] if root_sym else 0
            )
            self._expand(store, root, depth, defined, set(), forward=True)
            return root

    def change_impact(self, name: str, depth: int = 3) -> GraphNode:
        """Reverse call tree: who is (transitively) affected if ``name`` changes."""
        with IndexStore(self.db_path) as store:
            root_sym = store.conn.execute(
                "SELECT name, kind, path, start_line FROM symbols WHERE name=? ORDER BY path, start_line LIMIT 1", (name,)
            ).fetchone()
            root = GraphNode(
                name=name, kind=root_sym[1] if root_sym else "?", path=root_sym[2] if root_sym else "", line=root_sym[3] if root_sym else 0
            )
            self._expand(store, root, depth, set(), set(), forward=False)
            return root

    def _expand(
        self,
        store: IndexStore,
        node: GraphNode,
        depth: int,
        defined: set[str],
        visited: set[tuple[str, str, int]],
        *,
        forward: bool,
    ) -> None:
        node_key = (node.name, node.path, node.line)
        if depth <= 0 or node_key in visited:
            return
        branch_visited = visited | {node_key}
        neighbors = (
            self._dependencies(store, node.name, defined, path=node.path or None, start_line=node.line)
            if forward
            else self._dependents(store, node.name)
        )
        for n in neighbors:
            child = GraphNode(name=n.name, kind=n.kind, path=n.path, line=n.start_line)
            node.children.append(child)
            self._expand(store, child, depth - 1, defined, branch_visited, forward=forward)

    def file_summary(self, rel_path: str) -> FileSummary | None:
        with IndexStore(self.db_path) as store:
            frow = store.conn.execute("SELECT lang FROM files WHERE path=?", (rel_path,)).fetchone()
            if frow is None:
                return None
            syms = [
                SymbolRef(name=r[0], kind=r[1], path=rel_path, start_line=r[2], end_line=r[3])
                for r in store.conn.execute(
                    "SELECT name, kind, start_line, end_line FROM symbols WHERE path=? ORDER BY start_line", (rel_path,)
                )
            ]
            defined = self._defined_names(store)
            refs = {r[0] for r in store.conn.execute("SELECT DISTINCT name FROM refs WHERE path=?", (rel_path,)) if r[0] in defined}
            own = {s.name for s in syms}
            depends_on = sorted(refs - own)
            return FileSummary(path=rel_path, language=frow[0], symbols=syms, depends_on=depends_on)

    def project_map(self, max_files: int = 200) -> list[dict]:
        """High-level per-file overview: language, symbol count, top symbols."""
        with IndexStore(self.db_path) as store:
            files = store.conn.execute(
                "SELECT f.path, f.lang, COUNT(s.id) AS n FROM files f "
                "LEFT JOIN symbols s ON s.path=f.path GROUP BY f.path ORDER BY n DESC LIMIT ?",
                (max_files,),
            ).fetchall()
            out = []
            for path, lang, n in files:
                tops = [
                    {"name": r[0], "kind": r[1], "line": r[2]}
                    for r in store.conn.execute(
                        "SELECT name, kind, start_line FROM symbols WHERE path=? "
                        "AND kind IN ('class','interface','function','method','type','enum','module','struct') "
                        "ORDER BY start_line LIMIT 12",
                        (path,),
                    )
                ]
                out.append({"path": path, "language": lang, "symbol_count": n, "symbols": tops})
            return out
