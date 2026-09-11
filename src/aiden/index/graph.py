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

import re
from dataclasses import dataclass, field
from pathlib import Path

from aiden.index.indexer import default_db_path
from aiden.index.store import IndexStore


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


#: Symbol kinds that can participate in an inheritance relationship.
TYPE_KINDS = ("class", "interface", "struct", "trait", "enum", "type", "object", "protocol", "record")

#: Declaration noise to drop when reading base types out of a signature.
_DECL_KEYWORDS = frozenset(
    {
        "public",
        "private",
        "protected",
        "internal",
        "abstract",
        "final",
        "static",
        "sealed",
        "open",
        "export",
        "default",
        "class",
        "struct",
        "interface",
        "enum",
        "trait",
        "object",
        "data",
        "record",
        "type",
        "where",
        "implements",
        "extends",
        "case",
        "impl",
        "for",
        "with",
        "partial",
        "const",
        "var",
        "val",
        "func",
        "fn",
    }
)
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


def base_type_names(name: str, signature: str) -> list[str]:
    """Read the base types out of a declaration line.

    Deliberately syntactic and language-agnostic: it handles ``class X(A)``,
    ``class X extends A implements B``, ``class X : A, B`` and
    ``struct X : public Base`` alike. Callers must intersect the result with
    the types actually defined in the project, which is what removes generic
    parameters and standard-library names.
    """
    if not signature:
        return []
    tail = signature.split(name, 1)[1] if name in signature else signature
    tail = tail.split("{")[0]
    tail = tail.lstrip(" \t:(")
    out: list[str] = []
    for match in _IDENTIFIER_RE.finditer(tail):
        token = match.group(0)
        if token in _DECL_KEYWORDS or token == name:
            continue
        out.append(token.rsplit(".", 1)[-1])
    return list(dict.fromkeys(out))


@dataclass(slots=True)
class TypeHierarchy:
    """Inheritance around one type, as read from the index."""

    item: SymbolRef | None
    supertypes: list[SymbolRef] = field(default_factory=list)
    subtypes: list[SymbolRef] = field(default_factory=list)
    #: "index" here; the LSP-backed tool reports "lsp" when the server answers.
    resolved_by: str = "index"
    note: str = ""


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
        """Symbols that reference ``name`` (callers / users).

        One join over the ownership recorded at index time, rather than a
        range lookup per reference: a widely used symbol had thousands of
        references and therefore thousands of round trips.
        """
        rows = store.conn.execute(
            "SELECT DISTINCT s.name, s.kind, s.path, s.start_line, s.end_line "
            "FROM refs AS r JOIN symbols AS s ON s.id = r.owner_id "
            "WHERE r.name = ? AND s.name != ? "
            "ORDER BY s.path, s.start_line",
            (name, name),  # a symbol referencing itself is not its own caller
        ).fetchall()
        return [SymbolRef(name=r[0], kind=r[1], path=r[2], start_line=r[3], end_line=r[4]) for r in rows]

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
            # Scope by ownership where it is known, so a reference nested in
            # an inner function no longer counts as the outer one's callee.
            refs = store.conn.execute(
                "SELECT DISTINCT r.name FROM refs AS r "
                "LEFT JOIN symbols AS owner ON owner.id = r.owner_id "
                "WHERE r.path = ? AND r.line BETWEEN ? AND ? "
                "  AND (r.owner_id IS NULL OR (owner.start_line = ? AND owner.path = ?))",
                (path, start, end, start, path),
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

    def dependents_in(self, store: IndexStore, name: str) -> list[SymbolRef]:
        """Callers of ``name``, reusing an open store (batched callers)."""
        return self._dependents(store, name)

    def dependencies_in(self, store: IndexStore, name: str, *, path: str | None = None, start_line: int | None = None) -> list[SymbolRef]:
        """Callees of one specific definition of ``name``, reusing an open store."""
        return self._dependencies(store, name, self._defined_names(store), path=path, start_line=start_line)

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

    def type_hierarchy(self, name: str) -> TypeHierarchy:
        """Supertypes and subtypes of ``name``, derived from declaration lines.

        The fallback for language servers that do not implement
        ``textDocument/typeHierarchy`` (pyright, among others). It reads base
        types out of the stored signature and keeps only the names that are
        themselves defined in the project, so generic parameters and
        third-party bases drop out. Same-named types are not disambiguated -
        this is a structural answer, not a resolved one.
        """
        placeholders = ",".join("?" * len(TYPE_KINDS))
        with IndexStore(self.db_path) as store:
            rows = store.conn.execute(
                f"SELECT name, kind, path, start_line, end_line, COALESCE(signature, '') FROM symbols WHERE kind IN ({placeholders})",
                TYPE_KINDS,
            ).fetchall()
            defined = {r[0] for r in rows}

        item: SymbolRef | None = None
        supertypes: list[SymbolRef] = []
        subtypes: list[SymbolRef] = []
        by_name = {r[0]: r for r in rows}
        for row in rows:
            rname, kind, path, start, end, signature = row
            bases = base_type_names(rname, signature)
            if rname == name:
                item = item or SymbolRef(name=rname, kind=kind, path=path, start_line=start, end_line=end)
                for base in bases:
                    if base in defined and base != name:
                        b = by_name[base]
                        supertypes.append(SymbolRef(name=b[0], kind=b[1], path=b[2], start_line=b[3], end_line=b[4]))
            elif name in bases:
                subtypes.append(SymbolRef(name=rname, kind=kind, path=path, start_line=start, end_line=end))

        note = "" if item else f"No indexed type named {name!r}. Is the index current?"
        return TypeHierarchy(
            item=item,
            supertypes=sorted(supertypes, key=lambda s: (s.path, s.start_line)),
            subtypes=sorted(subtypes, key=lambda s: (s.path, s.start_line)),
            resolved_by="index",
            note=note,
        )

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
