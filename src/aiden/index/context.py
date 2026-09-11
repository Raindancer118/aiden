"""One-call code context: everything an agent needs to change a piece of code.

Answering "how does X work, and what breaks if I change it?" used to take an
agent four or five round-trips: search, read the file, look up callers, look
up dependencies, hunt for a test. Each one costs latency and context, and the
results arrive unrelated to each other.

:func:`build_context` does that assembly server-side and returns a single
budgeted bundle per matched symbol: the code itself, who calls it, what it
calls, and the tests that exercise it -- with an explicit token budget so the
answer stays affordable.

Caller/callee edges come from the name-based graph, which is fast and
language-agnostic but cannot disambiguate same-named symbols. Edges whose name
is defined more than once in the project are therefore marked ``speculative``
rather than presented as fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from aiden.index.graph import GraphEngine
from aiden.index.indexer import default_db_path
from aiden.index.search import SearchEngine, SearchFilter, _fold
from aiden.index.store import IndexStore

#: Lines of the primary symbol's body included verbatim.
DEFAULT_BODY_LINES = 80

#: Related symbols (callers/callees/tests) listed per focus symbol.
DEFAULT_RELATED = 5

_TEST_HINTS = ("test", "spec")


@dataclass(slots=True)
class RelatedSymbol:
    name: str
    kind: str
    path: str
    start_line: int
    #: True when the name resolves to several definitions, so this edge is a
    #: name-based guess rather than a resolved reference.
    speculative: bool = False


@dataclass(slots=True)
class SymbolContext:
    symbol_id: int
    name: str
    kind: str
    path: str
    start_line: int
    end_line: int
    lang: str
    signature: str
    code: str
    truncated: bool
    called_by: list[RelatedSymbol] = field(default_factory=list)
    calls: list[RelatedSymbol] = field(default_factory=list)
    tests: list[RelatedSymbol] = field(default_factory=list)


@dataclass(slots=True)
class CodeContext:
    query: str
    symbols: list[SymbolContext]
    #: Populated when the index cannot answer fully (no vectors, no index).
    warnings: list[str] = field(default_factory=list)


def _is_test_path(path: str) -> bool:
    lowered = path.lower()
    return any(hint in lowered for hint in _TEST_HINTS)


class ContextEngine:
    """Assembles :class:`CodeContext` bundles from the index."""

    def __init__(self, project_root: str | Path, db_path: str | Path | None = None):
        self.root = Path(project_root).resolve()
        self.db_path = Path(db_path) if db_path else default_db_path(self.root)
        self.search = SearchEngine(self.root, db_path=self.db_path)
        self.graph = GraphEngine(self.root, db_path=self.db_path)

    def build(
        self,
        query: str,
        *,
        max_symbols: int = 3,
        body_lines: int = DEFAULT_BODY_LINES,
        related: int = DEFAULT_RELATED,
        include_tests: bool = True,
        path_glob: str | None = None,
        lang: str | None = None,
    ) -> CodeContext:
        """Find the code ``query`` refers to and everything attached to it.

        :param query: natural language, an identifier, or both.
        :param max_symbols: how many matched symbols to expand. Each one costs
            roughly ``body_lines`` lines plus its related-symbol lists.
        :param body_lines: lines of each symbol's body to include verbatim;
            longer bodies are folded around the middle.
        :param related: max callers, callees and tests listed per symbol.
        :param include_tests: list the tests that reference each symbol.
        :param path_glob: restrict the search to matching paths.
        :param lang: restrict the search to one language.
        """
        warnings: list[str] = []
        if max_symbols <= 0 or not query.strip():
            return CodeContext(query=query, symbols=[], warnings=warnings)

        flt = SearchFilter(path_glob=path_glob, lang=lang, exclude_tests=True)
        hits = self.search.hybrid_search(query, limit=max_symbols, flt=flt, preview_lines=0)
        if not hits:
            # Tests may be the only place a name appears; do not hide that.
            hits = self.search.hybrid_search(query, limit=max_symbols, flt=SearchFilter(path_glob=path_glob, lang=lang), preview_lines=0)
        if not hits:
            warnings.append("No indexed symbol matched the query. Is the index built and current (reindex / sync_index)?")
            return CodeContext(query=query, symbols=[], warnings=warnings)

        with IndexStore(self.db_path) as store:
            if not store.has_vectors():
                warnings.append("No embeddings indexed: matching was lexical only.")
            ambiguous = self._ambiguous_names(store)
            bodies = self._bodies(store, [h.symbol_id for h in hits])
            contexts = []
            for hit in hits:
                body = bodies.get(hit.symbol_id, "")
                folded = _fold(body, body_lines)
                # _fold rejoins lines, so a body with a trailing newline is
                # never byte-identical; compare what was actually dropped.
                truncated = len(body.splitlines()) > body_lines > 0
                referencing = self.graph.dependents_in(store, hit.name)
                callees = self.graph.dependencies_in(store, hit.name, path=hit.path, start_line=hit.start_line)
                callers = self._related(referencing, ambiguous, related, tests=False)
                tests = self._related(referencing, ambiguous, related, tests=True) if include_tests else []
                contexts.append(
                    SymbolContext(
                        symbol_id=hit.symbol_id,
                        name=hit.name,
                        kind=hit.kind,
                        path=hit.path,
                        start_line=hit.start_line,
                        end_line=hit.end_line,
                        lang=hit.lang,
                        signature=hit.signature,
                        code=folded,
                        truncated=truncated,
                        called_by=callers,
                        calls=self._related(callees, ambiguous, related, tests=False),
                        tests=tests,
                    )
                )
        return CodeContext(query=query, symbols=contexts, warnings=warnings)

    @staticmethod
    def _ambiguous_names(store: IndexStore) -> set[str]:
        """Names with more than one definition -- edges to them are guesses."""
        return {r[0] for r in store.conn.execute("SELECT name FROM symbols GROUP BY name HAVING COUNT(*) > 1")}

    @staticmethod
    def _bodies(store: IndexStore, sids: list[int]) -> dict[int, str]:
        if not sids:
            return {}
        placeholders = ",".join("?" * len(sids))
        return dict(store.conn.execute(f"SELECT rowid, body FROM symbols_fts WHERE rowid IN ({placeholders})", sids))

    @staticmethod
    def _related(refs: list, ambiguous: set[str], limit: int, *, tests: bool) -> list[RelatedSymbol]:  # type: ignore[type-arg]
        out: list[RelatedSymbol] = []
        for ref in refs:
            if _is_test_path(ref.path) != tests:
                continue
            out.append(
                RelatedSymbol(
                    name=ref.name,
                    kind=ref.kind,
                    path=ref.path,
                    start_line=ref.start_line,
                    speculative=ref.name in ambiguous,
                )
            )
            if len(out) >= limit:
                break
        return out
