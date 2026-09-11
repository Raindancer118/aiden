"""tree-sitter parsing and symbol/reference extraction.

Important API note (tree-sitter 0.25 + tree-sitter-language-pack 1.8):
``tree_sitter_language_pack.get_parser`` returns a parser object with a broken
ABI in this version. We therefore use ``get_language`` (which returns a proper
``tree_sitter.Language``) and construct a standard ``tree_sitter.Parser``
ourselves. ``Parser.parse`` takes ``bytes``; ``node.text`` is ``bytes``.

Symbol extraction uses Aider's tags queries (vendored under ``queries/``,
Apache-2.0). Their capture vocabulary:
  ``@name.definition.<kind>`` - the identifier node (symbol name + position)
  ``@definition.<kind>``      - the full definition node (full range + body)
  ``@name.reference.<kind>``  - a reference (used for the call/usage graph)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from tree_sitter import Language, Parser, Query, QueryCursor

from aiden.index.languages import LanguageSpec, spec_for_path

log = logging.getLogger(__name__)

# Caps to keep the index compact.
#
# The body cap is deliberately generous: this text is what BM25 and the
# trigram index search over, so a tight cap made the tail of every long
# function invisible to *all* retrievers while hits still reported the full
# line range. The much smaller budget the embedding model needs is applied
# separately when the embedding text is built (see ``embed.build_embed_text``).
_MAX_BODY_CHARS = 20_000
_MAX_SIGNATURE_CHARS = 240

_NAME_DEF_PREFIX = "name.definition."
_DEF_PREFIX = "definition."
_NAME_REF_PREFIX = "name.reference."


@dataclass(slots=True)
class SymbolDef:
    name: str
    kind: str
    start_line: int  # 1-based
    start_col: int
    end_line: int
    end_col: int
    signature: str
    body: str


@dataclass(slots=True)
class RefHit:
    name: str
    kind: str
    line: int  # 1-based
    col: int


@dataclass(slots=True)
class ParseResult:
    language: str
    symbols: list[SymbolDef]
    refs: list[RefHit]


class TreeSitterParser:
    """Lazily loads grammars/queries and extracts symbols and references."""

    def __init__(self) -> None:
        self._lang_cache: dict[str, Language | None] = {}
        self._query_cache: dict[str, Query | None] = {}

    # -- resource loading -------------------------------------------------

    def _get_language(self, name: str) -> Language | None:
        if name not in self._lang_cache:
            try:
                from tree_sitter_language_pack import get_language

                self._lang_cache[name] = get_language(name)  # type: ignore[arg-type]
            except Exception as e:  # pragma: no cover - depends on grammar availability
                log.warning("Could not load grammar for %s: %s", name, e)
                self._lang_cache[name] = None
        return self._lang_cache[name]

    def _get_query(self, spec: LanguageSpec, lang: Language) -> Query | None:
        if spec.name not in self._query_cache:
            qpath = spec.query_path
            if not qpath.exists():
                log.debug("No tags query for %s (%s)", spec.name, qpath.name)
                self._query_cache[spec.name] = None
            else:
                try:
                    self._query_cache[spec.name] = Query(lang, qpath.read_text(encoding="utf-8"))
                except Exception as e:  # pragma: no cover - grammar/query mismatch
                    log.warning("Could not compile tags query for %s: %s", spec.name, e)
                    self._query_cache[spec.name] = None
        return self._query_cache[spec.name]

    # -- parsing ----------------------------------------------------------

    def parse(self, path: str, source: bytes) -> ParseResult | None:
        """Parse ``source`` (raw bytes) for ``path``. None if unsupported."""
        spec = spec_for_path(path)
        if spec is None:
            return None
        lang = self._get_language(spec.name)
        if lang is None:
            return None
        query = self._get_query(spec, lang)
        if query is None:
            return ParseResult(language=spec.name, symbols=[], refs=[])

        parser = Parser(lang)
        tree = parser.parse(source)
        cursor = QueryCursor(query)
        matches = cursor.matches(tree.root_node)

        symbols: list[SymbolDef] = []
        refs: list[RefHit] = []

        for _pattern_index, caps in matches:
            self._collect_from_match(caps, symbols, refs)

        return ParseResult(language=spec.name, symbols=symbols, refs=refs)

    @staticmethod
    def _node_text(node) -> str:  # type: ignore[no-untyped-def]
        raw = node.text
        if raw is None:
            return ""
        return raw.decode("utf-8", errors="replace")

    def _collect_from_match(self, caps: dict, symbols: list[SymbolDef], refs: list[RefHit]) -> None:  # type: ignore[type-arg]
        # Group name-definition captures by kind, and full-definition nodes by kind.
        name_defs: dict[str, list] = {}  # type: ignore[type-arg]
        full_defs: dict[str, list] = {}  # type: ignore[type-arg]

        for cname, nodes in caps.items():
            if cname.startswith(_NAME_DEF_PREFIX):
                name_defs.setdefault(cname[len(_NAME_DEF_PREFIX) :], []).extend(nodes)
            elif cname.startswith(_DEF_PREFIX):
                full_defs.setdefault(cname[len(_DEF_PREFIX) :], []).extend(nodes)
            elif cname.startswith(_NAME_REF_PREFIX):
                kind = cname[len(_NAME_REF_PREFIX) :]
                for n in nodes:
                    name = self._node_text(n)
                    if name:
                        refs.append(RefHit(name=name, kind=kind, line=n.start_point[0] + 1, col=n.start_point[1] + 1))

        for kind, name_nodes in name_defs.items():
            full_nodes = full_defs.get(kind, [])
            for i, name_node in enumerate(name_nodes):
                name = self._node_text(name_node)
                if not name:
                    continue
                # Pair with the full-definition node if available; else use the name node.
                range_node = full_nodes[i] if i < len(full_nodes) else (full_nodes[0] if full_nodes else name_node)
                body = self._node_text(range_node)
                signature = body.splitlines()[0].strip() if body else name
                symbols.append(
                    SymbolDef(
                        name=name,
                        kind=kind,
                        start_line=range_node.start_point[0] + 1,
                        start_col=range_node.start_point[1] + 1,
                        end_line=range_node.end_point[0] + 1,
                        end_col=range_node.end_point[1] + 1,
                        signature=signature[:_MAX_SIGNATURE_CHARS],
                        body=body[:_MAX_BODY_CHARS],
                    )
                )
