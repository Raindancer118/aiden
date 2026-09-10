"""Search over the Codescope index.

Retrievers:
  * name (SQL) - exact / prefix match on the symbol name
  * BM25 (FTS5) - lexical relevance over symbol name/path/body
  * vector (sqlite-vec) - semantic similarity over embedded symbols
  * trigram (FTS5) - substring / identifier-fragment matching
  * regex (ripgrep) - full regex over file contents (line hits, not symbols)

``hybrid_search`` fuses these with Reciprocal Rank Fusion (RRF, k=60), which
combines rankings by position only - no fragile mixing of BM25 scores with
vector distances. Only retrievers that actually *rank* their results take
part: the trigram index answers a containment question, so it joins the
fusion only for identifier-shaped queries and is ordered by BM25 first.

Every retriever accepts the same :class:`SearchFilter` and applies it before
its own top-k cut, so filtering narrows the search rather than discarding
results that were already chosen.
"""

from __future__ import annotations

import base64
import fnmatch
import json
import logging
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from codescope.index.embed import build_snippet_embed_text, embedder_from_id
from codescope.index.indexer import default_db_path
from codescope.index.store import IndexStore

log = logging.getLogger(__name__)

RRF_K = 60
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

#: An identifier-shaped query (one token, no spaces) is what the trigram
#: index can actually answer; a sentence cannot appear verbatim in code.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:$-]{3,}$")

#: Candidate pool per retriever, relative to the requested result count.
_POOL_FACTOR = 10
_POOL_MIN = 100

#: With a *selective* filter, score the allowed candidates exactly rather than
#: hoping they surface in an unfiltered kNN sweep. Kept small: decoding
#: vectors into Python costs more than the sweep once the set is large, and a
#: filter that leaves most of the corpus does not need the exact path anyway.
_EXACT_SCAN_MAX = 2_000
_FILTERED_KNN_OVERFETCH = 8

#: If the widened sweep still cannot fill the result set, try once more this
#: much wider before falling back to the exact scan.
_FILTERED_KNN_WIDEN = 8

#: Neighbours considered per added diff block before the self-match filter.
_DIFF_CANDIDATES = 5

#: Name candidates scored before the top-k cut. Bounded so a one-word query
#: on a huge codebase cannot pull the whole symbol table into Python.
_NAME_CANDIDATE_LIMIT = 2_000

#: Lines of code returned with each hit (head + tail around a fold marker).
_DEFAULT_PREVIEW_LINES = 12

_TEST_PATH_PATTERNS = (
    # Directory conventions, at the root and at any depth.
    "test/*",
    "tests/*",
    "spec/*",
    "*/test/*",
    "*/tests/*",
    "*/spec/*",
    "*/testing/*",
    # File-name conventions, at the root and in any directory.
    "test_*",
    "*/test_*",
    "*_test.*",
    "*/*_test.*",
    "*.test.*",
    "*/*.test.*",
    "*.spec.*",
    "*/*.spec.*",
    "*Test.*",
    "*/*Test.*",
    "*Tests.*",
    "*/*Tests.*",
)


@dataclass(slots=True)
class SearchFilter:
    """Narrows every retriever *before* it takes its top-k.

    :param path_glob: only symbols whose project-relative path matches this
        glob (e.g. ``src/auth/*``, ``**/*.ts``).
    :param lang: only symbols in this language (as reported by the index).
    :param kind: only symbols of this kind (``function``, ``class``, ...).
    :param exclude_tests: drop symbols living in test files.
    """

    path_glob: str | None = None
    lang: str | None = None
    kind: str | None = None
    exclude_tests: bool = False

    def is_active(self) -> bool:
        return bool(self.path_glob or self.lang or self.kind or self.exclude_tests)


def _glob_clause(pattern: str) -> tuple[str, list[str]]:
    """SQL prefilter for a path glob, using ``GLOB`` rather than ``LIKE``.

    ``LIKE`` is case-insensitive in SQLite and has no character classes, which
    made it both too broad and too narrow: ``*Test.*`` also excluded
    ``latest.py`` and ``contest.ts``, while ``*.[jt]s`` and bare basenames
    matched nothing at all. ``GLOB`` is case-sensitive and shares its syntax
    with the exact matcher below.

    A pattern without a separator also matches by basename, mirroring
    :func:`_matches_glob`, so this stays a superset of it.
    """
    if "/" in pattern:
        return "s.path GLOB ?", [pattern]
    return "(s.path GLOB ? OR s.path GLOB ?)", [pattern, f"*/{pattern}"]


def _matches_glob(path: str, pattern: str) -> bool:
    """Glob match that also matches a separator-free pattern by basename."""
    if fnmatch.fnmatchcase(path, pattern):
        return True
    return "/" not in pattern and fnmatch.fnmatchcase(path.rsplit("/", 1)[-1], pattern)


def _filter_sql(flt: "SearchFilter | None") -> tuple[str, list[object]]:
    """Build the shared ``AND ...`` fragment applied by every retriever."""
    if flt is None or not flt.is_active():
        return "", []
    clauses: list[str] = []
    params: list[object] = []
    if flt.lang:
        clauses.append("fi.lang = ?")
        params.append(flt.lang)
    if flt.kind:
        clauses.append("s.kind = ?")
        params.append(flt.kind)
    if flt.path_glob:
        clause, values = _glob_clause(flt.path_glob)
        clauses.append(clause)
        params.extend(values)
    if flt.exclude_tests:
        for pattern in _TEST_PATH_PATTERNS:
            clause, values = _glob_clause(pattern)
            clauses.append(f"NOT {clause}")
            params.extend(values)
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def _fold(body: str, max_lines: int) -> str:
    """Return at most ``max_lines`` lines of ``body``, folded in the middle.

    A hit without code forces the agent into a second read; a hit with the
    whole body burns its context. Head plus tail keeps the signature, the
    early logic and the return value, which is what a reader checks first.
    """
    if max_lines <= 0 or not body:
        return ""
    lines = body.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    head = max(1, (max_lines * 2) // 3)
    tail = max(1, max_lines - head - 1)
    hidden = len(lines) - head - tail
    return "\n".join([*lines[:head], f"... [{hidden} lines omitted] ...", *lines[-tail:]])


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _l2_from_cosine(cosine: float) -> float:
    """Inverse of :func:`_cosine_from_l2` for unit vectors."""
    return math.sqrt(max(0.0, 2.0 - 2.0 * cosine))


def _cosine_from_l2(distance: float) -> float:
    """Cosine similarity from sqlite-vec's L2 distance over L2-normalized vectors.

    For unit vectors ``|a-b|^2 == 2 - 2*cos``, so ``cos == 1 - d^2/2``.
    """
    return max(-1.0, min(1.0, 1.0 - (distance * distance) / 2.0))


@dataclass(slots=True)
class SearchHit:
    """One matched symbol, complete enough to act on without a second read."""

    symbol_id: int
    name: str
    kind: str
    path: str
    start_line: int
    end_line: int
    signature: str
    score: float
    lang: str = ""
    preview: str = ""
    sources: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RegexHit:
    path: str
    line: int
    text: str


@dataclass(slots=True)
class CloneMember:
    name: str
    kind: str
    path: str
    start_line: int
    end_line: int
    lines: int


@dataclass(slots=True)
class CloneGroup:
    """A cluster of symbols whose embeddings are near-identical.

    ``similarity`` is the mean over *all* member pairs and ``min_similarity``
    the weakest pair in the group. Both matter: clusters are grown by linking
    pairs above the threshold, and similarity is not transitive, so a chain
    A~B~C can contain an A/C pair that is not similar at all. A group with a
    high mean but a low ``min_similarity`` is a chain, not a set of clones.

    Semantic similarity is evidence of duplication, not proof of copy-paste;
    verify a group before refactoring on it.
    """

    members: list[CloneMember]
    similarity: float
    min_similarity: float = 0.0


@dataclass(slots=True)
class DiffClone:
    added_path: str
    added_start_line: int
    added_end_line: int
    similarity: float
    matches: SearchHit  # the closest existing symbol the added block duplicates


def _fts_or_query(query: str) -> str | None:
    """Build a safe FTS5 MATCH expression: OR of quoted word tokens."""
    tokens = _WORD_RE.findall(query)
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


def _escape_like(value: str) -> str:
    """Escape LIKE wildcards so ``get_user`` cannot match ``getXuserData``."""
    return value.replace("\\", "\\\\").replace("_", "\\_").replace("%", "\\%")


def _fts_phrase(value: str) -> str:
    """Build a quoted FTS5 phrase with embedded quotes escaped."""
    return '"' + value.replace('"', '""') + '"'


def _decode_rg_field(field: dict[str, str]) -> str:
    """Decode a ripgrep JSON text-or-base64 field."""
    text = field.get("text")
    if text is not None:
        return text
    encoded = field.get("bytes")
    if encoded is None:
        return ""
    return base64.b64decode(encoded).decode("utf-8", errors="replace")


class SearchEngine:
    def __init__(self, project_root: str | Path, db_path: str | Path | None = None):
        self.root = Path(project_root).resolve()
        self.db_path = Path(db_path) if db_path else default_db_path(self.root)

    # -- low-level retrievers (return ordered symbol ids) ------------------

    def _bm25_ids(self, store: IndexStore, query: str, k: int, flt: SearchFilter | None = None) -> list[int]:
        match = _fts_or_query(query)
        if match is None:
            return []
        where, params = _filter_sql(flt)
        rows = store.conn.execute(
            "SELECT f.rowid FROM symbols_fts AS f "
            "JOIN symbols AS s ON s.id = f.rowid "
            "JOIN files AS fi ON fi.path = s.path "
            f"WHERE symbols_fts MATCH ?{where} "
            "ORDER BY bm25(symbols_fts, 10.0, 2.0, 1.0) LIMIT ?",
            (match, *params, k),
        ).fetchall()
        return [r[0] for r in rows]

    def _name_ids(self, store: IndexStore, query: str, k: int, flt: SearchFilter | None = None) -> list[int]:
        """Symbols whose *name* the query names outright.

        Agents mostly search for things they can already name, and an exact
        name match is near-certain relevance -- but only when the name really
        answers the query. Ranking every symbol whose name contains any one
        query word puts ``new`` at the top of "reuse before writing new code",
        so candidates are scored by how much of the query their name covers,
        and a multi-word question needs more than a single incidental word.
        """
        tokens = [t.lower() for t in dict.fromkeys(_WORD_RE.findall(query)) if len(t) >= 3]
        if not tokens:
            return []
        where, params = _filter_sql(flt)
        like = " OR ".join(["LOWER(s.name) LIKE ? ESCAPE '\\'"] * len(tokens))
        rows = store.conn.execute(
            f"SELECT s.id, s.name FROM symbols AS s JOIN files AS fi ON fi.path = s.path WHERE ({like}){where} LIMIT ?",
            (*[f"%{_escape_like(t)}%" for t in tokens], *params, _NAME_CANDIDATE_LIMIT),
        ).fetchall()
        if not rows:
            return []

        # Coverage: how many distinct query words the name accounts for.
        min_coverage = 2 if len(tokens) >= 3 else 1
        scored: list[tuple[int, int, int, int]] = []
        for sid, name in rows:
            lowered = name.lower()
            covered = sum(1 for t in tokens if t in lowered)
            exact = 1 if lowered in tokens else 0
            scored.append((covered, exact, -len(name), sid))
        best = [entry for entry in scored if entry[0] >= min_coverage]
        if not best and len(tokens) < 3:
            # A one- or two-word query really is naming something.
            best = [entry for entry in scored if entry[1]]
        if not best:
            # No name answers this question. Contributing a weak guess here
            # would hand it the highest fusion weight; BM25 and the vector
            # index are better placed to answer a prose query.
            return []
        best.sort(reverse=True)
        return [sid for _c, _e, _l, sid in best[:k]]

    def _vector_ids(self, store: IndexStore, query: str, k: int, flt: SearchFilter | None = None) -> list[int]:
        qvec = self._embed_query(store, query)
        if qvec is None:
            return []
        return self._vector_ids_for(store, qvec, k, flt)

    @staticmethod
    def _embed_query(store: IndexStore, query: str) -> list[float] | None:
        if not store.has_vectors():
            return None
        embedder_id = store.get_meta("embedder_id")
        if not embedder_id:
            return None
        try:
            return embedder_from_id(embedder_id).embed_query(query)
        except Exception as e:
            log.warning("Query embedding failed (%s); falling back to lexical retrieval.", e)
            return None

    def _vector_ids_for(self, store: IndexStore, qvec: list[float], k: int, flt: SearchFilter | None = None) -> list[int]:
        """KNN over the vector index, honouring ``flt`` without losing recall.

        sqlite-vec's ``+path``/``+lang`` are auxiliary columns, not partition
        keys, so a filter cannot be pushed into the ``MATCH``. Post-filtering
        the top-k would silently return nothing whenever the unfiltered
        neighbours all sit outside the filter, so a filtered search instead
        scores the *allowed* candidates exactly when there are few enough of
        them, and only widens the kNN sweep when there are not.
        """
        allowed = self._filtered_ids(store, flt)
        if allowed is None:
            return [sid for sid, _dist in store.vector_search(qvec, k)]
        if not allowed:
            return []
        if len(allowed) <= _EXACT_SCAN_MAX:
            return self._exact_scan(store, qvec, k, allowed)

        # A filter that leaves thousands of candidates is not selective, so an
        # over-fetched sweep almost always fills the result set -- and costs a
        # fraction of decoding every allowed vector into Python.
        for factor in (_FILTERED_KNN_OVERFETCH, _FILTERED_KNN_OVERFETCH * _FILTERED_KNN_WIDEN):
            hits = store.vector_search(qvec, min(k * factor, len(allowed)))
            found = [sid for sid, _dist in hits if sid in allowed][:k]
            if len(found) >= k:
                return found
        return found or self._exact_scan(store, qvec, k, allowed)

    @staticmethod
    def _exact_scan(store: IndexStore, qvec: list[float], k: int, allowed: set[int]) -> list[int]:
        """Score every allowed candidate directly. Exact, but decodes vectors."""
        import numpy as np

        vectors = store.embeddings_for(sorted(allowed))
        if not vectors:
            return []
        sids = list(vectors)
        sims = np.asarray([vectors[s] for s in sids], dtype=np.float32) @ np.asarray(qvec, dtype=np.float32)
        return [sids[int(i)] for i in np.argsort(-sims)[:k]]

    def _trigram_ids(self, store: IndexStore, substring: str, k: int, flt: SearchFilter | None = None) -> list[int]:
        """Substring matches over symbol bodies, ranked rather than arbitrary.

        FTS5's trigram index answers "does this text contain that substring",
        so it is only meaningful for identifier-shaped queries; a whole
        natural-language question as one phrase matches nothing useful. The
        previous implementation also returned rows in storage order and fed
        that into rank fusion as if it were a ranking, which injected noise
        into every hybrid search.
        """
        if len(substring) < 3 or not substring.strip() or k <= 0:
            return []
        where, params = _filter_sql(flt)
        rows = store.conn.execute(
            "SELECT t.rowid FROM symbols_trgm AS t "
            "JOIN symbols AS s ON s.id = t.rowid "
            "JOIN files AS fi ON fi.path = s.path "
            f"WHERE symbols_trgm MATCH ?{where} "
            "ORDER BY bm25(symbols_trgm) LIMIT ?",
            (_fts_phrase(substring), *params, k),
        ).fetchall()
        return [r[0] for r in rows]

    @staticmethod
    def _filtered_ids(store: IndexStore, flt: SearchFilter | None) -> set[int] | None:
        """Symbol ids passing ``flt``; ``None`` when no filter is active."""
        where, params = _filter_sql(flt)
        if not where:
            return None
        rows = store.conn.execute(
            f"SELECT s.id FROM symbols AS s JOIN files AS fi ON fi.path = s.path WHERE 1=1{where}",
            params,
        ).fetchall()
        return {r[0] for r in rows}

    def _fetch_symbols(self, store: IndexStore, sids: list[int], preview_lines: int = _DEFAULT_PREVIEW_LINES) -> dict[int, SearchHit]:
        if not sids:
            return {}
        placeholders = ",".join("?" * len(sids))
        rows = store.conn.execute(
            "SELECT s.id, s.name, s.kind, s.path, s.start_line, s.end_line, s.signature, fi.lang, f.body "
            "FROM symbols AS s "
            "JOIN files AS fi ON fi.path = s.path "
            "LEFT JOIN symbols_fts AS f ON f.rowid = s.id "
            f"WHERE s.id IN ({placeholders})",
            sids,
        ).fetchall()
        return {
            r[0]: SearchHit(
                symbol_id=r[0],
                name=r[1],
                kind=r[2],
                path=r[3],
                start_line=r[4],
                end_line=r[5],
                signature=r[6] or "",
                lang=r[7] or "",
                score=0.0,
                preview=_fold(r[8] or "", preview_lines),
            )
            for r in rows
        }

    @staticmethod
    def _rrf(rankings: dict[str, list[int]], weights: dict[str, float]) -> dict[int, tuple[float, list[str]]]:
        fused: dict[int, tuple[float, list[str]]] = {}
        for source, ranking in rankings.items():
            w = weights.get(source, 1.0)
            for rank, sid in enumerate(ranking):
                prev_score, prev_sources = fused.get(sid, (0.0, []))
                fused[sid] = (prev_score + w / (RRF_K + rank + 1), [*prev_sources, source])
        return fused

    # -- public API -------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        flt: SearchFilter | None = None,
        preview_lines: int = _DEFAULT_PREVIEW_LINES,
    ) -> list[SearchHit]:
        """Name + BM25 + vector (+ trigram) fused via RRF. The flagship search."""
        if not query.strip() or limit <= 0:
            return []
        pool = max(limit * _POOL_FACTOR, _POOL_MIN)
        with IndexStore(self.db_path) as store:
            rankings = {
                "name": self._name_ids(store, query, pool, flt),
                "bm25": self._bm25_ids(store, query, pool, flt),
                "vector": self._vector_ids(store, query, pool, flt),
            }
            # Containment, not relevance: only meaningful when the query is
            # itself an identifier fragment.
            if _IDENTIFIER_RE.match(query.strip()):
                rankings["trigram"] = self._trigram_ids(store, query.strip(), pool, flt)
            weights = {"name": 1.5, "bm25": 1.0, "vector": 1.0, "trigram": 0.5}
            fused = self._rrf(rankings, weights)
            if not fused:
                return []
            ranked = sorted(fused.items(), key=lambda kv: kv[1][0], reverse=True)
            # Fetch a slice wider than the limit so the exact glob (which the
            # SQL prefilter deliberately over-approximates) trims candidates
            # rather than the final answer.
            candidates = ranked[: limit * 3] if flt is not None and flt.path_glob else ranked[:limit]
            hits = self._apply_glob(self._fetch_symbols(store, [sid for sid, _ in candidates], preview_lines), flt)
            top = [entry for entry in candidates if entry[0] in hits][:limit]
        result = []
        for sid, (score, sources) in top:
            hit = hits.get(sid)
            if hit is None:
                continue
            hit.score = round(score, 6)
            hit.sources = sources
            result.append(hit)
        return result

    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        flt: SearchFilter | None = None,
        preview_lines: int = _DEFAULT_PREVIEW_LINES,
    ) -> list[SearchHit]:
        """Pure vector search (falls back to BM25 if no vectors are indexed)."""
        if not query.strip() or limit <= 0:
            return []
        with IndexStore(self.db_path) as store:
            ids = self._vector_ids(store, query, limit, flt)
            source = "vector"
            if not ids:
                ids = self._bm25_ids(store, query, limit, flt)
                source = "bm25"
            hits = self._apply_glob(self._fetch_symbols(store, ids, preview_lines), flt)
        ordered = []
        for rank, sid in enumerate(ids):
            hit = hits.get(sid)
            if hit is None:
                continue
            hit.score = round(1.0 / (rank + 1), 6)
            hit.sources = [source]
            ordered.append(hit)
        return ordered

    def substring_search(
        self,
        substring: str,
        limit: int = 20,
        flt: SearchFilter | None = None,
        preview_lines: int = _DEFAULT_PREVIEW_LINES,
    ) -> list[SearchHit]:
        """Trigram-accelerated substring search over symbol bodies."""
        if limit <= 0:
            return []
        with IndexStore(self.db_path) as store:
            ids = self._trigram_ids(store, substring, limit, flt)
            hits = self._apply_glob(self._fetch_symbols(store, ids, preview_lines), flt)
        return [hits[sid] for sid in ids if sid in hits]

    @staticmethod
    def _apply_glob(hits: dict[int, SearchHit], flt: SearchFilter | None) -> dict[int, SearchHit]:
        """Apply exact glob semantics that the SQL LIKE prefilter cannot express."""
        if flt is None or not flt.path_glob:
            return hits
        return {sid: hit for sid, hit in hits.items() if _matches_glob(hit.path, flt.path_glob)}

    def regex_search(self, pattern: str, limit: int = 50) -> list[RegexHit]:
        """Full regex over file contents using ripgrep (line-level hits)."""
        if not pattern or limit <= 0:
            return []
        rg = shutil.which("rg")
        if rg is None:
            raise RuntimeError("ripgrep (rg) is not installed; regex search is unavailable.")
        proc = subprocess.run(
            # -e / -- : a pattern starting with '-' is a pattern, not a flag.
            [rg, "--json", "--max-count", str(limit), "-e", pattern, "--", str(self.root)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode not in (0, 1):
            detail = proc.stderr.strip() or f"exit status {proc.returncode}"
            raise RuntimeError(f"ripgrep search failed: {detail}")

        hits: list[RegexHit] = []
        for line in proc.stdout.splitlines():
            record = json.loads(line)
            if record.get("type") != "match":
                continue
            data = record["data"]
            path = _decode_rg_field(data["path"])
            line_number = data.get("line_number")
            if not isinstance(line_number, int):
                continue
            matched_text = _decode_rg_field(data["lines"])
            try:
                rel = str(Path(path).resolve().relative_to(self.root))
            except ValueError:
                rel = path
            hits.append(RegexHit(path=rel, line=line_number, text=matched_text.strip()[:200]))
            if len(hits) >= limit:
                break
        return hits

    # -- reuse / clone detection ------------------------------------------

    def find_similar_code(
        self,
        snippet: str,
        limit: int = 10,
        flt: SearchFilter | None = None,
        preview_lines: int = _DEFAULT_PREVIEW_LINES,
    ) -> list[SearchHit]:
        """Find indexed symbols whose code is most similar to ``snippet``.

        The inverse of ``semantic_search``: instead of a natural-language query
        you pass a *code snippet* (the block you are about to write) and get the
        nearest existing symbols by vector similarity - the "reuse before write"
        lookup. Returns [] when no embeddings are indexed.
        """
        if not snippet.strip() or limit <= 0:
            return []
        with IndexStore(self.db_path) as store:
            if not store.has_vectors():
                return []
            try:
                results = self._similar_to(store, [snippet], limit, flt)[0]
            except Exception as e:
                log.warning("Similarity search failed: %s", e)
                return []
            return self._score_similar(store, results, preview_lines, flt)

    def _similar_to(self, store: IndexStore, snippets: list[str], limit: int, flt: SearchFilter | None) -> list[list[tuple[int, float]]]:
        """Nearest indexed symbols for each snippet, embedding them in one pass.

        Callers with many snippets (the diff gate) must not re-resolve the
        embedder or reopen the store per snippet -- that turned a pre-commit
        check into a series of model loads.
        """
        embedder_id = store.get_meta("embedder_id")
        if not embedder_id:
            raise RuntimeError("The index has no embedder recorded; run reindex with embeddings enabled.")
        embedder = embedder_from_id(embedder_id)
        texts = [build_snippet_embed_text(s) for s in snippets]
        vectors: list[list[float]] = [[] for _ in texts]
        for batch in embedder.embed_batched(texts):
            for index, vector in batch:
                vectors[index] = vector
        out: list[list[tuple[int, float]]] = []
        for vec in vectors:
            if flt is None or not flt.is_active():
                out.append(store.vector_search(vec, limit))
            else:
                sids = self._vector_ids_for(store, vec, limit, flt)
                stored = store.embeddings_for(sids)
                out.append([(sid, _l2_from_cosine(_dot(vec, stored[sid]))) for sid in sids if sid in stored])
        return out

    def _score_similar(
        self, store: IndexStore, results: list[tuple[int, float]], preview_lines: int, flt: SearchFilter | None
    ) -> list[SearchHit]:
        hits = self._apply_glob(self._fetch_symbols(store, [sid for sid, _ in results], preview_lines), flt)
        ordered: list[SearchHit] = []
        for sid, dist in results:
            hit = hits.get(sid)
            if hit is None:
                continue
            hit.score = round(_cosine_from_l2(dist), 6)
            hit.sources = ["vector"]
            ordered.append(hit)
        return ordered

    def find_duplicate_code(self, min_lines: int = 5, similarity: float = 0.9, limit: int = 50) -> list[CloneGroup]:
        """Cluster near-duplicate symbol bodies across the codebase.

        Embeds every symbol spanning at least ``min_lines`` lines (already done
        at index time) and links any pair whose cosine similarity is at least
        ``similarity`` into the same clone group via union-find. Surfaces copy-
        paste that ``find_symbol`` / lexical search miss. Raises clearly when
        no embeddings are indexed so an empty result cannot be mistaken for a
        successful clone audit.

        :param min_lines: ignore symbols shorter than this (skips trivial code).
        :param similarity: cosine threshold in [0, 1] for two symbols to count
            as duplicates (1.0 == identical embedding).
        :param limit: maximum number of clone groups to return.
        """
        if limit <= 0:
            return []
        import numpy as np

        with IndexStore(self.db_path) as store:
            if not store.has_vectors():
                raise RuntimeError("Clone detection requires embeddings; run reindex with embeddings enabled")
            rows = store.symbols_with_min_lines(min_lines)
            meta = {r[0]: CloneMember(name=r[1], kind=r[2], path=r[3], start_line=r[4], end_line=r[5], lines=r[5] - r[4] + 1) for r in rows}
            vectors = store.embeddings_for(list(meta))

        sids = [sid for sid in meta if sid in vectors]
        if len(sids) < 2:
            return []

        # One dense matrix product instead of one exact kNN query per symbol.
        # Vectors are L2-normalized, so the Gram matrix *is* the pairwise
        # cosine similarity; blocking the rows keeps peak memory bounded for
        # large codebases.
        matrix = np.asarray([vectors[sid] for sid in sids], dtype=np.float32)
        parent: dict[int, int] = {sid: sid for sid in sids}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        block = 512
        for start in range(0, len(sids), block):
            sims = matrix[start : start + block] @ matrix.T
            for local_row, col in zip(*np.nonzero(sims >= similarity), strict=True):
                i, j = start + int(local_row), int(col)
                if i >= j:  # upper triangle only; skips self-similarity
                    continue
                union(sids[i], sids[j])

        clusters: dict[int, list[int]] = {}
        for sid in sids:
            clusters.setdefault(find(sid), []).append(sid)

        position = {sid: i for i, sid in enumerate(sids)}
        groups: list[CloneGroup] = []
        for cluster in clusters.values():
            if len(cluster) < 2:
                continue
            # Score over every pair in the cluster, not only the pairs that
            # happened to cross the threshold: reporting the mean of the
            # discovered edges alone would claim a chain is a clique.
            rows = matrix[[position[sid] for sid in cluster]]
            pairwise = rows @ rows.T
            upper = np.triu_indices(len(cluster), k=1)
            values = pairwise[upper]
            members = sorted((meta[sid] for sid in cluster), key=lambda m: (m.path, m.start_line))
            groups.append(
                CloneGroup(
                    members=members,
                    similarity=round(float(values.mean()), 6),
                    min_similarity=round(float(values.min()), 6),
                )
            )

        groups.sort(key=lambda g: (len(g.members), g.similarity), reverse=True)
        return groups[:limit]

    def detect_clones_in_diff(
        self, *, staged: bool = False, min_lines: int = 5, similarity: float = 0.85, limit: int = 50
    ) -> list[DiffClone]:
        """Flag newly added code that duplicates code already in the index.

        A pre-commit / pre-write gate for the "reuse before write" workflow:
        parses the git diff for blocks of added lines and, for each block of at
        least ``min_lines`` lines, finds its nearest existing indexed symbol.
        Blocks whose closest match scores at least ``similarity`` are reported
        as likely duplication. Self-matches (the same file/line range that the
        index already covers) are filtered out.

        :param staged: diff the staged changes instead of the working tree.
            Untracked files only appear once staged or marked intent-to-add.
        :param min_lines: ignore added blocks shorter than this many lines.
        :param similarity: cosine threshold for an added block to count as a
            duplicate of an existing symbol.
        :param limit: maximum number of findings to return.
        """
        if limit <= 0:
            return []
        from codescope.devops import vcs

        findings: list[DiffClone] = []
        with IndexStore(self.db_path) as store:
            # Checked before reading the diff: "no findings" must never be how
            # a missing vector index reports itself.
            if not store.has_vectors():
                raise RuntimeError("Clone detection requires embeddings; run reindex with embeddings enabled")
            blocks = list(vcs.added_blocks(self.root, staged=staged, min_lines=min_lines))
            if not blocks:
                return []
            # One embedder and one connection for the whole diff: the previous
            # per-block call re-resolved both, and its blanket ``except``
            # reported a broken backend as a clean diff.
            per_block = self._similar_to(store, [b.text for b in blocks], _DIFF_CANDIDATES, None)
            for block, results in zip(blocks, per_block, strict=True):
                for hit in self._score_similar(store, results, _DEFAULT_PREVIEW_LINES, None):
                    # Skip the block matching the indexed copy of itself.
                    if hit.path == block.path and not (hit.end_line < block.start_line or hit.start_line > block.end_line):
                        continue
                    if hit.score >= similarity:
                        findings.append(
                            DiffClone(
                                added_path=block.path,
                                added_start_line=block.start_line,
                                added_end_line=block.end_line,
                                similarity=hit.score,
                                matches=hit,
                            )
                        )
                    break
        findings.sort(key=lambda f: f.similarity, reverse=True)
        return findings[:limit]
