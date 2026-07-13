"""Search over the Codescope index.

Retrievers:
  * BM25 (FTS5) - lexical relevance over symbol name/path/body
  * vector (sqlite-vec) - semantic similarity over embedded symbol bodies
  * trigram (FTS5) - fast substring / identifier-fragment matching
  * regex (ripgrep) - full regex over file contents (line hits, not symbols)

``hybrid_search`` fuses BM25 and vector results with Reciprocal Rank Fusion
(RRF, k=60), which combines rankings by position only - no fragile mixing of
BM25 scores with vector distances.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from codescope.index.embed import embedder_from_id
from codescope.index.indexer import default_db_path
from codescope.index.store import IndexStore

RRF_K = 60
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _cosine_from_l2(distance: float) -> float:
    """Cosine similarity from sqlite-vec's L2 distance over L2-normalized vectors.

    For unit vectors ``|a-b|^2 == 2 - 2*cos``, so ``cos == 1 - d^2/2``.
    """
    return max(-1.0, min(1.0, 1.0 - (distance * distance) / 2.0))


@dataclass(slots=True)
class SearchHit:
    name: str
    kind: str
    path: str
    start_line: int
    end_line: int
    signature: str
    score: float
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
    members: list[CloneMember]
    similarity: float  # mean pairwise similarity within the cluster


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

    def _bm25_ids(self, store: IndexStore, query: str, k: int) -> list[int]:
        match = _fts_or_query(query)
        if match is None:
            return []
        rows = store.conn.execute(
            "SELECT rowid FROM symbols_fts WHERE symbols_fts MATCH ? ORDER BY bm25(symbols_fts, 10.0, 2.0, 1.0) LIMIT ?",
            (match, k),
        ).fetchall()
        return [r[0] for r in rows]

    def _vector_ids(self, store: IndexStore, query: str, k: int) -> list[int]:
        if not store.has_vectors():
            return []
        embedder_id = store.get_meta("embedder_id")
        if not embedder_id:
            return []
        try:
            embedder = embedder_from_id(embedder_id)
            qvec = embedder.embed_query(query)
            return [sid for sid, _dist in store.vector_search(qvec, k)]
        except Exception:
            return []

    def _trigram_ids(self, store: IndexStore, substring: str, k: int) -> list[int]:
        if len(substring) < 3 or not substring.strip() or k <= 0:
            return []
        rows = store.conn.execute(
            "SELECT rowid FROM symbols_trgm WHERE symbols_trgm MATCH ? LIMIT ?",
            (_fts_phrase(substring), k),
        ).fetchall()
        return [r[0] for r in rows]

    def _fetch_symbols(self, store: IndexStore, sids: list[int]) -> dict[int, SearchHit]:
        if not sids:
            return {}
        placeholders = ",".join("?" * len(sids))
        rows = store.conn.execute(
            f"SELECT id, name, kind, path, start_line, end_line, signature FROM symbols WHERE id IN ({placeholders})",
            sids,
        ).fetchall()
        return {
            r[0]: SearchHit(name=r[1], kind=r[2], path=r[3], start_line=r[4], end_line=r[5], signature=r[6] or "", score=0.0) for r in rows
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

    def hybrid_search(self, query: str, limit: int = 10) -> list[SearchHit]:
        """BM25 + vector + trigram fused via RRF. The flagship search."""
        if not query.strip() or limit <= 0:
            return []
        pool = max(limit * 5, 30)
        with IndexStore(self.db_path) as store:
            rankings = {
                "bm25": self._bm25_ids(store, query, pool),
                "vector": self._vector_ids(store, query, pool),
                "trigram": self._trigram_ids(store, query, pool),
            }
            weights = {"bm25": 1.0, "vector": 1.0, "trigram": 0.5}
            fused = self._rrf(rankings, weights)
            if not fused:
                return []
            top = sorted(fused.items(), key=lambda kv: kv[1][0], reverse=True)[:limit]
            hits = self._fetch_symbols(store, [sid for sid, _ in top])
        result = []
        for sid, (score, sources) in top:
            hit = hits.get(sid)
            if hit is None:
                continue
            hit.score = round(score, 6)
            hit.sources = sources
            result.append(hit)
        return result

    def semantic_search(self, query: str, limit: int = 10) -> list[SearchHit]:
        """Pure vector search (falls back to BM25 if no vectors are indexed)."""
        if not query.strip() or limit <= 0:
            return []
        with IndexStore(self.db_path) as store:
            ids = self._vector_ids(store, query, limit)
            source = "vector"
            if not ids:
                ids = self._bm25_ids(store, query, limit)
                source = "bm25"
            hits = self._fetch_symbols(store, ids)
        ordered = []
        for rank, sid in enumerate(ids):
            hit = hits.get(sid)
            if hit is None:
                continue
            hit.score = round(1.0 / (rank + 1), 6)
            hit.sources = [source]
            ordered.append(hit)
        return ordered

    def substring_search(self, substring: str, limit: int = 20) -> list[SearchHit]:
        """Trigram-accelerated substring search over symbol bodies."""
        if limit <= 0:
            return []
        with IndexStore(self.db_path) as store:
            ids = self._trigram_ids(store, substring, limit)
            hits = self._fetch_symbols(store, ids)
        return [hits[sid] for sid in ids if sid in hits]

    def regex_search(self, pattern: str, limit: int = 50) -> list[RegexHit]:
        """Full regex over file contents using ripgrep (line-level hits)."""
        if not pattern or limit <= 0:
            return []
        rg = shutil.which("rg")
        if rg is None:
            raise RuntimeError("ripgrep (rg) is not installed; regex search is unavailable.")
        proc = subprocess.run(
            [rg, "--json", "--max-count", str(limit), pattern, str(self.root)],
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

    def find_similar_code(self, snippet: str, limit: int = 10) -> list[SearchHit]:
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
            embedder_id = store.get_meta("embedder_id")
            if not embedder_id:
                return []
            try:
                embedder = embedder_from_id(embedder_id)
                qvec = embedder.embed_documents([snippet])[0]
                results = store.vector_search(qvec, limit)
            except Exception:
                return []
            hits = self._fetch_symbols(store, [sid for sid, _ in results])
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
        neighbor_k = 16
        with IndexStore(self.db_path) as store:
            if not store.has_vectors():
                raise RuntimeError("Clone detection requires embeddings; run reindex with embeddings enabled")
            rows = store.symbols_with_min_lines(min_lines)
            meta = {r[0]: CloneMember(name=r[1], kind=r[2], path=r[3], start_line=r[4], end_line=r[5], lines=r[5] - r[4] + 1) for r in rows}
            parent: dict[int, int] = {sid: sid for sid in meta}
            pair_sims: dict[tuple[int, int], float] = {}

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: int, b: int) -> None:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            for sid in meta:
                vec = store.get_embedding(sid)
                if vec is None:
                    continue
                for nsid, dist in store.vector_search(vec, neighbor_k):
                    if nsid == sid or nsid not in meta:
                        continue
                    sim = _cosine_from_l2(dist)
                    if sim < similarity:
                        continue
                    union(sid, nsid)
                    pair_sims[(min(sid, nsid), max(sid, nsid))] = sim

        clusters: dict[int, list[int]] = {}
        for sid in meta:
            clusters.setdefault(find(sid), []).append(sid)

        groups: list[CloneGroup] = []
        for sids in clusters.values():
            if len(sids) < 2:
                continue
            sims = [s for (a, b), s in pair_sims.items() if a in sids and b in sids]
            mean_sim = sum(sims) / len(sims) if sims else similarity
            members = sorted((meta[sid] for sid in sids), key=lambda m: (m.path, m.start_line))
            groups.append(CloneGroup(members=members, similarity=round(mean_sim, 6)))

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

        with IndexStore(self.db_path) as store:
            if not store.has_vectors():
                raise RuntimeError("Clone detection requires embeddings; run reindex with embeddings enabled")

        findings: list[DiffClone] = []
        for block in vcs.added_blocks(self.root, staged=staged, min_lines=min_lines):
            for hit in self.find_similar_code(block.text, limit=5):
                same_block = hit.path == block.path and not (hit.end_line < block.start_line or hit.start_line > block.end_line)
                if same_block:
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
