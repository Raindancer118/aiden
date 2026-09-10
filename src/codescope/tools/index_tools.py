"""Index management tools: build/refresh the index and report its status."""

from __future__ import annotations

from dataclasses import asdict

from codescope.index.indexer import Indexer
from serena.tools import Tool


class ReindexTool(Tool):
    """
    Build or refresh the Codescope hybrid index for the active project.

    The index extracts symbols (functions, classes, methods, types, ...) from
    every supported source file using tree-sitter and stores them in a local
    SQLite database under ``.serena/codescope/index.db``. Indexing is
    incremental: unchanged files (matched by content hash) are skipped, and
    files deleted from disk are pruned. Run this once after activating a
    project, and again after large external changes.
    """

    def apply(self, force: bool = False, embeddings: bool = True) -> str:
        """
        Build or refresh the project index.

        :param force: if true, reindex every file even if its content hash is
            unchanged (use after upgrading Codescope or changing the embedder).
        :param embeddings: build semantic vectors. Disable this for a fast
            lexical-only first pass; a later normal reindex fills missing
            vectors without reparsing unchanged files.
        :return: a JSON summary with counts of indexed/skipped/removed files,
            errors, total symbols, per-language file counts, and duration.
        """
        indexer = Indexer(self.get_project_root())
        report = indexer.reindex(force=force, embeddings=embeddings)
        return self._to_json(
            {
                "indexed": report.indexed,
                "skipped_unchanged": report.skipped_unchanged,
                "removed": report.removed,
                "errors": report.errors,
                "duration_s": report.duration_s,
                "totals": asdict(report.stats),
            }
        )


class IndexStatusTool(Tool):
    """
    Report what the Codescope index can currently answer, and what it cannot.

    Beyond the counts (files, symbols, references, vectors, languages), this
    states whether semantic search and clone detection are actually usable:
    an index built without embeddings, or one that fell back to hash vectors
    because the embedding backend failed to load, still answers every query -
    lexically. Check this when results look thin, before concluding that
    something is not in the codebase.

    ``advice`` lists the concrete next step for each problem found (build the
    index, finish the vector backfill, sync after git changes).
    """

    def apply(self) -> str:
        """
        Return a JSON health report for the current index.

        :return: JSON with the index contents, whether semantic search and
            clone detection are ready, how many symbols still lack a vector,
            how many source files changed in git since the last sync, and
            what to do about each.
        """
        return self._to_json(Indexer(self.get_project_root()).health())
