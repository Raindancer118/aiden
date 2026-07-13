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
    Report the status of the Codescope index for the active project: number of
    indexed files, extracted symbols and references, and a per-language
    breakdown. If the index has not been built yet, all counts are zero.
    """

    def apply(self) -> str:
        """
        Return a JSON summary of the current index contents.

        :return: JSON with total indexed files, symbols, references, and a
            per-language file-count breakdown.
        """
        indexer = Indexer(self.get_project_root())
        return self._to_json(asdict(indexer.status()))
