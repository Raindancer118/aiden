"""Search tools: hybrid, semantic, and regex search over the codebase."""

from __future__ import annotations

from dataclasses import asdict

from codescope.index.search import SearchEngine
from serena.tools import Tool, ToolMarkerSymbolicRead


class SearchCodeTool(Tool, ToolMarkerSymbolicRead):
    """
    Search the codebase by meaning AND keywords at once (the flagship search).

    Combines BM25 lexical relevance, semantic vector similarity, and trigram
    matching over the Codescope index, fusing them with Reciprocal Rank Fusion.
    Use this for most "where/how is X done?" questions. Requires the index to
    be built first (see the reindex tool). Semantic ranking uses local
    code-aware embeddings by default; if embeddings are unavailable the results
    are lexical (BM25 + trigram), which still works well.
    """

    def apply(self, query: str, limit: int = 10) -> str:
        """
        Run a hybrid (lexical + semantic) search over the indexed codebase.

        :param query: a natural-language description or keywords (e.g.
            "where are auth tokens validated", "parse config file").
        :param limit: maximum number of results to return.
        :return: JSON list of matches, each with name, kind, path, line range,
            signature, fused score, and which retrievers matched it.
        """
        hits = SearchEngine(self.get_project_root()).hybrid_search(query, limit=limit)
        return self._to_json([asdict(h) for h in hits])


class SearchSemanticTool(Tool, ToolMarkerSymbolicRead):
    """
    Search the codebase purely by semantic similarity (vector search).

    Best when you describe behavior in your own words and exact keywords may
    not appear in the code. Falls back to BM25 if no embeddings are indexed.
    """

    def apply(self, query: str, limit: int = 10) -> str:
        """
        Run a semantic (vector) search over the indexed codebase.

        :param query: a natural-language description of what you are looking for.
        :param limit: maximum number of results to return.
        :return: JSON list of matching symbols ranked by semantic similarity.
        """
        hits = SearchEngine(self.get_project_root()).semantic_search(query, limit=limit)
        return self._to_json([asdict(h) for h in hits])


class SearchRegexTool(Tool, ToolMarkerSymbolicRead):
    """
    Search file contents with a regular expression (powered by ripgrep).

    Use for exact patterns, identifiers, or structural text matches across the
    whole project. Returns line-level hits (path + line + text), not symbols.
    For short identifier fragments, the hybrid search is usually better.
    """

    def apply(self, pattern: str, limit: int = 50) -> str:
        """
        Run a regular-expression search over file contents.

        :param pattern: a regular expression (ripgrep/Rust regex syntax).
        :param limit: maximum number of line hits to return.
        :return: JSON list of hits, each with path, line number, and text.
        """
        hits = SearchEngine(self.get_project_root()).regex_search(pattern, limit=limit)
        return self._to_json([asdict(h) for h in hits])


class FindSimilarCodeTool(Tool, ToolMarkerSymbolicRead):
    """
    Find existing code most similar to a code snippet ("reuse before you write").

    The inverse of search_semantic: instead of a natural-language query you pass
    the *code block you are about to write*, and this returns the nearest
    existing symbols by vector similarity. Use it before adding a new function,
    helper, or test to check whether the project already has (almost) the same
    thing. Requires the index to be built with embeddings (see reindex).
    """

    def apply(self, snippet: str, limit: int = 10) -> str:
        """
        Find indexed symbols whose code is closest to ``snippet``.

        :param snippet: the code block to compare against the codebase.
        :param limit: maximum number of results to return.
        :return: JSON list of matching symbols ranked by cosine similarity
            (``score`` is the similarity in [-1, 1]; 1.0 == identical).
        """
        hits = SearchEngine(self.get_project_root()).find_similar_code(snippet, limit=limit)
        return self._to_json([asdict(h) for h in hits])


class FindDuplicateCodeTool(Tool, ToolMarkerSymbolicRead):
    """
    Detect near-duplicate (copy-pasted) code blocks across the whole codebase.

    Clusters indexed symbols whose bodies are semantically near-identical into
    clone groups (like jscpd / PMD-CPD, but embedding-based and language-
    agnostic). Use it to find refactoring opportunities and accidental
    duplication that name- and keyword-based search miss. Requires the index to
    be built with embeddings (see reindex).
    """

    def apply(self, min_lines: int = 5, similarity: float = 0.9, limit: int = 50) -> str:
        """
        Report groups of duplicate / near-duplicate symbols.

        :param min_lines: ignore symbols shorter than this many lines (skips
            trivial getters/one-liners).
        :param similarity: cosine threshold in [0, 1] for two symbols to be
            considered duplicates (higher == stricter; 1.0 == identical).
        :param limit: maximum number of clone groups to return.
        :return: JSON list of clone groups, each with its members (name, kind,
            path, line range, line count) and the mean pairwise similarity.
        """
        groups = SearchEngine(self.get_project_root()).find_duplicate_code(
            min_lines=min_lines, similarity=similarity, limit=limit
        )
        return self._to_json([asdict(g) for g in groups])


class DetectClonesInDiffTool(Tool, ToolMarkerSymbolicRead):
    """
    Check whether newly added (diff) code duplicates code already in the index.

    A pre-commit / pre-write gate for the "reuse before write" workflow: scans
    the git diff for blocks of added lines and reports any block that closely
    matches an existing indexed symbol, so duplication is caught before it
    lands. Untracked files must be staged (git add) or marked intent-to-add
    (git add -N) to show up. Requires an embeddings index (see reindex).
    """

    def apply(
        self, staged: bool = False, min_lines: int = 5, similarity: float = 0.85, limit: int = 50
    ) -> str:
        """
        Report added diff blocks that duplicate existing code.

        :param staged: diff staged changes instead of the working tree.
        :param min_lines: ignore added blocks shorter than this many lines.
        :param similarity: cosine threshold for an added block to count as a
            duplicate (higher == stricter).
        :param limit: maximum number of findings to return.
        :return: JSON list of findings, each with the added block's path and
            line range, the similarity score, and the existing symbol it
            duplicates (``matches``).
        """
        findings = SearchEngine(self.get_project_root()).detect_clones_in_diff(
            staged=staged, min_lines=min_lines, similarity=similarity, limit=limit
        )
        return self._to_json([asdict(f) for f in findings])
