"""Search tools: hybrid, semantic, and regex search over the codebase."""

from __future__ import annotations

from dataclasses import asdict

from serena.tools import Tool, ToolMarkerSymbolicRead

from codescope.index.search import SearchEngine


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
