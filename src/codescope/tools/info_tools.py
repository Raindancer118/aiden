"""Introductory Codescope tools.

``CodescopeInfoTool`` doubles as a smoke test that the Codescope tool layer is
correctly registered into Serena's tool registry.
"""

from codescope import PRODUCT_NAME, __version__
from serena.tools import Tool, ToolMarkerDoesNotRequireActiveProject


class CodescopeInfoTool(Tool, ToolMarkerDoesNotRequireActiveProject):
    """
    Returns information about the Volantic Codescope server: version and the
    extra capability groups it adds on top of Serena (hybrid search, code
    graphs, incremental indexing, and dev-ops tooling).
    """

    def apply(self) -> str:
        """
        Report the Codescope version and the capability groups it provides.

        Use this to confirm that the Codescope extensions are active and to
        discover which extra tool groups are available beyond Serena's
        built-in LSP-based tools.
        """
        lines = [
            f"{PRODUCT_NAME} v{__version__}",
            "Built on Serena (LSP-accurate navigation & structural editing).",
            "",
            "Codescope adds the following capability groups:",
            "  - Persistent index: tree-sitter symbols in SQLite (33 languages)              [reindex, index_status]",
            "  - Hybrid search: BM25 (FTS5) + local vector (sqlite-vec) + trigram, RRF-fused  [search_code, search_semantic, search_regex]",
            "  - Code graphs: dependencies, dependents, call chains, change impact           [milestone M4]",
            "  - Dev-ops: auto test runner, project scaffolding, GitHub operations           [milestone M6]",
        ]
        return "\n".join(lines)
