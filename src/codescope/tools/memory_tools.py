"""Diary-backed memory tools for Codescope.

The local Markdown memory tools inherited from Serena are disabled by the
Codescope entrypoint. These tools proxy Diary's public MCP API instead.
"""

from __future__ import annotations

from typing import Optional

from codescope.diary import get_diary_bridge, project_slug
from serena.tools import Tool, ToolMarkerCanEdit, ToolMarkerDoesNotRequireActiveProject


class MemoryContextTool(Tool, ToolMarkerDoesNotRequireActiveProject):
    """Return Diary's session-start memory snapshot."""

    def apply(self) -> str:
        """Return the complete Diary memory overview and recent full entries."""
        return get_diary_bridge().call("memory_context")


class MemoryProjectContextTool(Tool):
    """Return Diary context for the active Codescope project."""

    def apply(self, project_slug_override: Optional[str] = None, only_pinned: bool = True) -> str:
        """Register and load the active project through Diary.

        :param project_slug_override: explicit Diary slug; by default the active
            project directory name is normalized to a lower-case slug.
        :param only_pinned: return only project memories pinned for session start.
        :return: Diary's project memory context.
        """
        root = self.get_project_root()
        slug = project_slug_override or project_slug(root)
        bridge = get_diary_bridge()
        bridge.call("memory_set_project_dir", {"project_slug": slug, "dir_path": root})
        return bridge.call("memory_project_context", {"project_slug": slug, "only_pinned": only_pinned})


class MemoryGetTool(Tool, ToolMarkerDoesNotRequireActiveProject):
    """Read one Diary memory node by its absolute tree path."""

    def apply(self, path: str) -> str:
        """Return the complete Diary node.

        :param path: absolute Diary path such as ``/projects/codescope/status``.
        :return: the node including its metadata and body.
        """
        return get_diary_bridge().call("memory_get", {"path": path})


class MemoryTreeTool(Tool, ToolMarkerDoesNotRequireActiveProject):
    """List Diary's memory tree below a path."""

    def apply(self, path: str = "/", include_extracted: bool = False) -> str:
        """Return a compact Diary subtree.

        :param path: root path of the requested subtree.
        :param include_extracted: include automatically extracted tier-2 nodes.
        :return: Diary's rendered tree.
        """
        return get_diary_bridge().call("memory_tree", {"path": path, "include_extracted": include_extracted})


class MemorySearchTool(Tool, ToolMarkerDoesNotRequireActiveProject):
    """Run Diary's lexical memory search."""

    def apply(self, query: str, include_expired: bool = False, include_extracted: bool = False) -> str:
        """Search Diary by exact terms and metadata.

        :param query: lexical search query.
        :param include_expired: include nodes past their validity date.
        :param include_extracted: include automatically extracted tier-2 nodes.
        :return: ranked Diary matches.
        """
        return get_diary_bridge().call(
            "memory_search",
            {"query": query, "include_expired": include_expired, "include_extracted": include_extracted},
        )


class MemorySearchSemanticTool(Tool, ToolMarkerDoesNotRequireActiveProject):
    """Run Diary's semantic vector search."""

    def apply(
        self,
        query: str,
        top_k: int = 10,
        include_expired: bool = False,
        include_extracted: bool = False,
    ) -> str:
        """Search Diary by meaning.

        :param query: natural-language semantic query.
        :param top_k: maximum number of matches.
        :param include_expired: include nodes past their validity date.
        :param include_extracted: include automatically extracted tier-2 nodes.
        :return: ranked Diary matches.
        """
        return get_diary_bridge().call(
            "memory_search_semantic",
            {
                "query": query,
                "top_k": top_k,
                "include_expired": include_expired,
                "include_extracted": include_extracted,
            },
        )


class MemoryUpsertTool(Tool, ToolMarkerCanEdit, ToolMarkerDoesNotRequireActiveProject):
    """Create or update a Diary memory node."""

    def apply(
        self,
        path: str,
        title: str,
        body: str,
        type: str = "note",
        tags: Optional[str] = None,
        importance: float = 0.5,
        valid_until: Optional[str] = None,
        origin: str = "curated",
    ) -> str:
        """Upsert one node through Diary's official MCP API.

        :param path: absolute lower-case Diary tree path.
        :param title: concise node title.
        :param body: complete durable memory content.
        :param type: Diary type such as project, feedback, reference, or note.
        :param tags: optional comma-separated tags.
        :param importance: importance score from 0.0 through 1.0.
        :param valid_until: optional ISO validity date.
        :param origin: ``curated`` or ``extracted``.
        :return: Diary's confirmation.
        """
        arguments = {
            "path": path,
            "title": title,
            "body": body,
            "type": type,
            "importance": importance,
            "origin": origin,
        }
        if tags is not None:
            arguments["tags"] = tags
        if valid_until is not None:
            arguments["valid_until"] = valid_until
        return get_diary_bridge().call("memory_upsert", arguments)


class MemoryDeleteTool(Tool, ToolMarkerCanEdit, ToolMarkerDoesNotRequireActiveProject):
    """Delete a Diary memory node through Diary's public API."""

    def apply(self, path: str) -> str:
        """Delete one Diary node.

        :param path: absolute path of the node explicitly approved for deletion.
        :return: Diary's confirmation.
        """
        return get_diary_bridge().call("memory_delete", {"path": path})
