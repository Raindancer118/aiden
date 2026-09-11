"""Project-wide diagnostics - the IDE's Problems view for an agent."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from aiden.index.diagnostics import DEFAULT_MAX_FILES, DEFAULT_MAX_PROBLEMS, collect_diagnostics, files_to_inspect
from serena.tools import Tool, ToolMarkerSymbolicRead


class GetProjectDiagnosticsTool(Tool, ToolMarkerSymbolicRead):
    """
    Find compile/type/lint problems across many files at once (the IDE's Problems view).

    ``get_diagnostics_for_file`` answers for a single file, which is the wrong
    shape for the two questions that actually come up: "did my change break
    anything anywhere?" and "what is already broken before I start?".

    ``scope="changed"`` (the default) inspects the files git reports as
    modified - the cheap, high-signal check after an edit. ``scope="project"``
    sweeps every indexed source file.

    Files whose language has no running server appear under ``files_skipped``
    rather than counting as clean, so an empty problem list can be trusted.
    """

    def apply(
        self,
        scope: str = "changed",
        min_severity: int = 2,
        max_files: int = DEFAULT_MAX_FILES,
        max_problems: int = DEFAULT_MAX_PROBLEMS,
        path_glob: str = "",
    ) -> str:
        """
        Collect diagnostics across the project.

        :param scope: "changed" (files git reports as dirty) or "project"
            (every indexed source file).
        :param min_severity: 1=error, 2=warning, 3=information, 4=hint;
            diagnostics at or above this severity are returned.
        :param max_files: stop after inspecting this many files.
        :param max_problems: stop after collecting this many problems.
        :param path_glob: only inspect files matching this glob.
        :return: JSON with per-severity counts, problems grouped by file
            (line, severity, source, code, message), the files that could not
            be checked, and whether the sweep was truncated.
        """
        if min_severity not in (1, 2, 3, 4):
            # Otherwise the language server rejects every request and the
            # per-file error handling reports "all files skipped, 0 problems",
            # which reads like a clean project.
            raise ValueError("min_severity must be 1 (error), 2 (warning), 3 (information) or 4 (hint)")
        root = Path(self.get_project_root())
        files = files_to_inspect(root, scope, path_glob)
        retriever = self.create_language_server_symbol_retriever()

        def fetch(rel: str) -> list[dict]:
            return list(retriever.get_file_diagnostics(relative_file_path=rel, min_severity=min_severity))

        report = collect_diagnostics(files, fetch, scope=scope, max_files=max_files, max_problems=max_problems)
        return self._to_json(asdict(report))
