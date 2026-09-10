"""Project-wide diagnostics: sweep behaviour and a real language-server run."""

from __future__ import annotations

from pathlib import Path

import pytest

from codescope.index.diagnostics import collect_diagnostics, files_to_inspect
from solidlsp.ls_config import Language


def _diag(line: int, severity: int = 1, message: str = "boom") -> dict:
    return {"range": {"start": {"line": line, "character": 0}}, "severity": severity, "message": message, "source": "test"}


def test_sweep_groups_by_file_and_counts_by_severity() -> None:
    findings = {"a.py": [_diag(0), _diag(4, severity=2)], "b.py": []}
    report = collect_diagnostics(["a.py", "b.py"], lambda rel: findings[rel])

    assert report.files_inspected == 2
    assert report.counts == {"error": 1, "warning": 1, "information": 0, "hint": 0}
    assert [p["line"] for p in report.problems["a.py"]] == [1, 5]  # 1-based for humans
    assert "b.py" not in report.problems
    assert not report.truncated


def test_uncheckable_file_is_reported_not_counted_as_clean() -> None:
    """'No server for this language' must never look like 'no problems'."""

    def fetch(rel: str) -> list[dict]:
        if rel == "app.rb":
            raise RuntimeError("no language server for ruby\nsecond line ignored")
        return [_diag(1)]

    report = collect_diagnostics(["app.rb", "ok.py"], fetch)
    assert report.files_inspected == 1
    assert "app.rb" in report.files_skipped
    assert report.files_skipped["app.rb"] == "no language server for ruby"
    assert report.counts["error"] == 1


def test_budgets_are_enforced_and_flagged() -> None:
    files = [f"f{i}.py" for i in range(10)]
    report = collect_diagnostics(files, lambda _rel: [_diag(0)], max_files=3)
    assert report.files_inspected == 3
    assert report.truncated

    report = collect_diagnostics(files, lambda _rel: [_diag(0), _diag(1)], max_problems=3)
    assert sum(report.counts.values()) == 3
    assert report.truncated


def test_file_selection_rejects_unknown_scope_and_filters_non_source(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "notes.bin").write_bytes(b"\x00\x01")
    with pytest.raises(ValueError, match="scope"):
        files_to_inspect(tmp_path, "everything")

    files = files_to_inspect(tmp_path, "project")
    assert files == ["a.py"]
    assert files_to_inspect(tmp_path, "project", path_glob="*.go") == []


@pytest.mark.parametrize("language_server", [Language.PYTHON], indirect=True)
def test_sweep_against_a_real_language_server(language_server) -> None:
    """End-to-end: real diagnostics from the shared Python test repo."""
    from serena.symbol import LanguageServerSymbolRetriever

    ls = language_server.language_server
    root = Path(ls.repository_root_path)

    def fetch(rel: str) -> list[dict]:
        return list(ls.request_text_document_diagnostics(rel) or [])

    files = files_to_inspect(root, "project")
    assert "test_repo/diagnostics_sample.py" in files

    report = collect_diagnostics(["test_repo/diagnostics_sample.py"], fetch, scope="project")
    assert report.files_inspected == 1
    assert sum(report.counts.values()) > 0, "the sample file is meant to contain problems"
    problem = report.problems["test_repo/diagnostics_sample.py"][0]
    assert problem["line"] >= 1
    assert problem["message"]
    assert problem["severity"] in {"error", "warning", "information", "hint"}
    assert LanguageServerSymbolRetriever  # the tool uses this retriever in production


def test_diagnostics_tool_is_registered() -> None:
    from codescope.cli import register_codescope_tools

    register_codescope_tools()
    from serena.tools import ToolRegistry

    assert "get_project_diagnostics" in ToolRegistry().get_tool_names()


def test_invalid_severity_is_rejected_not_reported_as_a_clean_project() -> None:
    """Every file failing must not read as 'no problems found'."""
    from codescope.tools.diagnostics_tools import GetProjectDiagnosticsTool

    tool = object.__new__(GetProjectDiagnosticsTool)
    with pytest.raises(ValueError, match="min_severity"):
        GetProjectDiagnosticsTool.apply(tool, min_severity=9)
