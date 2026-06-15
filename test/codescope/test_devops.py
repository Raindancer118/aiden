"""Tests for the dev-ops layer: test runner, scaffolding, and git operations."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from codescope.devops import scaffold, testrunner, vcs


# -- test runner ----------------------------------------------------------- #


def test_detect_pytest(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\ntestpaths = ['tests']\n")
    (tmp_path / "tests").mkdir()
    fw = testrunner.detect_framework(tmp_path)
    assert fw is not None and fw.name == "pytest"


def test_detect_cargo_and_go(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
    assert testrunner.detect_framework(tmp_path).name == "cargo"


def test_detect_none(tmp_path: Path) -> None:
    assert testrunner.detect_framework(tmp_path) is None


@pytest.mark.skipif(shutil.which("pytest") is None, reason="pytest not on PATH")
def test_run_pytest_project(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\ntestpaths = ['tests']\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_sample.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")
    result = testrunner.run_tests(tmp_path)
    assert result.framework == "pytest"
    assert result.ok
    assert result.summary.get("passed", 0) >= 1


# -- scaffolding ----------------------------------------------------------- #


@pytest.mark.parametrize("stack", ["python", "node", "rust"])
def test_create_project(tmp_path: Path, stack: str) -> None:
    result = scaffold.create_project(stack, "demo", tmp_path, git_init=False)
    assert result.stack == stack
    assert Path(result.path).is_dir()
    assert any("README.md" in f for f in result.files)
    assert (Path(result.path) / "README.md").exists()


def test_create_project_refuses_nonempty(tmp_path: Path) -> None:
    target = tmp_path / "demo"
    target.mkdir()
    (target / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        scaffold.create_project("python", "demo", tmp_path)


def test_create_project_unknown_stack(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        scaffold.create_project("haskell", "demo", tmp_path)


# -- vcs ------------------------------------------------------------------- #


def test_sanitize_message_strips_ai_attribution() -> None:
    msg = (
        "feat: add thing\n\n"
        "Body line.\n\n"
        "Co-Authored-By: Claude <noreply@anthropic.com>\n"
        "🤖 Generated with Claude Code\n"
    )
    cleaned, sanitized = vcs.sanitize_message(msg)
    assert sanitized is True
    assert "claude" not in cleaned.lower()
    assert "anthropic" not in cleaned.lower()
    assert "feat: add thing" in cleaned


def test_commit_and_changelog(tmp_path: Path) -> None:
    pytest.importorskip("pygit2")
    import pygit2

    pygit2.init_repository(str(tmp_path))
    (tmp_path / "f.py").write_text("def f():\n    pass\n")
    result = vcs.commit(
        tmp_path,
        "feat: initial\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n",
        add_all=True,
    )
    assert result.committed, result.output
    assert result.sanitized is True
    assert "claude" not in result.message.lower()

    cl = vcs.generate_changelog(tmp_path)
    md = cl.to_markdown()
    assert "Features" in md
    assert "initial" in md


def test_devops_tools_registered() -> None:
    from codescope.cli import register_codescope_tools

    register_codescope_tools()
    from serena.tools import ToolRegistry

    names = ToolRegistry().get_tool_names()
    for t in (
        "run_tests",
        "detect_test_framework",
        "create_project",
        "git_status",
        "git_diff",
        "git_commit",
        "generate_changelog",
        "github_repo_create",
        "github_pr_list",
        "github_pr_create",
        "github_issue_list",
        "github_issue_create",
        "github_actions_status",
        "github_release_create",
    ):
        assert t in names, f"missing tool: {t}"
