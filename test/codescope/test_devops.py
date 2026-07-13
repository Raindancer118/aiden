"""Tests for the dev-ops layer: test runner, scaffolding, and git operations."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from codescope.devops import github, scaffold, testrunner, vcs

# -- test runner ----------------------------------------------------------- #


def test_detect_pytest(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\ntestpaths = ['tests']\n")
    (tmp_path / "tests").mkdir()
    fw = testrunner.detect_framework(tmp_path)
    assert fw is not None and fw.name == "pytest"


def test_detect_cargo_and_go(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
    framework = testrunner.detect_framework(tmp_path)

    assert framework is not None
    assert framework.name == "cargo"


def test_detect_node_test_directory_as_node(tmp_path: Path) -> None:
    project = Path(scaffold.create_project("node", "demo", tmp_path, git_init=False).path)

    framework = testrunner.detect_framework(project)

    assert framework is not None
    assert framework.name == "node:npm"


@pytest.mark.parametrize(
    ("wrapper", "expected_name", "expected_command"),
    [
        ("gradlew", "gradle", ["./gradlew.bat", "test"]),
        ("mvnw", "maven", ["./mvnw.bat", "-q", "test"]),
    ],
)
def test_detects_windows_jvm_wrappers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, wrapper: str, expected_name: str, expected_command: list[str]
) -> None:
    (tmp_path / wrapper).write_text("")
    (tmp_path / f"{wrapper}.bat").write_text("")
    monkeypatch.setattr(testrunner.sys, "platform", "win32")

    framework = testrunner.detect_framework(tmp_path)

    assert framework is not None
    assert framework.name == expected_name
    assert framework.command == expected_command


def test_node_detection_requires_test_script(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"dependencies": {"test": "1.0.0"}}')

    assert testrunner.detect_framework(tmp_path) is None


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


def test_run_tests_forwards_npm_args_after_separator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"scripts": {"test": "vitest"}}')
    commands: list[list[str]] = []

    monkeypatch.setattr(shutil, "which", lambda command: f"/usr/bin/{command}")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, returncode=0, stdout="tests passed", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = testrunner.run_tests(tmp_path, extra_args=["--runInBand"])

    assert result.ok is True
    assert commands == [["npm", "test", "--silent", "--", "--runInBand"]]


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


@pytest.mark.parametrize("name", [".", "..", ".hidden", "trailing."])
def test_create_project_refuses_unsafe_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        scaffold.create_project("python", name, tmp_path, git_init=False)


def test_create_project_refuses_symlink_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "demo").symlink_to(outside, target_is_directory=True)

    with pytest.raises(FileExistsError):
        scaffold.create_project("python", "demo", destination, git_init=False)

    assert list(outside.iterdir()) == []


def test_python_project_with_numeric_name_has_valid_module(tmp_path: Path) -> None:
    project = Path(scaffold.create_project("python", "123-demo", tmp_path, git_init=False).path)

    test_source = (project / "tests" / "test_main.py").read_text()

    assert (project / "src" / "_123_demo" / "main.py").is_file()
    compile(test_source, "test_main.py", "exec")


def test_node_package_name_is_normalized_to_lowercase(tmp_path: Path) -> None:
    project = Path(scaffold.create_project("node", "MyApp", tmp_path, git_init=False).path)

    assert '"name": "myapp"' in (project / "package.json").read_text()


# -- vcs ------------------------------------------------------------------- #


def test_sanitize_message_strips_ai_attribution() -> None:
    msg = "feat: add thing\n\nBody line.\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n🤖 Generated with Claude Code\n"
    cleaned, sanitized = vcs.sanitize_message(msg)
    assert sanitized is True
    assert "claude" not in cleaned.lower()
    assert "anthropic" not in cleaned.lower()
    assert "feat: add thing" in cleaned


@pytest.mark.parametrize(
    "attribution",
    [
        "Co-Authored-By: OpenAI Codex <codex@openai.com>",
        "Assisted-By: GitHub Copilot <copilot@github.com>",
        "Generated by ChatGPT",
    ],
)
def test_sanitize_message_strips_other_ai_attribution(attribution: str) -> None:
    cleaned, sanitized = vcs.sanitize_message(f"fix: keep human subject\n\n{attribution}\n")

    assert sanitized is True
    assert cleaned == "fix: keep human subject\n"


def test_commit_does_not_continue_after_git_add_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_git(args: list[str], cwd: str | Path, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(["git", *args], returncode=1, stdout="", stderr="add failed")

    monkeypatch.setattr(vcs, "_git", fake_git)

    result = vcs.commit(tmp_path, "fix: safe staging", add_all=True)

    assert result.committed is False
    assert result.output == "add failed"
    assert calls == [["add", "-A", "--", "."]]


def test_status_preserves_columns_and_rename_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = " M worktree.py\0M  staged.py\0R  renamed.py\0original.py\0"
    monkeypatch.setattr(
        vcs,
        "_git",
        lambda *args, **kwargs: subprocess.CompletedProcess(["git"], returncode=0, stdout=output, stderr=""),
    )

    assert vcs.status(tmp_path) == [
        {"status": "M", "index_status": "", "worktree_status": "M", "path": "worktree.py"},
        {"status": "M", "index_status": "M", "worktree_status": "", "path": "staged.py"},
        {"status": "R", "index_status": "R", "worktree_status": "", "path": "renamed.py", "original_path": "original.py"},
    ]


def test_status_rejects_non_repository(tmp_path: Path) -> None:
    with pytest.raises(vcs.GitError, match="Could not read git status"):
        vcs.status(tmp_path)


def test_changelog_rejects_option_like_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not start"):
        vcs.generate_changelog(tmp_path, to_ref="--all")


def test_added_blocks_rejects_nonpositive_minimum() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        vcs._parse_added_blocks("", min_lines=0)


# -- GitHub CLI ------------------------------------------------------------ #


def test_gh_timeout_is_reported_as_gh_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda command: "/usr/bin/gh")

    def time_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="gh", timeout=120)

    monkeypatch.setattr(subprocess, "run", time_out)

    with pytest.raises(github.GhError, match="timed out"):
        github.pr_list(Path.cwd())


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
