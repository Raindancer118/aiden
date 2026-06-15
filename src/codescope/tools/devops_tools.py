"""Dev-ops tools: test runner, scaffolding, git, and GitHub operations."""

from __future__ import annotations

import os
import shlex
from dataclasses import asdict
from pathlib import Path

from serena.tools import Tool, ToolMarkerDoesNotRequireActiveProject

from codescope.devops import github as gh
from codescope.devops import scaffold, testrunner, vcs

# --------------------------------------------------------------------------- #
# Test runner
# --------------------------------------------------------------------------- #


class RunTestsTool(Tool):
    """
    Detect and run the project's test suite, returning a structured result.

    Auto-detects the framework (pytest, gradle, maven, npm/yarn/pnpm, cargo,
    go) from project files, runs it, and reports the exit code, parsed
    pass/fail counts where available, and the tail of the output.
    """

    def apply(self, extra_args: str = "") -> str:
        """
        :param extra_args: extra CLI args appended to the test command
            (e.g. "-k test_login -q" for pytest). Optional.
        :return: JSON with framework, command, exit_code, ok, summary, output_tail.
        """
        args = shlex.split(extra_args) if extra_args else None
        result = testrunner.run_tests(Path(self.get_project_root()), extra_args=args)
        return self._to_json(asdict(result))


class DetectTestFrameworkTool(Tool):
    """Report which test framework Codescope detects for the active project."""

    def apply(self) -> str:
        """:return: JSON with the detected framework, command, and reason (or a note)."""
        fw = testrunner.detect_framework(Path(self.get_project_root()))
        if fw is None:
            return self._to_json({"detected": False, "note": "No known test framework detected."})
        return self._to_json({"detected": True, **asdict(fw)})


# --------------------------------------------------------------------------- #
# Scaffolding
# --------------------------------------------------------------------------- #


class CreateProjectTool(Tool, ToolMarkerDoesNotRequireActiveProject):
    """
    Scaffold a new project from a template.

    Supported stacks: 'python' (uv + hatchling + pytest, src layout),
    'node' (TypeScript + vitest), 'rust' (cargo). Creates the source layout, a
    test scaffold, .gitignore, and README, and initializes a git repository.
    Never overwrites a non-empty target directory.
    """

    def apply(self, stack: str, name: str, dest_dir: str = "", git_init: bool = True) -> str:
        """
        :param stack: one of 'python', 'node', 'rust'.
        :param name: the project (and directory) name.
        :param dest_dir: parent directory for the new project; defaults to the
            current working directory.
        :param git_init: initialize a git repository in the new project.
        :return: JSON with the stack, created path, files, and git status.
        """
        dest = dest_dir or os.getcwd()
        result = scaffold.create_project(stack, name, dest, git_init=git_init)
        return self._to_json(asdict(result))


# --------------------------------------------------------------------------- #
# Git
# --------------------------------------------------------------------------- #


class GitStatusTool(Tool):
    """Show the git working-tree status of the active project (porcelain)."""

    def apply(self) -> str:
        """:return: JSON list of {status, path} entries."""
        return self._to_json(vcs.status(self.get_project_root()))


class GitDiffTool(Tool):
    """Show the git diff of the active project (unstaged by default)."""

    def apply(self, staged: bool = False) -> str:
        """
        :param staged: show the staged diff instead of the unstaged one.
        :return: the diff text (truncated for very large diffs).
        """
        return vcs.diff(self.get_project_root(), staged=staged)


class GitCommitTool(Tool):
    """
    Create a git commit in the active project.

    The commit message is sanitized to remove any AI-assistant attribution
    (Co-Authored-By / "Generated with ..." lines), so commits carry only the
    repository's configured author identity.
    """

    def apply(self, message: str, add_all: bool = False) -> str:
        """
        :param message: the commit message (conventional commits recommended).
        :param add_all: stage all changes (git add -A) before committing.
        :return: JSON with committed flag, final message, git output, and whether
            the message was sanitized.
        """
        return self._to_json(asdict(vcs.commit(self.get_project_root(), message, add_all=add_all)))


class GenerateChangelogTool(Tool):
    """
    Generate a changelog from git history, grouped by Conventional Commit type.

    Defaults to the range from the latest tag (or the first commit) to HEAD.
    """

    def apply(self, from_ref: str = "", to_ref: str = "HEAD") -> str:
        """
        :param from_ref: starting ref (default: latest tag or root commit).
        :param to_ref: ending ref (default: HEAD).
        :return: JSON with the resolved range, grouped sections, and rendered markdown.
        """
        cl = vcs.generate_changelog(self.get_project_root(), from_ref or None, to_ref)
        data = asdict(cl)
        data["markdown"] = cl.to_markdown()
        return self._to_json(data)


# --------------------------------------------------------------------------- #
# GitHub (via gh CLI)
# --------------------------------------------------------------------------- #


class GithubRepoCreateTool(Tool, ToolMarkerDoesNotRequireActiveProject):
    """Create a GitHub repository via the gh CLI (private by default)."""

    def apply(self, name: str, private: bool = True, description: str = "") -> str:
        """
        :param name: repository name (optionally 'owner/name').
        :param private: create as private (default true).
        :param description: optional repository description.
        :return: JSON with the created repository URL or an error.
        """
        try:
            url = gh.repo_create(name, private=private, description=description)
            return self._to_json({"created": True, "repo": url})
        except gh.GhError as e:
            return self._to_json({"created": False, "error": str(e)})


class GithubPrListTool(Tool):
    """List pull requests for the active project's GitHub repository."""

    def apply(self, state: str = "open", limit: int = 30) -> str:
        """
        :param state: 'open', 'closed', 'merged', or 'all'.
        :param limit: maximum number of PRs to return.
        :return: JSON list of pull requests, or an error.
        """
        try:
            return self._to_json(gh.pr_list(self.get_project_root(), state=state, limit=limit))
        except gh.GhError as e:
            return self._to_json({"error": str(e)})


class GithubPrCreateTool(Tool):
    """Create a pull request for the active project's repository."""

    def apply(self, title: str, body: str = "", base: str = "", draft: bool = False) -> str:
        """
        :param title: PR title.
        :param body: PR description.
        :param base: base branch to merge into (default: repo default).
        :param draft: create as a draft PR.
        :return: JSON with the created PR URL or an error.
        """
        try:
            url = gh.pr_create(self.get_project_root(), title=title, body=body, base=base or None, draft=draft)
            return self._to_json({"created": True, "pr": url})
        except gh.GhError as e:
            return self._to_json({"created": False, "error": str(e)})


class GithubIssueListTool(Tool):
    """List issues for the active project's GitHub repository."""

    def apply(self, state: str = "open", limit: int = 30) -> str:
        """
        :param state: 'open', 'closed', or 'all'.
        :param limit: maximum number of issues to return.
        :return: JSON list of issues, or an error.
        """
        try:
            return self._to_json(gh.issue_list(self.get_project_root(), state=state, limit=limit))
        except gh.GhError as e:
            return self._to_json({"error": str(e)})


class GithubIssueCreateTool(Tool):
    """Create an issue in the active project's GitHub repository."""

    def apply(self, title: str, body: str = "", labels: str = "") -> str:
        """
        :param title: issue title.
        :param body: issue body.
        :param labels: comma-separated label names (optional).
        :return: JSON with the created issue URL or an error.
        """
        label_list = [s.strip() for s in labels.split(",") if s.strip()]
        try:
            url = gh.issue_create(self.get_project_root(), title=title, body=body, labels=label_list)
            return self._to_json({"created": True, "issue": url})
        except gh.GhError as e:
            return self._to_json({"created": False, "error": str(e)})


class GithubActionsStatusTool(Tool):
    """List recent GitHub Actions workflow runs for the active project."""

    def apply(self, limit: int = 15) -> str:
        """
        :param limit: maximum number of workflow runs to return.
        :return: JSON list of recent runs (status/conclusion/branch/url), or an error.
        """
        try:
            return self._to_json(gh.run_list(self.get_project_root(), limit=limit))
        except gh.GhError as e:
            return self._to_json({"error": str(e)})


class GithubReleaseCreateTool(Tool):
    """Create a GitHub release for the active project's repository."""

    def apply(self, tag: str, title: str = "", notes: str = "") -> str:
        """
        :param tag: the release tag (e.g. 'v1.2.0').
        :param title: release title (default: the tag).
        :param notes: release notes (markdown).
        :return: JSON with the release URL or an error.
        """
        try:
            url = gh.release_create(self.get_project_root(), tag=tag, title=title, notes=notes)
            return self._to_json({"created": True, "release": url})
        except gh.GhError as e:
            return self._to_json({"created": False, "error": str(e)})
