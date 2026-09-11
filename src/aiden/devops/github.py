"""Thin wrapper around the GitHub CLI (``gh``).

Codebase-aware GitHub operations (repos, PRs, issues, releases, Actions). All
calls shell out to an authenticated ``gh`` and return parsed JSON where the CLI
supports it. Requires ``gh`` to be installed and logged in.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Literal, cast, overload


class GhError(RuntimeError):
    pass


@overload
def _gh(args: list[str], *, cwd: str | Path | None = None, parse_json: Literal[False] = False, timeout: int = 120) -> str: ...


@overload
def _gh(args: list[str], *, cwd: str | Path | None = None, parse_json: Literal[True], timeout: int = 120) -> list[dict[str, object]]: ...


def _gh(args: list[str], *, cwd: str | Path | None = None, parse_json: bool = False, timeout: int = 120) -> str | list[dict[str, object]]:
    executable = shutil.which("gh")
    if executable is None:
        raise GhError("GitHub CLI 'gh' is not installed.")
    try:
        proc = subprocess.run(
            [executable, *args],
            check=False,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise GhError(f"GitHub CLI command timed out after {timeout}s.") from error
    except OSError as error:
        raise GhError(f"Could not start GitHub CLI: {error}") from error
    if proc.returncode != 0:
        raise GhError((proc.stderr or proc.stdout or "gh command failed").strip())
    out = proc.stdout.strip()
    if parse_json:
        try:
            data = json.loads(out) if out else []
        except json.JSONDecodeError as error:
            raise GhError("GitHub CLI returned invalid JSON.") from error
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise GhError("GitHub CLI returned an unexpected JSON response.")
        return cast(list[dict[str, object]], data)
    return out


def repo_create(name: str, *, private: bool = True, description: str = "", source: str | Path | None = None, push: bool = False) -> str:
    if not name or "\0" in name:
        raise GhError("Repository name must be non-empty and must not contain NUL characters.")
    args = ["repo", "create", "--private" if private else "--public"]
    if description:
        args += ["--description", description]
    if source:
        args += ["--source", str(source)]
        if push:
            args += ["--push"]
    # End option parsing before the user-controlled positional value. A name
    # such as `--help` or `--source=...` must never alter the gh operation.
    args += ["--", name]
    return _gh(args)


def pr_list(cwd: str | Path, *, state: str = "open", limit: int = 30) -> list[dict]:
    return _gh(
        ["pr", "list", "--state", state, "--limit", str(limit), "--json", "number,title,author,state,headRefName,url"],
        cwd=cwd,
        parse_json=True,
    )


def pr_create(cwd: str | Path, *, title: str, body: str = "", base: str | None = None, draft: bool = False) -> str:
    args = ["pr", "create", "--title", title, "--body", body]
    if base:
        args += ["--base", base]
    if draft:
        args += ["--draft"]
    return _gh(args, cwd=cwd)


def issue_list(cwd: str | Path, *, state: str = "open", limit: int = 30) -> list[dict]:
    return _gh(
        ["issue", "list", "--state", state, "--limit", str(limit), "--json", "number,title,author,state,labels,url"],
        cwd=cwd,
        parse_json=True,
    )


def issue_create(cwd: str | Path, *, title: str, body: str = "", labels: list[str] | None = None) -> str:
    args = ["issue", "create", "--title", title, "--body", body]
    for label in labels or []:
        args += ["--label", label]
    return _gh(args, cwd=cwd)


def run_list(cwd: str | Path, *, limit: int = 15) -> list[dict]:
    return _gh(
        ["run", "list", "--limit", str(limit), "--json", "displayTitle,status,conclusion,workflowName,headBranch,createdAt,url"],
        cwd=cwd,
        parse_json=True,
    )


def release_create(cwd: str | Path, *, tag: str, title: str = "", notes: str = "") -> str:
    if not tag or "\0" in tag:
        raise GhError("Release tag must be non-empty and must not contain NUL characters.")
    args = ["release", "create"]
    if title:
        args += ["--title", title]
    args += ["--notes", notes]
    args += ["--", tag]
    return _gh(args, cwd=cwd)
