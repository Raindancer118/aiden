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


class GhError(RuntimeError):
    pass


def _gh(args: list[str], *, cwd: str | Path | None = None, parse_json: bool = False, timeout: int = 120):
    if shutil.which("gh") is None:
        raise GhError("GitHub CLI 'gh' is not installed.")
    proc = subprocess.run(
        ["gh", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise GhError((proc.stderr or proc.stdout or "gh command failed").strip())
    out = proc.stdout.strip()
    if parse_json:
        return json.loads(out) if out else []
    return out


def repo_create(name: str, *, private: bool = True, description: str = "", source: str | Path | None = None, push: bool = False) -> str:
    args = ["repo", "create", name, "--private" if private else "--public"]
    if description:
        args += ["--description", description]
    if source:
        args += ["--source", str(source)]
        if push:
            args += ["--push"]
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
    args = ["release", "create", tag]
    if title:
        args += ["--title", title]
    args += ["--notes", notes]
    return _gh(args, cwd=cwd)
