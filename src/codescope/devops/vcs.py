"""Git operations: status, diff, commit, and changelog generation.

Commit messages are sanitized to strip any AI-assistant attribution
(Co-Authored-By / "Generated with ..." lines mentioning Claude/Anthropic),
enforcing the project rule that commits carry only the human author's identity.
"""

from __future__ import annotations

import re
import subprocess
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

_ATTRIBUTION_RE = re.compile(
    r"(?im)^\s*(co-authored-by:.*(claude|anthropic).*|.*generated with.*(claude|claude code).*|🤖.*)$"
)

_CONVENTIONAL_TYPES = OrderedDict(
    [
        ("feat", "Features"),
        ("fix", "Fixes"),
        ("perf", "Performance"),
        ("refactor", "Refactoring"),
        ("docs", "Documentation"),
        ("test", "Tests"),
        ("build", "Build"),
        ("ci", "CI"),
        ("chore", "Chores"),
    ]
)


@dataclass(slots=True)
class CommitResult:
    committed: bool
    message: str
    output: str
    sanitized: bool = False


@dataclass(slots=True)
class Changelog:
    from_ref: str
    to_ref: str
    sections: dict[str, list[str]] = field(default_factory=dict)
    other: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"# Changelog ({self.from_ref}..{self.to_ref})", ""]
        for title, entries in self.sections.items():
            if not entries:
                continue
            lines.append(f"## {title}")
            lines += [f"- {e}" for e in entries]
            lines.append("")
        if self.other:
            lines.append("## Other")
            lines += [f"- {e}" for e in self.other]
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def _git(args: list[str], cwd: str | Path, *, input_text: str | None = None, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], check=False, cwd=str(cwd), capture_output=True, text=True, input=input_text, timeout=timeout)


def sanitize_message(message: str) -> tuple[str, bool]:
    cleaned = _ATTRIBUTION_RE.sub("", message)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip() + "\n"
    return cleaned, cleaned.strip() != message.strip()


def status(cwd: str | Path) -> list[dict]:
    proc = _git(["status", "--porcelain=v1"], cwd)
    entries = []
    for line in proc.stdout.splitlines():
        if len(line) < 3:
            continue
        entries.append({"status": line[:2].strip(), "path": line[3:]})
    return entries


def diff(cwd: str | Path, *, staged: bool = False, max_chars: int = 20000) -> str:
    args = ["diff", "--staged"] if staged else ["diff"]
    out = _git(args, cwd).stdout
    return out[:max_chars]


@dataclass(slots=True)
class DiffBlock:
    """A run of contiguous added lines in a unified diff (new-file coordinates)."""

    path: str
    start_line: int
    end_line: int
    text: str


_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _parse_added_blocks(diff_text: str, min_lines: int) -> list[DiffBlock]:
    """Extract contiguous added-line blocks from unified ``git diff`` output."""
    blocks: list[DiffBlock] = []
    path: str | None = None
    new_line = 0
    buf: list[str] = []
    buf_start = 0

    def flush() -> None:
        nonlocal buf
        if path and len(buf) >= min_lines:
            blocks.append(
                DiffBlock(path=path, start_line=buf_start, end_line=buf_start + len(buf) - 1, text="\n".join(buf))
            )
        buf = []

    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            flush()
            target = line[4:].strip()
            path = None if target == "/dev/null" else target[2:] if target.startswith("b/") else target
            continue
        if line.startswith(("--- ", "diff ", "index ")):
            flush()
            continue
        m = _HUNK_RE.match(line)
        if m:
            flush()
            new_line = int(m.group(1))
            continue
        if line.startswith("+"):
            if not buf:
                buf_start = new_line
            buf.append(line[1:])
            new_line += 1
        elif line.startswith("-"):
            flush()
        else:  # context line (leading space) or blank
            flush()
            new_line += 1
    flush()
    return blocks


def added_blocks(cwd: str | Path, *, staged: bool = False, min_lines: int = 1) -> list[DiffBlock]:
    """Blocks of newly added lines in the working tree (or staged) diff.

    Note: untracked files only appear once staged (``git add``) or marked
    intent-to-add (``git add -N``).
    """
    return _parse_added_blocks(diff(cwd, staged=staged, max_chars=200000), min_lines)


def commit(cwd: str | Path, message: str, *, add_all: bool = False) -> CommitResult:
    cleaned, was_sanitized = sanitize_message(message)
    if add_all:
        _git(["add", "-A"], cwd)
    proc = _git(["commit", "-m", cleaned], cwd)
    return CommitResult(
        committed=proc.returncode == 0,
        message=cleaned,
        output=(proc.stdout + proc.stderr).strip(),
        sanitized=was_sanitized,
    )


def log(cwd: str | Path, *, n: int = 20) -> list[str]:
    proc = _git(["log", f"-{n}", "--pretty=%h %s"], cwd)
    return proc.stdout.splitlines()


def generate_changelog(cwd: str | Path, from_ref: str | None = None, to_ref: str = "HEAD") -> Changelog:
    if from_ref is None:
        tag = _git(["describe", "--tags", "--abbrev=0"], cwd)
        from_ref = tag.stdout.strip() or None  # no tag -> log full history below
    rng = f"{from_ref}..{to_ref}" if from_ref else to_ref
    proc = _git(["log", rng, "--pretty=%s"], cwd)
    sections: dict[str, list[str]] = {title: [] for title in _CONVENTIONAL_TYPES.values()}
    other: list[str] = []
    for subject in proc.stdout.splitlines():
        m = re.match(r"^(\w+)(?:\([^)]*\))?!?:\s*(.*)$", subject)
        if m and m.group(1).lower() in _CONVENTIONAL_TYPES:
            sections[_CONVENTIONAL_TYPES[m.group(1).lower()]].append(m.group(2))
        else:
            other.append(subject)
    return Changelog(from_ref=from_ref or "", to_ref=to_ref, sections=sections, other=other)
