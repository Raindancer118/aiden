"""Sweeping diagnostics across many files.

Kept separate from the tool so the sweep - file selection, grouping, the
budgets, and the difference between "clean" and "could not check" - can be
exercised directly against a language server without an agent in the loop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from codescope.index.indexer import Indexer
from codescope.index.languages import spec_for_path

log = logging.getLogger(__name__)

SEVERITY_NAMES = {1: "error", 2: "warning", 3: "information", 4: "hint"}

#: Files inspected per call unless the caller raises it. Each one costs a
#: language-server round trip, and an agent cannot act on 500 findings.
DEFAULT_MAX_FILES = 60
DEFAULT_MAX_PROBLEMS = 200

#: Fetches the diagnostics of one file; raises when it cannot be checked.
DiagnosticsFetcher = Callable[[str], Iterable[dict]]


@dataclass(slots=True)
class DiagnosticsReport:
    scope: str
    files_inspected: int
    #: path -> why it could not be checked (no language server, server error).
    #: A file listed here is *unknown*, not clean.
    files_skipped: dict[str, str] = field(default_factory=dict)
    truncated: bool = False
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(SEVERITY_NAMES.values(), 0))
    problems: dict[str, list[dict]] = field(default_factory=dict)


def collect_diagnostics(
    files: list[str],
    fetch: DiagnosticsFetcher,
    *,
    scope: str = "changed",
    max_files: int = DEFAULT_MAX_FILES,
    max_problems: int = DEFAULT_MAX_PROBLEMS,
) -> DiagnosticsReport:
    """Run ``fetch`` over ``files`` and group the findings by file.

    A file whose fetch raises is recorded in ``files_skipped`` rather than
    contributing zero problems: an agent must be able to tell "nothing wrong
    here" from "nobody looked".
    """
    truncated = len(files) > max_files
    selected = files[:max_files]
    report = DiagnosticsReport(scope=scope, files_inspected=0)
    total = 0

    for rel in selected:
        if total >= max_problems:
            truncated = True
            break
        try:
            diagnostics = list(fetch(rel))
        except Exception as e:
            report.files_skipped[rel] = str(e).splitlines()[0][:200]
            continue
        report.files_inspected += 1
        for diagnostic in diagnostics:
            severity = SEVERITY_NAMES.get(diagnostic.get("severity", 1), "error")
            report.counts[severity] += 1
            report.problems.setdefault(rel, []).append(
                {
                    "line": diagnostic.get("range", {}).get("start", {}).get("line", 0) + 1,
                    "severity": severity,
                    "source": diagnostic.get("source", ""),
                    "code": str(diagnostic.get("code", "")),
                    "message": (diagnostic.get("message") or "").strip()[:500],
                }
            )
            total += 1
            if total >= max_problems:
                truncated = True
                break

    report.truncated = truncated
    return report


def files_to_inspect(root: Path, scope: str, path_glob: str = "") -> list[str]:
    """Relative source paths to inspect, in a stable order.

    ``changed`` is the default because it is the check worth running after an
    edit; it falls back to the full set outside a git repository.
    """
    if scope not in ("changed", "project"):
        raise ValueError("scope must be 'changed' or 'project'")

    candidates: list[str] | None = None
    if scope == "changed":
        candidates = _changed_files(root)
        if candidates is None:
            log.info("Not a git repository; falling back to a project-wide diagnostics sweep.")
    if candidates is None:
        candidates = _indexed_files(root)

    files = [p for p in candidates if spec_for_path(p) is not None and (root / p).is_file()]
    if path_glob:
        from codescope.index.search import _matches_glob

        files = [p for p in files if _matches_glob(p, path_glob)]
    return sorted(dict.fromkeys(files))


def _changed_files(root: Path) -> list[str] | None:
    from codescope.index.incremental import git_changes

    changes = git_changes(root)
    if changes is None:
        return None
    changed, _deleted = changes
    return list(changed)


def _indexed_files(root: Path) -> list[str]:
    from codescope.index.store import IndexStore

    indexer = Indexer(root)
    if not indexer.db_path.exists():
        return [rel for _abs, rel in indexer.iter_source_files()]
    with IndexStore(indexer.db_path) as store:
        return sorted(store.indexed_paths())
