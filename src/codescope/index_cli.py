r"""``codescope index`` - build and inspect the hybrid index from the shell.

Indexing is the one long, memory-hungry operation Codescope performs, and it
had no entry point outside the MCP server: a first index of a large repository
therefore ran inside the server process, where it competes with the editor and
cannot be given a resource budget. Running it here lets you cap it::

    systemd-run --user --scope -p MemoryMax=2G -p CPUWeight=20 \\
        nice -n 19 ionice -c3 codescope index build --project .

Vectors are written batch by batch, so a run killed by that cap (or by
Ctrl-C) keeps everything it had already embedded; running ``build`` again
resumes with the symbols that are still missing a vector.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import click

from codescope.index.embed import DEFAULT_BATCH_SIZE, get_embedder
from codescope.index.incremental import sync_incremental
from codescope.index.indexer import Indexer
from codescope.index.search import SearchEngine

log = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _report(payload: object, as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps(payload, indent=2, default=str))
        return
    click.echo(json.dumps(payload, indent=2, default=str) if not isinstance(payload, str) else payload)


@click.group(name="index")
def index_group() -> None:
    """Build, refresh and inspect the Codescope hybrid index."""


_project_option = click.option(
    "--project",
    "project",
    default=".",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    show_default=True,
    help="Project root to index.",
)
_json_option = click.option("--json", "as_json", is_flag=True, help="Print machine-readable JSON.")
_verbose_option = click.option("-v", "--verbose", is_flag=True, help="Log progress to stderr.")


@index_group.command("build")
@_project_option
@click.option("--force", is_flag=True, help="Reparse every file, ignoring unchanged content hashes.")
@click.option("--no-embeddings", is_flag=True, help="Lexical index only; skip the (slow) vector build.")
@click.option("--embedder", default="auto", show_default=True, help="Embedder to use: auto, fastembed, gemini, hashing.")
@click.option(
    "--batch-size",
    default=DEFAULT_BATCH_SIZE,
    show_default=True,
    type=click.IntRange(1, 512),
    help="Upper bound on documents per forward pass. For memory, set CODESCOPE_EMBED_BATCH_COST instead: that budget, not this cap, is what sets the peak.",
)
@_json_option
@_verbose_option
def build(project: Path, force: bool, no_embeddings: bool, embedder: str, batch_size: int, as_json: bool, verbose: bool) -> None:
    """Build or refresh the index (resumable; safe to re-run)."""
    _configure_logging(verbose)
    indexer = Indexer(project, embedder_name=embedder)
    resolved = None
    if not no_embeddings:
        # Resolve through get_embedder so the batch size reaches the backend,
        # but only hand the instance to reindex() when the user *named* a
        # backend. Passing an auto-resolved one would mark the choice as
        # explicit and let a silently degraded hashing fallback rebuild a
        # real semantic index (see Indexer._update_embeddings).
        candidate = get_embedder(embedder, batch_size=batch_size) if embedder == "fastembed" else get_embedder(embedder)
        click.echo(f"Embedding with {candidate.id} (batch size {candidate.batch_size})", err=True)
        if embedder.lower() not in ("", "auto"):
            resolved = candidate

    started = time.time()
    report = indexer.reindex(force=force, embeddings=not no_embeddings, embedder=resolved)
    payload = {
        "indexed": report.indexed,
        "skipped_unchanged": report.skipped_unchanged,
        "removed": report.removed,
        "errors": report.errors,
        "duration_s": report.duration_s,
        "totals": asdict(report.stats),
    }
    _report(payload, as_json)
    if not as_json:
        click.echo(f"Done in {time.time() - started:.1f}s", err=True)
    sys.exit(1 if report.errors else 0)


@index_group.command("sync")
@_project_option
@click.option("--no-embeddings", is_flag=True, help="Skip the vector update.")
@_json_option
@_verbose_option
def sync(project: Path, no_embeddings: bool, as_json: bool, verbose: bool) -> None:
    """Update only what git reports as changed (falls back to a full pass)."""
    _configure_logging(verbose)
    report = sync_incremental(Indexer(project), embeddings=not no_embeddings)
    _report(
        {
            "indexed": report.indexed,
            "skipped_unchanged": report.skipped_unchanged,
            "removed": report.removed,
            "errors": report.errors,
            "duration_s": report.duration_s,
            "totals": asdict(report.stats),
        },
        as_json,
    )
    sys.exit(1 if report.errors else 0)


@index_group.command("status")
@_project_option
@_json_option
def status(project: Path, as_json: bool) -> None:
    """Report what the index currently contains."""
    indexer = Indexer(project)
    stats = asdict(indexer.status())
    stats["db_path"] = str(indexer.db_path)
    stats["db_size_bytes"] = indexer.db_path.stat().st_size if indexer.db_path.exists() else 0
    _report(stats, as_json)


@index_group.command("doctor")
@_project_option
@_json_option
def doctor(project: Path, as_json: bool) -> None:
    """Say what the index can answer right now, and what to do about the rest."""
    health = Indexer(project).health()
    if as_json:
        _report(health, True)
    else:
        _report({k: v for k, v in health.items() if k != "languages"}, False)
    # Exit non-zero only for states that actually block a query, so `doctor`
    # is usable as a CI gate; a merely dirty working tree is normal.
    blocking = not health.get("indexed") or not health.get("semantic_search_ready") or health.get("symbols_missing_vectors")
    sys.exit(1 if blocking else 0)


@index_group.command("search")
@_project_option
@click.argument("query")
@click.option("--limit", default=10, show_default=True, type=click.IntRange(1, 100))
@click.option("--lang", default="", help="Restrict to one language.")
@click.option("--kind", default="", help="Restrict to one symbol kind.")
@click.option("--path-glob", default="", help="Restrict to matching paths.")
@click.option("--exclude-tests", is_flag=True, help="Skip symbols defined in test files.")
@_json_option
def search(project: Path, query: str, limit: int, lang: str, kind: str, path_glob: str, exclude_tests: bool, as_json: bool) -> None:
    """Run the hybrid search from the shell (useful for sanity-checking an index)."""
    from codescope.index.search import SearchFilter

    hits = SearchEngine(project).hybrid_search(
        query,
        limit=limit,
        flt=SearchFilter(path_glob=path_glob or None, lang=lang or None, kind=kind or None, exclude_tests=exclude_tests),
    )
    if as_json:
        _report([asdict(h) for h in hits], True)
        return
    if not hits:
        click.echo("No matches. Is the index built? (codescope index build)")
        return
    for hit in hits:
        click.echo(f"{hit.score:.4f}  {hit.path}:{hit.start_line}  {hit.kind} {hit.name}  [{','.join(hit.sources)}]")
