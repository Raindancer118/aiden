# Volantic Codescope

**An all-in-one codebase-intelligence MCP server.** Codescope gives a coding
agent everything it needs to work on a whole codebase — understand it, search
it, navigate it, edit it structurally, and run the development loop — through a
single Model Context Protocol server.

It is built as a private fork of [Serena](https://github.com/oraios/serena)
(MIT), inheriting its LSP-accurate symbol navigation and structural editing for
30+ languages, and adds the layers Serena deliberately lacks:

| Layer | What it adds | Tools |
|-------|--------------|-------|
| **Persistent index** | tree-sitter symbol extraction (33 languages) into a single SQLite database; incremental, gitignore-aware | `reindex`, `index_status`, `codescope index` (CLI) |
| **Hybrid search** | exact-name + BM25 (FTS5) + **local semantic vectors** (sqlite-vec) + trigram, fused with Reciprocal Rank Fusion; path/language/kind/test filters; every hit carries a folded code preview | `search_code`, `search_semantic`, `search_regex` |
| **One-call context** | code, callers, callees and the tests exercising it, in a single budgeted answer | `get_code_context` |
| **Code graphs** | dependencies, dependents, call chains, change-impact (blast radius), project map, file summaries | `get_dependencies`, `get_dependents`, `get_call_chain`, `get_change_impact`, `get_project_map`, `get_file_summary` |
| **IDE views** | type hierarchy and resolved call hierarchy via the language server; a Problems view across changed files or the whole project | `get_type_hierarchy`, `get_call_hierarchy`, `get_project_diagnostics` |
| **Reuse & clones** | "does this already exist?" before writing; semantic clone clusters; a pre-commit duplication gate | `find_similar_code`, `find_duplicate_code`, `detect_clones_in_diff` |
| **Live indexing** | git-scoped fast sync + a debounced background file watcher | `sync_index`, `watch_start`, `watch_stop`, `watch_status` |
| **Dev-ops** | auto-detecting test runner, project scaffolding, GitHub (via `gh`), git commit/diff/changelog | `run_tests`, `detect_test_framework`, `create_project`, `git_status`, `git_diff`, `git_commit`, `generate_changelog`, `github_*` |
| **Explorer (web UI)** | one shared local page for every project Codescope runs in: hybrid search, call graph, whole-project dependency graph, file tree, clone clusters, index health, and index actions | `codescope index explorer`, auto-started with the MCP server |
| **Durable memory** | mandatory Diary MCP backend with project auto-registration; no local Markdown fallback | `memory_context`, `memory_project_context`, `memory_get`, `memory_tree`, `memory_search*`, `memory_upsert`, `memory_delete` |

Plus the full Serena toolset (`find_symbol`, `find_referencing_symbols`,
`rename_symbol`, `replace_symbol_body`, shell, …). Serena's local
`read_memory`/`write_memory`/onboarding workflow is intentionally removed.

## Why it's strong

- **LSP accuracy *and* a persistent hybrid index in one server.** Serena gives
  precise, type-aware symbol resolution; Codescope adds fast lexical + semantic
  retrieval and structural graphs over a persistent index. Neither half exists
  in the other open-source code-index MCPs alone.
- **Local-first.** Semantic embeddings run locally (code-aware ONNX model via
  `fastembed`) — no API key, nothing leaves the machine. An optional Gemini
  backend is available for the highest retrieval quality.
- **One durable memory source.** Codescope maintains a persistent MCP session
  to Diary and uses only Diary's public tools. It never reads or writes the
  Diary database directly and never falls back to `.serena/memories`.
- **Mergeable fork.** Almost all Codescope value lives in `src/codescope/`.
  Two deliberately small Serena hooks support tool exclusion and the Diary
  activation hint; they keep upstream merges reviewable.

## Install

```bash
uv sync                       # installs Codescope + Serena + the index stack
uv run codescope start-mcp-server
```

Diary must be installed as the `diary-mcp-local` executable. Override its safe,
argument-aware launch command with `CODESCOPE_DIARY_COMMAND`; adjust the
per-call timeout with `CODESCOPE_DIARY_TIMEOUT_SECONDS`. If Diary is unavailable,
memory tools fail explicitly—there is no local fallback.

### Indexing

The first index downloads the local embedding model once. For an immediate
lexical index, call `reindex(embeddings=false)` first; a later normal reindex
backfills every missing vector without reparsing unchanged files.

For a large repository, run the first index from the shell instead of inside
the MCP server, where you can give it a budget:

```bash
codescope index build --project . -v
# under a hard memory cap, at low priority:
systemd-run --user --scope -p MemoryMax=2G -p CPUWeight=20 \
    nice -n 19 ionice -c3 codescope index build --project .
```

Vectors are written batch by batch, so an interrupted run keeps everything it
had already embedded and `build` resumes with what is still missing.
`codescope index sync|status|search` cover the rest.

Tuning (all optional):

| Variable | Meaning |
|----------|---------|
| `CODESCOPE_EMBED_MODEL` | fastembed model id. The default is code-aware and 768-dimensional; a smaller general model trades retrieval quality for a much faster, lighter index. |
| `CODESCOPE_EMBED_BATCH_COST` | memory budget per forward pass, as `item count x longest item^2` in char² (default 20000000). This is the dial for peak memory: ONNX pads every item to the longest one in the batch, and attention is quadratic in that length. Batches run longest-first, so this budget sets the high-water mark on the very first batch and the rest of the run reuses it. Raise it for throughput, lower it for a smaller footprint. |
| `CODESCOPE_EMBED_BATCH` | hard cap on documents per forward pass (default 128); the character budget above usually binds first. |
| `CODESCOPE_EMBED_THREADS` | ONNX thread count. Unset means "all cores". |

## The explorer

Activating a project starts a local web UI on <http://127.0.0.1:24256> and
opens it once. A second Codescope instance does not start a second server: it
registers its project with the one already running, and the open page picks it
up. `CODESCOPE_EXPLORER=0` turns the whole thing off; `codescope index explorer`
starts it by hand.

What it shows, per project:

- **Search** — the same hybrid search the agent uses, with the test filter.
- **Call graph** — callers and callees around a symbol, click any node to walk;
  or the **whole project** as a dependency ring aggregated by directory.
- **Files** — every indexed file, and the symbols inside one.
- **Clones** — near-duplicate clusters with their weakest pair.
- **Health** — what the index can and cannot answer, and what to do about it.

It can also run index maintenance (sync, reindex, watch start/stop). Those are
the only write operations exposed; editing, deletion and shell access stay out
of the browser on purpose. When you run one, the agent is told with its next
tool result — explicitly as a notification it does not have to act on.

The counter in the header opens the list of every project Codescope has run
in, running or not, and can attach one that is not. The power button stops
every attached instance and the server.

## Use with Claude Code

```bash
claude mcp add codescope -- uv run --directory "/path/to/codescope" codescope start-mcp-server
```

Or add to `.mcp.json`:

```json
{
  "mcpServers": {
    "codescope": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/codescope", "codescope", "start-mcp-server"]
    }
  }
}
```

Then, in a session: activate a project, run `reindex`, and start with
`get_code_context` for "how does X work / what breaks if I change it", falling
back to `search_code`, `get_call_hierarchy`, `get_change_impact`,
`get_project_diagnostics` and `run_tests` for the details.

## Architecture

```
codescope MCP (FastMCP, stdio)  —  Serena tool registry (extended at runtime)
  |- A  LSP / semantics        (serena + solidlsp; two small integration hooks)
  |- B  Hybrid index           (src/codescope/index: parser, store, embed, search, graph, incremental, watcher)
  |- C  Dev-ops                (src/codescope/devops: testrunner, scaffold, github, vcs)
  '- D  Durable memory         (persistent MCP client -> Diary; no local storage)
```

The per-project index lives at `.serena/codescope/index.db`.

## Development

```bash
uv run --with pytest pytest test/codescope -o addopts="" -p no:cacheprovider
uv run --with ruff ruff check src/codescope test/codescope
```

## Credits & license

Built on [Serena](https://github.com/oraios/serena) (MIT). Symbol-extraction
queries under `src/codescope/index/queries/` are vendored from
[Aider](https://github.com/Aider-AI/aider) (Apache-2.0); see the `NOTICE.md`
there. This project retains Serena's MIT license (see `LICENSE`).
