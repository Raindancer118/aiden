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
| **Persistent index** | tree-sitter symbol extraction (33 languages) into a single SQLite database; incremental, gitignore-aware | `reindex`, `index_status` |
| **Hybrid search** | BM25 (FTS5) + **local semantic vectors** (sqlite-vec) + trigram, fused with Reciprocal Rank Fusion; ripgrep regex | `search_code`, `search_semantic`, `search_regex` |
| **Code graphs** | dependencies, dependents, call chains, change-impact (blast radius), project map, file summaries | `get_dependencies`, `get_dependents`, `get_call_chain`, `get_change_impact`, `get_project_map`, `get_file_summary` |
| **Live indexing** | git-scoped fast sync + a debounced background file watcher | `sync_index`, `watch_start`, `watch_stop`, `watch_status` |
| **Dev-ops** | auto-detecting test runner, project scaffolding, GitHub (via `gh`), git commit/diff/changelog | `run_tests`, `detect_test_framework`, `create_project`, `git_status`, `git_diff`, `git_commit`, `generate_changelog`, `github_*` |

Plus the full Serena toolset (`find_symbol`, `find_referencing_symbols`,
`rename_symbol`, `replace_symbol_body`, memory, shell, …).

## Why it's strong

- **LSP accuracy *and* a persistent hybrid index in one server.** Serena gives
  precise, type-aware symbol resolution; Codescope adds fast lexical + semantic
  retrieval and structural graphs over a persistent index. Neither half exists
  in the other open-source code-index MCPs alone.
- **Local-first.** Semantic embeddings run locally (code-aware ONNX model via
  `fastembed`) — no API key, nothing leaves the machine. An optional Gemini
  backend is available for the highest retrieval quality.
- **Mergeable fork.** Serena's `serena/` and `solidlsp/` packages are kept
  unmodified; all Codescope value lives in `src/codescope/` and registers into
  Serena's tool registry at runtime, so upstream Serena updates stay mergeable.

## Install

```bash
uv sync                       # installs Codescope + Serena + the index stack
uv run codescope start-mcp-server
```

The first `reindex` downloads the local embedding model (~160 MB) once.

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

Then, in a session: activate a project, run `reindex`, and use `search_code`,
`get_call_chain`, `get_change_impact`, `run_tests`, etc.

## Architecture

```
codescope MCP (FastMCP, stdio)  —  Serena tool registry (extended at runtime)
  |- A  LSP / semantics        (serena + solidlsp, unmodified)
  |- B  Hybrid index           (src/codescope/index: parser, store, embed, search, graph, incremental, watcher)
  '- C  Dev-ops                (src/codescope/devops: testrunner, scaffold, github, vcs)
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
