"""File-extension -> tree-sitter language resolution.

Each supported language maps to:
  - the ``tree-sitter-language-pack`` language name (passed to ``get_language``)
  - the vendored tags query file stem under ``queries/`` (``<stem>-tags.scm``)

Only languages for which we have a tags query are listed. Languages whose
grammar or query fails to load at runtime are skipped gracefully by the parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

QUERIES_DIR = Path(__file__).parent / "queries"


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    """tree-sitter-language-pack language name."""
    query_stem: str
    """tags query file stem: ``queries/<query_stem>-tags.scm``."""

    @property
    def query_path(self) -> Path:
        return QUERIES_DIR / f"{self.query_stem}-tags.scm"


# Language name -> query stem. Stems usually equal the language name.
_LANG_QUERY_STEM: dict[str, str] = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "tsx": "tsx",
    "go": "go",
    "rust": "rust",
    "java": "java",
    "ruby": "ruby",
    "c": "c",
    "cpp": "cpp",
    "csharp": "csharp",
    "lua": "lua",
    "elixir": "elixir",
    "elisp": "elisp",
    "clojure": "clojure",
    "commonlisp": "commonlisp",
    "dart": "dart",
    "swift": "swift",
    "bash": "bash",
    "r": "r",
    "ocaml": "ocaml",
    "ocaml_interface": "ocaml_interface",
    "elm": "elm",
    "solidity": "solidity",
    "d": "d",
    "gleam": "gleam",
    "racket": "racket",
    "arduino": "arduino",
    "matlab": "matlab",
    "pony": "pony",
}

# File extension (lowercase, with dot) -> language name.
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".rake": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".cs": "csharp",
    ".lua": "lua",
    ".ex": "elixir",
    ".exs": "elixir",
    ".el": "elisp",
    ".clj": "clojure",
    ".cljs": "clojure",
    ".cljc": "clojure",
    ".lisp": "commonlisp",
    ".cl": "commonlisp",
    ".dart": "dart",
    ".swift": "swift",
    ".sh": "bash",
    ".bash": "bash",
    ".r": "r",
    ".ml": "ocaml",
    ".mli": "ocaml_interface",
    ".elm": "elm",
    ".sol": "solidity",
    ".d": "d",
    ".gleam": "gleam",
    ".rkt": "racket",
    ".ino": "arduino",
    ".m": "matlab",
    ".pony": "pony",
}


def spec_for_path(path: str | Path) -> LanguageSpec | None:
    """Return the LanguageSpec for a file path, or None if unsupported."""
    ext = Path(path).suffix.lower()
    lang = _EXT_TO_LANG.get(ext)
    if lang is None:
        return None
    stem = _LANG_QUERY_STEM.get(lang)
    if stem is None:
        return None
    return LanguageSpec(name=lang, query_stem=stem)


def supported_extensions() -> set[str]:
    return set(_EXT_TO_LANG)
