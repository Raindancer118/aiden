"""Project scaffolding from templates.

Creates a new project skeleton for a given stack (Python/uv, Node/TS, Rust),
with a sensible source layout, a test scaffold, .gitignore, and README. Never
overwrites a non-empty target directory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_STACKS = ("python", "node", "rust")


@dataclass(slots=True)
class ScaffoldResult:
    stack: str
    path: str
    files: list[str]
    git_initialized: bool


def _module_name(name: str) -> str:
    mod = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return mod or "app"


def _python_template(name: str) -> dict[str, str]:
    mod = _module_name(name)
    return {
        "pyproject.toml": (
            "[project]\n"
            f'name = "{name}"\n'
            'version = "0.1.0"\n'
            'description = ""\n'
            'requires-python = ">=3.11"\n'
            "dependencies = []\n\n"
            "[project.scripts]\n"
            f'{name} = "{mod}.main:main"\n\n'
            "[build-system]\n"
            'requires = ["hatchling"]\n'
            'build-backend = "hatchling.build"\n\n'
            "[tool.hatch.build.targets.wheel]\n"
            f'packages = ["src/{mod}"]\n\n'
            "[dependency-groups]\n"
            'dev = ["pytest>=8.0"]\n\n'
            "[tool.pytest.ini_options]\n"
            'pythonpath = ["src"]\n'
            'testpaths = ["tests"]\n'
        ),
        f"src/{mod}/__init__.py": '__version__ = "0.1.0"\n',
        f"src/{mod}/main.py": (
            "def main() -> None:\n"
            f'    print("Hello from {name}")\n\n\n'
            'if __name__ == "__main__":\n'
            "    main()\n"
        ),
        "tests/__init__.py": "",
        f"tests/test_main.py": (
            f"from {mod}.main import main\n\n\n"
            "def test_main_runs(capsys):\n"
            "    main()\n"
            "    assert capsys.readouterr().out\n"
        ),
        ".gitignore": "__pycache__/\n*.py[cod]\n.venv/\ndist/\nbuild/\n*.egg-info/\n.pytest_cache/\n.mypy_cache/\n",
        "README.md": f"# {name}\n\n```bash\nuv sync\nuv run pytest\nuv run {name}\n```\n",
    }


def _node_template(name: str) -> dict[str, str]:
    return {
        "package.json": (
            "{\n"
            f'  "name": "{name}",\n'
            '  "version": "0.1.0",\n'
            '  "type": "module",\n'
            '  "scripts": {\n'
            '    "build": "tsc",\n'
            '    "test": "vitest run"\n'
            "  },\n"
            '  "devDependencies": {\n'
            '    "typescript": "^5.6.0",\n'
            '    "vitest": "^2.1.0"\n'
            "  }\n"
            "}\n"
        ),
        "tsconfig.json": (
            "{\n"
            '  "compilerOptions": {\n'
            '    "target": "ES2022",\n'
            '    "module": "ESNext",\n'
            '    "moduleResolution": "bundler",\n'
            '    "strict": true,\n'
            '    "outDir": "dist",\n'
            '    "rootDir": "src"\n'
            "  },\n"
            '  "include": ["src"]\n'
            "}\n"
        ),
        "src/index.ts": "export function greet(name: string): string {\n  return `Hello, ${name}`;\n}\n",
        "test/index.test.ts": (
            'import { describe, it, expect } from "vitest";\n'
            'import { greet } from "../src/index.js";\n\n'
            'describe("greet", () => {\n'
            '  it("greets", () => {\n'
            '    expect(greet("world")).toBe("Hello, world");\n'
            "  });\n"
            "});\n"
        ),
        ".gitignore": "node_modules/\ndist/\n*.log\n",
        "README.md": f"# {name}\n\n```bash\nnpm install\nnpm test\n```\n",
    }


def _rust_template(name: str) -> dict[str, str]:
    crate = _module_name(name)
    return {
        "Cargo.toml": (
            "[package]\n"
            f'name = "{crate}"\n'
            'version = "0.1.0"\n'
            'edition = "2021"\n\n'
            "[dependencies]\n"
        ),
        "src/lib.rs": (
            "pub fn add(a: i64, b: i64) -> i64 {\n"
            "    a + b\n"
            "}\n\n"
            "#[cfg(test)]\n"
            "mod tests {\n"
            "    use super::*;\n\n"
            "    #[test]\n"
            "    fn it_adds() {\n"
            "        assert_eq!(add(2, 2), 4);\n"
            "    }\n"
            "}\n"
        ),
        ".gitignore": "/target\n",
        "README.md": f"# {name}\n\n```bash\ncargo test\n```\n",
    }


_TEMPLATES = {"python": _python_template, "node": _node_template, "rust": _rust_template}


def create_project(stack: str, name: str, dest_dir: str | Path, *, git_init: bool = True) -> ScaffoldResult:
    stack = stack.lower()
    if stack not in _TEMPLATES:
        raise ValueError(f"Unsupported stack {stack!r}. Supported: {', '.join(SUPPORTED_STACKS)}")
    if not re.match(r"^[A-Za-z0-9._-]+$", name):
        raise ValueError("Project name may only contain letters, digits, '.', '_' and '-'.")

    target = Path(dest_dir).expanduser().resolve() / name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Target directory is not empty: {target}")

    files = _TEMPLATES[stack](name)
    written: list[str] = []
    for rel, content in files.items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(rel)

    git_done = False
    if git_init:
        try:
            import pygit2

            pygit2.init_repository(str(target))
            git_done = True
        except Exception:
            git_done = False

    return ScaffoldResult(stack=stack, path=str(target), files=sorted(written), git_initialized=git_done)
