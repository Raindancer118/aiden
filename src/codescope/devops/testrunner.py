"""Test-framework auto-detection and execution.

Detects the project's test stack from marker files and runs it, returning a
structured result (framework, command, exit code, parsed pass/fail counts where
possible, and the tail of the output).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

_OUTPUT_TAIL_LINES = 60


@dataclass(slots=True)
class Framework:
    name: str
    command: list[str]
    reason: str


@dataclass(slots=True)
class TestResult:
    framework: str
    command: str
    exit_code: int
    ok: bool
    summary: dict[str, int] = field(default_factory=dict)
    output_tail: str = ""
    note: str = ""


def _has(root: Path, *names: str) -> bool:
    return any((root / n).exists() for n in names)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.is_file() else ""


def _has_python_tests(root: Path) -> bool:
    for directory_name in ("tests", "test"):
        directory = root / directory_name
        has_prefixed_test = next(directory.rglob("test_*.py"), None) is not None if directory.is_dir() else False
        has_suffixed_test = next(directory.rglob("*_test.py"), None) is not None if directory.is_dir() else False
        if has_prefixed_test or has_suffixed_test:
            return True
    return False


def _wrapper(root: Path, name: str) -> str | None:
    windows_wrapper = root / f"{name}.bat"
    if sys.platform == "win32" and windows_wrapper.is_file():
        return f"./{name}.bat"
    return f"./{name}" if (root / name).is_file() else None


def detect_framework(root: Path) -> Framework | None:
    """Detect the test framework for ``root`` from marker files."""
    # prefer explicit pytest configuration over other ecosystem markers.
    python_config = "\n".join(_read_text(root / name) for name in ("pyproject.toml", "setup.cfg", "tox.ini"))
    if "pytest" in python_config or _has(root, "conftest.py"):
        runner = ["uv", "run", "pytest"] if (root / "uv.lock").exists() and shutil.which("uv") else ["pytest"]
        return Framework(name="pytest", command=runner, reason="found pytest configuration")

    # JVM
    gradle_wrapper = _wrapper(root, "gradlew")
    if gradle_wrapper:
        return Framework(name="gradle", command=[gradle_wrapper, "test"], reason="found Gradle wrapper")
    if _has(root, "build.gradle", "build.gradle.kts"):
        return Framework(name="gradle", command=["gradle", "test"], reason="found build.gradle")
    maven_wrapper = _wrapper(root, "mvnw")
    if maven_wrapper:
        return Framework(name="maven", command=[maven_wrapper, "-q", "test"], reason="found Maven wrapper")
    if _has(root, "pom.xml"):
        return Framework(name="maven", command=["mvn", "-q", "test"], reason="found pom.xml")

    # Node
    pkg = root / "package.json"
    if pkg.is_file():
        try:
            package_data = json.loads(_read_text(pkg))
        except json.JSONDecodeError:
            package_data = None
        scripts = package_data.get("scripts") if isinstance(package_data, dict) else None
        test_script = scripts.get("test") if isinstance(scripts, dict) else None
        if isinstance(test_script, str) and test_script.strip():
            mgr = "pnpm" if _has(root, "pnpm-lock.yaml") else "yarn" if _has(root, "yarn.lock") else "npm"
            cmd = [mgr, "test"] if mgr != "npm" else ["npm", "test", "--silent"]
            return Framework(name=f"node:{mgr}", command=cmd, reason="package.json has a test script")

    # Rust / Go
    if _has(root, "Cargo.toml"):
        return Framework(name="cargo", command=["cargo", "test"], reason="found Cargo.toml")
    if _has(root, "go.mod"):
        return Framework(name="go", command=["go", "test", "./..."], reason="found go.mod")

    # fall back to Python only for actual Python test modules. A generic
    # ``test`` directory is common in Node projects and is not a pytest marker.
    if _has_python_tests(root):
        runner = ["uv", "run", "pytest"] if (root / "uv.lock").exists() and shutil.which("uv") else ["pytest"]
        return Framework(name="pytest", command=runner, reason="found Python test modules")

    return None


def _parse_summary(framework: str, output: str) -> dict[str, int]:
    summary: dict[str, int] = {}
    if framework == "pytest":
        for key in ("passed", "failed", "error", "errors", "skipped", "xfailed", "xpassed"):
            m = re.search(rf"(\d+)\s+{key}\b", output)
            if m:
                summary[key.rstrip("s") if key in ("errors",) else key] = int(m.group(1))
    elif framework == "cargo":
        m = re.search(r"test result:.*?(\d+)\s+passed;\s+(\d+)\s+failed", output)
        if m:
            summary = {"passed": int(m.group(1)), "failed": int(m.group(2))}
    elif framework == "go":
        summary = {"passed": len(re.findall(r"^--- PASS", output, re.M)), "failed": len(re.findall(r"^--- FAIL", output, re.M))}
    return summary


def run_tests(root: Path, extra_args: list[str] | None = None, timeout: int = 1800) -> TestResult:
    fw = detect_framework(root)
    if fw is None:
        return TestResult(
            framework="unknown",
            command="",
            exit_code=-1,
            ok=False,
            note="Could not detect a test framework (no pytest/gradle/maven/npm/cargo/go markers found).",
        )
    args = extra_args or []
    separator = ["--"] if args and fw.name == "node:npm" else []
    cmd = [*fw.command, *separator, *args]
    command = " ".join(cmd)
    if timeout <= 0:
        return TestResult(framework=fw.name, command=command, exit_code=-1, ok=False, note="Timeout must be greater than zero.")
    if shutil.which(cmd[0]) is None and not cmd[0].startswith("./"):
        return TestResult(
            framework=fw.name,
            command=command,
            exit_code=-1,
            ok=False,
            note=f"Test command '{cmd[0]}' is not installed.",
        )
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return TestResult(framework=fw.name, command=command, exit_code=-1, ok=False, note=f"Timed out after {timeout}s.")
    except OSError as error:
        return TestResult(framework=fw.name, command=command, exit_code=-1, ok=False, note=f"Could not start test command: {error}")
    output = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    tail = "\n".join(output.splitlines()[-_OUTPUT_TAIL_LINES:])
    return TestResult(
        framework=fw.name,
        command=command,
        exit_code=proc.returncode,
        ok=proc.returncode == 0,
        summary=_parse_summary(fw.name, output),
        output_tail=tail,
    )
