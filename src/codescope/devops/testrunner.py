"""Test-framework auto-detection and execution.

Detects the project's test stack from marker files and runs it, returning a
structured result (framework, command, exit code, parsed pass/fail counts where
possible, and the tail of the output).
"""

from __future__ import annotations

import re
import shutil
import subprocess
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


def detect_framework(root: Path) -> Framework | None:
    """Detect the test framework for ``root`` from marker files."""
    # Python: pytest preferred when configured or tests present.
    if _has(root, "pyproject.toml", "setup.cfg", "setup.py", "tox.ini") or _has(root, "tests", "test"):
        pyproject = root / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8", errors="ignore") if pyproject.exists() else ""
        if "pytest" in text or _has(root, "conftest.py", "tests", "test") or "[tool.pytest" in text:
            runner = ["uv", "run", "pytest"] if (root / "uv.lock").exists() and shutil.which("uv") else ["pytest"]
            return Framework(name="pytest", command=runner, reason="found pyproject/tests indicating pytest")

    # JVM
    if _has(root, "gradlew"):
        return Framework(name="gradle", command=["./gradlew", "test"], reason="found gradlew")
    if _has(root, "build.gradle", "build.gradle.kts"):
        return Framework(name="gradle", command=["gradle", "test"], reason="found build.gradle")
    if _has(root, "pom.xml"):
        return Framework(name="maven", command=["mvn", "-q", "test"], reason="found pom.xml")

    # Node
    pkg = root / "package.json"
    if pkg.exists():
        text = pkg.read_text(encoding="utf-8", errors="ignore")
        if '"test"' in text:
            mgr = "pnpm" if _has(root, "pnpm-lock.yaml") else "yarn" if _has(root, "yarn.lock") else "npm"
            cmd = [mgr, "test"] if mgr != "npm" else ["npm", "test", "--silent"]
            return Framework(name=f"node:{mgr}", command=cmd, reason="package.json has a test script")

    # Rust / Go
    if _has(root, "Cargo.toml"):
        return Framework(name="cargo", command=["cargo", "test"], reason="found Cargo.toml")
    if _has(root, "go.mod"):
        return Framework(name="go", command=["go", "test", "./..."], reason="found go.mod")

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
    cmd = [*fw.command, *(extra_args or [])]
    if shutil.which(cmd[0]) is None and not cmd[0].startswith("./"):
        return TestResult(
            framework=fw.name,
            command=" ".join(cmd),
            exit_code=-1,
            ok=False,
            note=f"Test command '{cmd[0]}' is not installed.",
        )
    try:
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return TestResult(framework=fw.name, command=" ".join(cmd), exit_code=-1, ok=False, note=f"Timed out after {timeout}s.")
    output = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(output.splitlines()[-_OUTPUT_TAIL_LINES:])
    return TestResult(
        framework=fw.name,
        command=" ".join(cmd),
        exit_code=proc.returncode,
        ok=proc.returncode == 0,
        summary=_parse_summary(fw.name, output),
        output_tail=tail,
    )
