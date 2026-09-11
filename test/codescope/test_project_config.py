"""Project configuration must survive real-world project.yml files.

Three failures seen in the wild, all of them fatal for `activate_project`:

* a project.yml written by a newer Serena, which calls the field
  ``language_servers`` instead of ``languages`` -> KeyError: 'languages'
* a project.yml with no language field at all -> KeyError: 'languages'
* a polyglot repo where only the single most common language was enabled, so
  every symbol tool on the other half of the repo answered
  "Cannot extract symbols ... Active languages: ['typescript']"
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from serena.config.serena_config import ProjectConfig, SerenaConfig
from solidlsp.ls_config import Language


@pytest.fixture(scope="module")
def serena_config() -> SerenaConfig:
    return SerenaConfig(gui_log_window=False, web_dashboard=False, log_level=logging.ERROR)


def _write(root: Path, relpath: str, text: str = "") -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _polyglot_project(root: Path) -> None:
    """A Java backend with a TypeScript frontend: both halves matter."""
    for i in range(12):
        _write(root, f"backend/src/main/java/de/x/C{i}.java", f"class C{i} {{}}\n")
    for i in range(9):
        _write(root, f"frontend/src/c{i}.ts", f"export const c{i} = {i};\n")
    for i in range(3):
        _write(root, f"scripts/s{i}.py", f"x = {i}\n")


# -- reading existing configuration files ---------------------------------


def test_language_servers_field_is_read_as_languages(tmp_path: Path) -> None:
    """Newer Serena writes `language_servers`; we must not choke on it."""
    yml = tmp_path / "project.yml"
    yml.write_text('project_name: "x"\nlanguage_servers:\n- java\n- typescript\n', encoding="utf-8")

    data, _ = ProjectConfig._load_yaml_dict(str(yml))

    assert data["languages"] == ["java", "typescript"]
    assert "language_servers" not in data


def test_singular_language_field_is_still_read(tmp_path: Path) -> None:
    yml = tmp_path / "project.yml"
    yml.write_text('project_name: "x"\nlanguage: python\n', encoding="utf-8")

    data, _ = ProjectConfig._load_yaml_dict(str(yml))

    assert data["languages"] == ["python"]


def test_missing_language_field_does_not_raise(tmp_path: Path) -> None:
    """A config without any language field must load, not KeyError."""
    yml = tmp_path / "project.yml"
    yml.write_text('project_name: "x"\n', encoding="utf-8")

    data, _ = ProjectConfig._load_yaml_dict(str(yml))
    config = ProjectConfig._from_dict(data, local_override_keys=[])

    assert config.languages == []


def test_load_backfills_languages_for_a_config_that_has_none(tmp_path: Path, serena_config: SerenaConfig) -> None:
    """A language-less config is repaired from what is actually in the repo."""
    _polyglot_project(tmp_path)
    yml = tmp_path / ".serena" / "project.yml"
    yml.parent.mkdir(parents=True, exist_ok=True)
    yml.write_text('project_name: "poly"\n', encoding="utf-8")

    config = ProjectConfig.load(tmp_path, serena_config=serena_config)

    assert Language.JAVA in config.languages
    assert Language.TYPESCRIPT in config.languages
    # and the repair is persisted, so the next activation is cheap
    assert "java" in yml.read_text(encoding="utf-8")


def test_load_keeps_an_explicit_single_language(tmp_path: Path, serena_config: SerenaConfig) -> None:
    """Detection must not second-guess a choice the user wrote down."""
    _polyglot_project(tmp_path)
    yml = tmp_path / ".serena" / "project.yml"
    yml.parent.mkdir(parents=True, exist_ok=True)
    yml.write_text('project_name: "poly"\nlanguages:\n- python\n', encoding="utf-8")

    config = ProjectConfig.load(tmp_path, serena_config=serena_config)

    assert config.languages == [Language.PYTHON]


# -- autogeneration --------------------------------------------------------


def test_autogenerate_enables_every_significant_language(tmp_path: Path, serena_config: SerenaConfig) -> None:
    """The bug: only the top language was enabled, so Java files were unreadable."""
    _polyglot_project(tmp_path)

    config = ProjectConfig.autogenerate(tmp_path, serena_config=serena_config, save_to_disk=False)

    assert Language.JAVA in config.languages
    assert Language.TYPESCRIPT in config.languages
    assert config.languages[0] == Language.JAVA, "the most common language stays the fallback"


def test_autogenerate_ignores_a_negligible_language(tmp_path: Path, serena_config: SerenaConfig) -> None:
    """One stray file must not start a whole language server."""
    for i in range(60):
        _write(tmp_path, f"src/m{i}.py", f"x = {i}\n")
    _write(tmp_path, "tools/one_off.rb", "puts 1\n")

    config = ProjectConfig.autogenerate(tmp_path, serena_config=serena_config, save_to_disk=False)

    assert config.languages == [Language.PYTHON]


def test_autogenerate_caps_the_number_of_language_servers(tmp_path: Path, serena_config: SerenaConfig) -> None:
    """Even a true polyglot repo must not spawn an unbounded server fleet."""
    for ext in ("py", "java", "ts", "go", "rs", "rb", "php", "cs"):
        for i in range(10):
            _write(tmp_path, f"src/{ext}/f{i}.{ext}", "\n")

    config = ProjectConfig.autogenerate(tmp_path, serena_config=serena_config, save_to_disk=False)

    assert 1 <= len(config.languages) <= ProjectConfig.MAX_AUTODETECTED_LANGUAGES


def test_autogenerate_skips_markup_and_config_languages(tmp_path: Path, serena_config: SerenaConfig) -> None:
    """JSON/YAML/Markdown outnumber code in many repos; they are not the point."""
    for i in range(30):
        _write(tmp_path, f"data/d{i}.json", "{}\n")
        _write(tmp_path, f"docs/d{i}.md", "# t\n")
    for i in range(10):
        _write(tmp_path, f"src/m{i}.py", f"x = {i}\n")

    config = ProjectConfig.autogenerate(tmp_path, serena_config=serena_config, save_to_disk=False)

    assert config.languages == [Language.PYTHON]


# -- error messages --------------------------------------------------------


def test_unanalysable_file_message_names_the_missing_language() -> None:
    """ "Active languages: ['typescript']" on a .java file must point at the fix."""
    from serena.tools.symbol_tools import _unanalysable_file_message

    message = _unanalysable_file_message("backend/src/main/java/de/x/AuthController.java", [Language.TYPESCRIPT])

    assert "java" in message
    assert "project.yml" in message


def test_unanalysable_file_message_stays_plain_for_an_unsupported_file() -> None:
    from serena.tools.symbol_tools import _unanalysable_file_message

    message = _unanalysable_file_message("notes/todo.rst", [Language.PYTHON])

    assert "project.yml" not in message
