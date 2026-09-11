"""Shared fixtures for the AIDEN suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="session")
def _git_identity() -> None:
    """Give git an identity for the whole suite.

    Several tests create a throwaway repository and commit into it. Without
    this they pass on a developer machine, which has a global user.name, and
    fail on a fresh CI runner, which does not -- so the suite was testing the
    machine as much as the code. Environment variables win over config, so
    this needs no global state and leaks nothing into the user's setup.
    """
    import os

    os.environ.setdefault("GIT_AUTHOR_NAME", "AIDEN Tests")
    os.environ.setdefault("GIT_AUTHOR_EMAIL", "tests@aiden.invalid")
    os.environ.setdefault("GIT_COMMITTER_NAME", "AIDEN Tests")
    os.environ.setdefault("GIT_COMMITTER_EMAIL", "tests@aiden.invalid")
