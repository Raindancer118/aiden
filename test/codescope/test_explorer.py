"""Tests for the explorer server: attaching, payloads, actions and notices."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codescope.index.embed import HashingEmbedder
from codescope.index.indexer import Indexer
from codescope.web import explorer
from codescope.web.notices import format_events


@pytest.fixture
def project(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "orders.py").write_text(
        "def load_config(path):\n"
        "    return path\n"
        "\n"
        "\n"
        "def submit_order(order):\n"
        "    '''Send an order after validating it.'''\n"
        "    return load_config('cfg')\n"
    )
    (tmp_path / "util.py").write_text("def helper():\n    return submit_order({})\n")
    assert Indexer(tmp_path).reindex(embedder=HashingEmbedder()).errors == 0
    return tmp_path


@pytest.fixture
def client(project: Path):  # type: ignore[no-untyped-def]
    pytest.importorskip("flask")
    registry = explorer.ProjectRegistry()
    registry.history = explorer.ProjectHistory(project / ".history.json")
    registry.register(project, pid=4242)
    app = explorer.build_app(registry)
    app.config.update(TESTING=True)
    return app.test_client(), registry, explorer._project_id(project.resolve())


# -- registry and attaching --------------------------------------------------


def test_registering_the_same_root_twice_yields_one_project(tmp_path: Path) -> None:
    registry = explorer.ProjectRegistry()
    registry.history = explorer.ProjectHistory(tmp_path / "h.json")
    first = registry.register(tmp_path)
    second = registry.register(str(tmp_path) + "/")
    assert first.id == second.id
    assert len(registry.all()) == 1
    assert registry.version == 1  # the second registration is not a change


def test_history_survives_a_new_registry(tmp_path: Path) -> None:
    """The explorer must list a project even when nothing is attached to it."""
    store = tmp_path / "h.json"
    first = explorer.ProjectRegistry()
    first.history = explorer.ProjectHistory(store)
    first.register(tmp_path)

    later = explorer.ProjectHistory(store)
    assert [entry["root"] for entry in later.all()] == [str(tmp_path.resolve())]

    later.forget(str(tmp_path))
    assert later.all() == []


# -- payloads ---------------------------------------------------------------


def test_search_and_graph_endpoints(client) -> None:  # type: ignore[no-untyped-def]
    http, _registry, project_id = client

    hits = http.get(f"/api/projects/{project_id}/search?q=submit_order").get_json()["hits"]
    assert any(hit["name"] == "submit_order" for hit in hits)

    graph = http.get(f"/api/projects/{project_id}/graph?symbol=submit_order").get_json()
    assert [c["name"] for c in graph["callers"]] == ["helper"]
    assert [c["name"] for c in graph["callees"]] == ["load_config"]
    assert graph["ambiguous"] is False


def test_files_endpoint_lists_files_then_symbols(client) -> None:  # type: ignore[no-untyped-def]
    http, _registry, project_id = client

    files = http.get(f"/api/projects/{project_id}/files").get_json()["files"]
    assert {f["path"] for f in files} == {"src/orders.py", "util.py"}

    symbols = http.get(f"/api/projects/{project_id}/files?path=src/orders.py").get_json()["symbols"]
    assert [s["name"] for s in symbols] == ["load_config", "submit_order"]


def test_project_graph_aggregates_and_ignores_ambiguous_names(tmp_path: Path) -> None:
    (tmp_path / "core").mkdir()
    (tmp_path / "app").mkdir()
    (tmp_path / "core" / "db.py").write_text("def connect():\n    return 1\n")
    (tmp_path / "app" / "main.py").write_text("def run():\n    return connect()\n")
    Indexer(tmp_path).reindex(embeddings=False)

    payload = explorer._project_graph_payload(str(tmp_path), level="dir")
    assert {n["id"] for n in payload["nodes"]} == {"core", "app"}
    assert payload["edges"] == [{"source": "app", "target": "core", "weight": 1}]

    # A second definition of the same name makes the edge unresolvable, so it
    # must disappear rather than be drawn to both files.
    (tmp_path / "app" / "shadow.py").write_text("def connect():\n    return 2\n")
    Indexer(tmp_path).reindex(embeddings=False)
    assert explorer._project_graph_payload(str(tmp_path), level="dir")["edges"] == []


def test_unknown_project_is_a_404(client) -> None:  # type: ignore[no-untyped-def]
    http, _registry, _project_id = client
    assert http.get("/api/projects/deadbeef/overview").status_code == 404


def test_browsing_cannot_escape_the_home_directory() -> None:
    payload = explorer._browse_payload("/etc")
    assert payload["path"] == str(Path.home().resolve())


# -- actions and notices -----------------------------------------------------


def test_action_runs_and_is_recorded(client) -> None:  # type: ignore[no-untyped-def]
    http, registry, project_id = client
    response = http.post(f"/api/projects/{project_id}/actions/sync", json={})
    assert response.status_code == 200

    for _ in range(100):
        events = registry.actions.since(0)
        if events and events[0].status != "running":
            break
        import time as _time

        _time.sleep(0.05)
    events = registry.actions.since(0)
    assert events and events[0].action == "sync"
    assert events[0].status == "done", events[0].detail


def test_unknown_action_is_rejected(client) -> None:  # type: ignore[no-untyped-def]
    http, _registry, project_id = client
    response = http.post(f"/api/projects/{project_id}/actions/rm_rf", json={})
    assert response.status_code == 400
    assert "reindex" in response.get_json()["available"]


def test_events_are_handed_to_an_agent_once() -> None:
    log = explorer.ActionLog()
    event = log.add("proj", "reindex", "running", "started")
    assert log.take_unseen() == []  # a running action is not news yet
    log.update(event.id, "done", "12 indexed")

    notice = format_events(log.take_unseen())
    assert "reindex on proj" in notice
    assert "12 indexed" in notice
    assert "not a request" in notice
    assert format_events(log.take_unseen()) == ""


def test_notice_is_appended_to_a_tool_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from codescope.web import notices

    log = explorer.ActionLog()
    done = log.add("proj", "watch_start", "done", "watcher running")
    assert done.status == "done"

    class _Registry:
        actions = log

    monkeypatch.setattr(explorer, "hosted_registry", lambda: _Registry())
    combined = notices.append_notice('{"result": 1}')
    assert combined.startswith('{"result": 1}')
    assert "watch_start on proj" in combined


def test_a_failing_notice_never_breaks_a_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    from codescope.web import notices

    def boom() -> None:
        raise RuntimeError("registry exploded")

    monkeypatch.setattr(explorer, "hosted_registry", boom)
    assert notices.append_notice("payload") == "payload"


def test_shutdown_signals_instances_then_itself(client, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    http, _registry, _project_id = client
    signalled: list[int] = []
    stopped_self: list[float] = []
    monkeypatch.setattr("os.kill", lambda pid, sig: signalled.append(pid))
    monkeypatch.setattr(explorer, "stop_this_process", lambda delay_s=0.5: stopped_self.append(delay_s))

    payload = http.post("/api/shutdown").get_json()
    assert [entry["pid"] for entry in payload["stopped"]] == [4242]
    assert payload["server"] == "stopping"
    assert signalled == [4242]
    assert stopped_self, "the server must also stop itself"


def test_static_page_is_served(client) -> None:  # type: ignore[no-untyped-def]
    http, _registry, _project_id = client
    page = http.get("/")
    assert page.status_code == 200
    body = page.get_data(as_text=True)
    assert "Codescope Explorer" in body
    assert "app.js" in body
    assert json.loads(http.get("/api/projects").get_data(as_text=True))["projects"]
