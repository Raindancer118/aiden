"""A single shared explorer server that every Codescope instance attaches to.

Serena starts one dashboard per agent, each on its own port. That is the wrong
shape here: a developer runs Codescope in several repositories at once and
wants *one* page listing all of them, not five tabs. So the explorer binds a
fixed port, and an instance that finds it taken registers its project with the
server already running instead of starting a second one. The open page picks
the new project up on its next poll.

Everything is served on the loopback interface only. The API reads the index
and the working tree; it never writes to either.
"""

from __future__ import annotations

import logging
import os
import secrets
import socket
import threading
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger(__name__)

#: Fixed so a second instance can find the first. Serena's dashboard sits at
#: 0x5EDA; this is deliberately adjacent and out of its scan range.
EXPLORER_PORT = 0x5EC0  # 24256
EXPLORER_HOST = "127.0.0.1"

_ATTACH_TIMEOUT_S = 2.0
#: A port can be held by an explorer that has not finished starting, or by a
#: socket the kernel has not released yet. Both clear up in well under a
#: second, so a couple of retries beat refusing to start.
_ATTACH_RETRIES = 4
_ATTACH_RETRY_DELAY_S = 0.4
_STATIC_DIR = Path(__file__).parent / "static"

#: Every project Codescope has ever been used in, so the explorer can list
#: them when no instance is currently attached to them.
_HISTORY_FILE = Path.home() / ".codescope" / "projects.json"

#: Shared secret for state-changing requests, readable only by this user.
#: Loopback is not a boundary on its own: any local process can connect, and
#: a page the user visits can point a hostname at 127.0.0.1 and POST to it.
_TOKEN_FILE = Path.home() / ".codescope" / "token"
_TOKEN_HEADER = "X-Codescope-Token"

#: Host names that may address the explorer. A rebound DNS name resolves to
#: loopback but still arrives with the attacker's Host header.
_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})

_server_lock = threading.Lock()
#: The server hosted by *this* process, if any. Held in a container so the
#: lifecycle helpers can rebind it without a ``global`` statement.
_hosted: dict[str, "ExplorerServer"] = {}


@dataclass
class ActionEvent:
    """Something the user did in the explorer that an agent may want to know."""

    id: int
    at: float
    project: str
    action: str
    status: str  # running | done | failed
    detail: str = ""


class ActionLog:
    """A short, read-once record of user-triggered actions.

    The point is to let an agent working in the same repository notice that
    the index was rebuilt or a watcher started underneath it. Entries are
    handed out once and then marked seen, so a long session does not keep
    re-reading the same notice.
    """

    LIMIT = 200

    def __init__(self) -> None:
        self._events: list[ActionEvent] = []
        self._next_id = 1
        self._seen_through = 0
        self._lock = threading.Lock()

    def add(self, project: str, action: str, status: str, detail: str = "") -> ActionEvent:
        with self._lock:
            event = ActionEvent(id=self._next_id, at=time.time(), project=project, action=action, status=status, detail=detail)
            self._next_id += 1
            self._events.append(event)
            del self._events[: -self.LIMIT]
            return event

    def update(self, event_id: int, status: str, detail: str) -> None:
        with self._lock:
            for event in self._events:
                if event.id == event_id:
                    event.status = status
                    event.detail = detail
                    return

    def since(self, after_id: int) -> list[ActionEvent]:
        with self._lock:
            return [e for e in self._events if e.id > after_id]

    def take_unseen(self) -> list[ActionEvent]:
        """Return finished events not handed out yet, and mark them seen."""
        with self._lock:
            fresh = [e for e in self._events if e.id > self._seen_through and e.status != "running"]
            if fresh:
                self._seen_through = max(e.id for e in fresh)
            return fresh


@dataclass
class RegisteredProject:
    """One project visible in the explorer."""

    id: str
    name: str
    root: str
    registered_at: float = field(default_factory=time.time)
    #: The process that attached this project, so the explorer can shut the
    #: whole set down. ``None`` for a project added from the browser, which
    #: belongs to no process but the explorer itself.
    pid: int | None = None

    @classmethod
    def for_root(cls, root: str | Path, pid: int | None = None) -> "RegisteredProject":
        resolved = Path(root).resolve()
        return cls(id=_project_id(resolved), name=resolved.name, root=str(resolved), pid=pid)


@lru_cache(maxsize=1)
def explorer_token() -> str:
    """The per-install secret, created on first use with mode 0600."""
    try:
        existing = _TOKEN_FILE.read_text(encoding="utf-8").strip()
        if len(existing) >= 32:
            return existing
    except OSError:
        pass

    token = secrets.token_urlsafe(32)
    try:
        _TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Create with 0600 from the start rather than chmod-ing afterwards,
        # which would leave a window where the secret is world-readable.
        fd = os.open(_TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token)
    except OSError as e:  # pragma: no cover - read-only home
        log.warning("Could not persist the explorer token: %s", e)
    return token


def is_own_codescope_process(pid: int) -> bool:
    """Whether ``pid`` is a live Codescope process belonging to this user.

    A registered pid is a claim made over the network, so it is never
    signalled on trust: an attacker who can reach the API would otherwise
    have an arbitrary-process-kill primitive for everything this user owns.
    """
    if pid <= 1:
        return False
    try:
        import psutil

        process = psutil.Process(pid)
        if process.uids().real != os.getuid():
            return False
        haystack = " ".join(process.cmdline() or []).lower()
    except Exception:
        return False
    return "codescope" in haystack or pid == os.getpid()


def _under_home(root: str | Path) -> Path | None:
    """Resolve ``root`` if it lies inside the user's home, else ``None``.

    The same confinement the directory browser applies. Registering a path
    makes the API read an index under it and report what it finds, so the
    reachable set has to be bounded somewhere.
    """
    home = Path.home().resolve()
    try:
        resolved = Path(root).resolve()
        resolved.relative_to(home)
    except (OSError, ValueError):
        return None
    return resolved


def _project_id(root: Path) -> str:
    """Stable, URL-safe id for a project root."""
    import hashlib

    return hashlib.blake2b(str(root).encode(), digest_size=6).hexdigest()


class ProjectHistory:
    """Projects Codescope has run in before, remembered across restarts.

    The explorer is most useful as the place you come back to, which means
    it has to know about a project even when nothing is attached to it right
    now. A small JSON file is enough; the index itself stays where it is.
    """

    def __init__(self, path: Path = _HISTORY_FILE) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict[str, dict]:
        try:
            import json

            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def remember(self, root: str | Path) -> None:
        import json

        resolved = str(Path(root).resolve())
        with self._lock:
            entries = self._read()
            entries[resolved] = {"root": resolved, "name": Path(resolved).name, "last_seen": time.time()}
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            except OSError as e:  # pragma: no cover - a read-only home
                log.warning("Could not remember project %s: %s", resolved, e)

    def forget(self, root: str) -> None:
        import json

        with self._lock:
            entries = self._read()
            if entries.pop(str(Path(root).resolve()), None) is None:
                return
            try:
                self.path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            except OSError:  # pragma: no cover
                pass

    def all(self) -> list[dict]:
        entries = list(self._read().values())
        entries.sort(key=lambda e: -e.get("last_seen", 0))
        return entries


class ProjectRegistry:
    """The set of projects the running explorer knows about."""

    def __init__(self) -> None:
        self._projects: dict[str, RegisteredProject] = {}
        self._lock = threading.Lock()
        #: Bumped on every change so the page can poll cheaply.
        self.version = 0
        self.actions = ActionLog()
        self.history = ProjectHistory()

    def register(self, root: str | Path, pid: int | None = None) -> RegisteredProject:
        project = RegisteredProject.for_root(root, pid)
        with self._lock:
            existing = self._projects.get(project.id)
            if existing is not None:
                return existing
            self._projects[project.id] = project
            self.version += 1
        self.history.remember(project.root)
        log.info("Explorer: registered project %s (%s)", project.name, project.root)
        return project

    def get(self, project_id: str) -> RegisteredProject | None:
        with self._lock:
            return self._projects.get(project_id)

    def all(self) -> list[RegisteredProject]:
        with self._lock:
            return sorted(self._projects.values(), key=lambda p: p.registered_at)


def build_app(registry: ProjectRegistry):  # type: ignore[no-untyped-def]
    """Build the Flask app. Imported lazily so the CLI stays fast."""
    from flask import Flask, Response, jsonify, request, send_from_directory

    app = Flask(__name__, static_folder=None)

    #: Reads are open to anything that already reached loopback with the
    #: right Host; writes additionally need the token, which only a
    #: same-origin page (or a local process reading ~/.codescope/token) has.
    safe_methods = {"GET", "HEAD", "OPTIONS"}

    @app.before_request
    def guard():  # type: ignore[no-untyped-def]
        host = request.host.rsplit(":", 1)[0] if request.host else ""
        if host not in _ALLOWED_HOSTS:
            # DNS rebinding: the request reaches loopback, but under a name
            # we never serve.
            return jsonify({"error": "This explorer only answers to localhost."}), 403

        origin = request.headers.get("Origin")
        if origin and urlparse(origin).hostname not in _ALLOWED_HOSTS:
            return jsonify({"error": "Cross-origin requests are not accepted."}), 403

        if request.method in safe_methods:
            return None
        if not secrets.compare_digest(request.headers.get(_TOKEN_HEADER, ""), explorer_token()):
            return jsonify({"error": "Missing or invalid explorer token."}), 403
        return None

    def _project_or_404(project_id: str):  # type: ignore[no-untyped-def]
        project = registry.get(project_id)
        if project is None:
            return None, (jsonify({"error": f"Unknown project {project_id!r}"}), 404)
        return project, None

    # -- static ---------------------------------------------------------

    @app.get("/")
    def index():  # type: ignore[no-untyped-def]
        # The token is handed to the page, not to the network: a
        # cross-origin request cannot read this response, so it cannot
        # learn the value it would need to POST with.
        page = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
        return Response(page.replace("__CODESCOPE_TOKEN__", explorer_token()), mimetype="text/html")

    @app.get("/<path:filename>")
    def static_file(filename: str):  # type: ignore[no-untyped-def]
        return send_from_directory(_STATIC_DIR, filename)

    # -- projects -------------------------------------------------------

    @app.get("/api/projects")
    def list_projects():  # type: ignore[no-untyped-def]
        return jsonify({"version": registry.version, "projects": [asdict(p) for p in registry.all()]})

    @app.post("/api/projects")
    def register_project():  # type: ignore[no-untyped-def]
        payload = request.get_json(silent=True) or {}
        root = payload.get("root")
        if not root:
            return jsonify({"error": "root is required"}), 400
        resolved = _under_home(root)
        if resolved is None:
            return jsonify({"error": f"Only paths inside your home directory can be registered: {root}"}), 400
        if not resolved.is_dir():
            return jsonify({"error": f"Not a directory: {root}"}), 400
        return jsonify(asdict(registry.register(resolved, payload.get("pid"))))

    @app.get("/api/known")
    def known_projects():  # type: ignore[no-untyped-def]
        """Projects Codescope has run in, whether attached right now or not."""
        attached = {p.root: p for p in registry.all()}
        rows = []
        for entry in registry.history.all():
            root = entry["root"]
            index_db = Path(root) / ".serena" / "codescope" / "index.db"
            live = attached.get(root)
            rows.append(
                {
                    "name": entry.get("name") or Path(root).name,
                    "root": root,
                    "id": live.id if live else _project_id(Path(root)),
                    "attached": live is not None,
                    "pid": live.pid if live else None,
                    "exists": Path(root).is_dir(),
                    "indexed": index_db.exists(),
                    "index_size": index_db.stat().st_size if index_db.exists() else 0,
                    "last_seen": entry.get("last_seen", 0),
                }
            )
        return jsonify({"projects": rows})

    @app.post("/api/known/forget")
    def forget_project():  # type: ignore[no-untyped-def]
        payload = request.get_json(silent=True) or {}
        root = payload.get("root")
        if not root:
            return jsonify({"error": "root is required"}), 400
        registry.history.forget(root)
        return jsonify({"forgotten": root})

    @app.get("/api/browse")
    def browse():  # type: ignore[no-untyped-def]
        return jsonify(_browse_payload(request.args.get("path") or None))

    @app.post("/api/shutdown")
    def shutdown():  # type: ignore[no-untyped-def]
        """Stop every attached instance and then this server.

        Explicitly confirmed in the UI: it terminates other processes, which
        is the one thing here that reaches outside the explorer.
        """
        import signal

        stopped, failed, refused = [], [], []
        for project in registry.all():
            pid = project.pid
            if not pid or pid == os.getpid():
                continue
            if not is_own_codescope_process(pid):
                # The pid arrived over the API. Signalling it unchecked
                # would let anything that can reach this endpoint kill any
                # process the user owns.
                refused.append({"project": project.name, "pid": pid, "reason": "not a verifiable Codescope process"})
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                stopped.append({"project": project.name, "pid": pid})
            except OSError as e:
                failed.append({"project": project.name, "pid": pid, "error": str(e)})
        registry.actions.add("explorer", "shutdown", "done", f"{len(stopped)} instance(s) signalled")
        stop_this_process(delay_s=0.5)
        return jsonify({"stopped": stopped, "failed": failed, "refused": refused, "server": "stopping"})

    @app.get("/api/projects/<project_id>/status")
    def project_status(project_id: str):  # type: ignore[no-untyped-def]
        from codescope.index.indexer import Indexer

        project, error = _project_or_404(project_id)
        if error:
            return error
        return jsonify(Indexer(project.root).health())

    # -- search ---------------------------------------------------------

    @app.get("/api/projects/<project_id>/search")
    def search(project_id: str):  # type: ignore[no-untyped-def]
        from codescope.index.search import SearchEngine, SearchFilter

        project, error = _project_or_404(project_id)
        if error:
            return error
        query = request.args.get("q", "").strip()
        if not query:
            return jsonify({"hits": []})
        flt = SearchFilter(
            path_glob=request.args.get("path_glob") or None,
            lang=request.args.get("lang") or None,
            kind=request.args.get("kind") or None,
            exclude_tests=request.args.get("exclude_tests") == "1",
        )
        hits = SearchEngine(project.root).hybrid_search(
            query,
            limit=min(int(request.args.get("limit", 20)), 100),
            flt=flt,
            preview_lines=int(request.args.get("preview_lines", 8)),
        )
        return jsonify({"hits": [asdict(h) for h in hits]})

    # -- graph ----------------------------------------------------------

    @app.get("/api/projects/<project_id>/graph")
    def graph(project_id: str):  # type: ignore[no-untyped-def]
        project, error = _project_or_404(project_id)
        if error:
            return error
        name = request.args.get("symbol", "").strip()
        if not name:
            return jsonify({"error": "symbol is required"}), 400
        return jsonify(_graph_payload(project.root, name, depth=min(int(request.args.get("depth", 1)), 3)))

    @app.get("/api/projects/<project_id>/symbol")
    def symbol(project_id: str):  # type: ignore[no-untyped-def]
        project, error = _project_or_404(project_id)
        if error:
            return error
        return jsonify(_symbol_payload(project.root, request.args.get("name", ""), request.args.get("path") or None))

    # -- actions --------------------------------------------------------

    @app.get("/api/events")
    def events():  # type: ignore[no-untyped-def]
        after = int(request.args.get("since", 0))
        return jsonify({"events": [asdict(e) for e in registry.actions.since(after)]})

    @app.post("/api/projects/<project_id>/actions/<action>")
    def run_action(project_id: str, action: str):  # type: ignore[no-untyped-def]
        project, error = _project_or_404(project_id)
        if error:
            return error
        if action not in ACTIONS:
            return jsonify({"error": f"Unknown action {action!r}", "available": sorted(ACTIONS)}), 400
        payload = request.get_json(silent=True) or {}
        event = registry.actions.add(project.name, action, "running", "started")

        def worker() -> None:
            try:
                detail = ACTIONS[action](project.root, payload)
                registry.actions.update(event.id, "done", detail)
            except Exception as e:
                log.warning("Explorer action %s failed for %s: %s", action, project.name, e)
                registry.actions.update(event.id, "failed", str(e)[:300])

        threading.Thread(target=worker, name=f"codescope-action:{action}", daemon=True).start()
        return jsonify(asdict(event))

    @app.get("/api/projects/<project_id>/project-graph")
    def project_graph(project_id: str):  # type: ignore[no-untyped-def]
        project, error = _project_or_404(project_id)
        if error:
            return error
        return jsonify(
            _project_graph_payload(
                project.root,
                level=request.args.get("level", "dir"),
                limit=min(int(request.args.get("limit", 28)), 60),
            )
        )

    @app.get("/api/projects/<project_id>/files")
    def files(project_id: str):  # type: ignore[no-untyped-def]
        project, error = _project_or_404(project_id)
        if error:
            return error
        return jsonify(_files_payload(project.root, request.args.get("path") or None))

    @app.get("/api/projects/<project_id>/clones")
    def clones(project_id: str):  # type: ignore[no-untyped-def]
        project, error = _project_or_404(project_id)
        if error:
            return error
        return jsonify(
            _clones_payload(
                project.root,
                min_lines=max(int(request.args.get("min_lines", 8)), 2),
                similarity=float(request.args.get("similarity", 0.92)),
            )
        )

    @app.get("/api/projects/<project_id>/type-hierarchy")
    def type_hierarchy(project_id: str):  # type: ignore[no-untyped-def]
        from codescope.index.graph import GraphEngine

        project, error = _project_or_404(project_id)
        if error:
            return error
        name = request.args.get("name", "").strip()
        if not name:
            return jsonify({"error": "name is required"}), 400
        return jsonify(asdict(GraphEngine(project.root).type_hierarchy(name)))

    @app.get("/api/projects/<project_id>/impact")
    def impact(project_id: str):  # type: ignore[no-untyped-def]
        from codescope.index.graph import GraphEngine

        project, error = _project_or_404(project_id)
        if error:
            return error
        name = request.args.get("name", "").strip()
        if not name:
            return jsonify({"error": "name is required"}), 400
        depth = min(int(request.args.get("depth", 3)), 5)
        return jsonify(asdict(GraphEngine(project.root).change_impact(name, depth=depth)))

    @app.get("/api/projects/<project_id>/overview")
    def overview(project_id: str):  # type: ignore[no-untyped-def]
        project, error = _project_or_404(project_id)
        if error:
            return error
        return jsonify(_overview_payload(project.root))

    return app


# -- actions ---------------------------------------------------------------
#
# Only index maintenance is exposed. These are operations the user could run
# in a terminal anyway, they touch nothing but the index, and none of them
# can lose work. Editing, deleting and shell execution stay out of the web UI
# on purpose.


def _action_reindex(root: str, payload: dict) -> str:
    from codescope.index.indexer import Indexer

    report = Indexer(root).reindex(force=bool(payload.get("force")), embeddings=payload.get("embeddings", True))
    return f"{report.indexed} indexed, {report.skipped_unchanged} unchanged, {report.removed} pruned in {report.duration_s}s"


def _action_sync(root: str, _payload: dict) -> str:
    from codescope.index.incremental import sync_incremental
    from codescope.index.indexer import Indexer

    report = sync_incremental(Indexer(root))
    return f"{report.indexed} indexed, {report.removed} pruned in {report.duration_s}s"


def _action_watch_start(root: str, _payload: dict) -> str:
    from codescope.index.watcher import start_watcher

    status = start_watcher(root)
    return "watcher running" if status.get("running") else f"watcher did not start: {status.get('error')}"


def _action_watch_stop(root: str, _payload: dict) -> str:
    from codescope.index.watcher import stop_watcher

    stop_watcher(root)
    return "watcher stopped"


#: Action name -> handler. Handlers run off the request thread and return the
#: one-line summary shown in the UI and handed to the agent.
ACTIONS = {
    "reindex": _action_reindex,
    "sync": _action_sync,
    "watch_start": _action_watch_start,
    "watch_stop": _action_watch_stop,
}


# -- payload builders (pure, so they can be tested without a server) --------


def _graph_payload(root: str, name: str, depth: int = 1) -> dict:
    """Callers and callees around ``name``, plus where it is defined."""
    from codescope.index.graph import GraphEngine
    from codescope.index.indexer import default_db_path
    from codescope.index.store import IndexStore

    engine = GraphEngine(root)
    with IndexStore(default_db_path(root)) as store:
        definitions = [
            {"name": r[0], "kind": r[1], "path": r[2], "start_line": r[3], "end_line": r[4]}
            for r in store.conn.execute(
                "SELECT name, kind, path, start_line, end_line FROM symbols WHERE name=? ORDER BY path, start_line",
                (name,),
            )
        ]
        callers = [asdict(r) for r in engine.dependents_in(store, name)]
        callees = [asdict(r) for r in engine.dependencies_in(store, name)]
        ambiguous = len(definitions) > 1

        expanded: dict[str, dict] = {}
        if depth > 1:
            for related in (*callers, *callees):
                child = related["name"]
                if child in expanded:
                    continue
                expanded[child] = {
                    "callers": [asdict(r) for r in engine.dependents_in(store, child)],
                    "callees": [asdict(r) for r in engine.dependencies_in(store, child)],
                }

    return {
        "symbol": name,
        "definitions": definitions,
        "ambiguous": ambiguous,
        "callers": callers,
        "callees": callees,
        "expanded": expanded,
    }


def _group_key(path: str, level: str) -> str:
    """Collapse a file path to the level the project graph is drawn at."""
    if level == "file":
        return path
    parts = path.split("/")
    if len(parts) == 1:
        return "."
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0]


def _project_graph_payload(root: str, level: str = "dir", limit: int = 28) -> dict:
    """Whole-project dependency graph, aggregated so it can be read.

    A symbol-level graph of a real codebase is thousands of nodes and tells
    you nothing, so edges are rolled up to files or directories and weighted
    by how many references cross the boundary.

    Only names with a single definition in the project contribute. An
    ambiguous name would otherwise draw an edge to every file that happens
    to define something with that name, which is noise rather than structure.
    """
    from codescope.index.indexer import default_db_path
    from codescope.index.store import IndexStore

    level = "file" if level == "file" else "dir"
    db = default_db_path(root)
    if not db.exists():
        return {"level": level, "nodes": [], "edges": [], "truncated": False}

    with IndexStore(db) as store:
        rows = store.conn.execute(
            "WITH unique_def AS ("
            "  SELECT name, MIN(path) AS path FROM symbols GROUP BY name HAVING COUNT(DISTINCT path) = 1"
            ") "
            "SELECT owner.path, ud.path, COUNT(*) "
            "FROM refs AS r "
            "JOIN symbols AS owner ON owner.id = r.owner_id "
            "JOIN unique_def AS ud ON ud.name = r.name "
            "WHERE owner.path <> ud.path "
            "GROUP BY 1, 2"
        ).fetchall()
        sizes = dict(store.conn.execute("SELECT path, COUNT(*) FROM symbols GROUP BY path"))

    weights: dict[tuple[str, str], int] = {}
    for src, dst, count in rows:
        key = (_group_key(src, level), _group_key(dst, level))
        if key[0] == key[1]:
            continue
        weights[key] = weights.get(key, 0) + count

    degree: dict[str, int] = {}
    for (src, dst), weight in weights.items():
        degree[src] = degree.get(src, 0) + weight
        degree[dst] = degree.get(dst, 0) + weight

    kept = sorted(degree, key=lambda n: -degree[n])[:limit]
    keep = set(kept)
    edges = [
        {"source": src, "target": dst, "weight": weight}
        for (src, dst), weight in sorted(weights.items(), key=lambda kv: -kv[1])
        if src in keep and dst in keep
    ]

    symbol_counts: dict[str, int] = {}
    for path, count in sizes.items():
        group = _group_key(path, level)
        if group in keep:
            symbol_counts[group] = symbol_counts.get(group, 0) + count

    nodes = [{"id": name, "symbols": symbol_counts.get(name, 0), "degree": degree[name]} for name in sorted(kept, key=lambda n: -degree[n])]
    return {"level": level, "nodes": nodes, "edges": edges, "truncated": len(degree) > len(kept)}


#: Directory names never worth showing in the project browser.
_BROWSE_SKIP = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


def _browse_payload(path: str | None) -> dict:
    """List subdirectories so a project can be picked without typing a path.

    Confined to the user's home directory: the explorer listens on loopback,
    but there is no reason for a browser page to enumerate the whole disk.
    """
    home = Path.home().resolve()
    current = Path(path).resolve() if path else home
    try:
        current.relative_to(home)
    except ValueError:
        current = home
    if not current.is_dir():
        current = home

    entries = []
    try:
        for entry in sorted(current.iterdir(), key=lambda e: e.name.lower()):
            if not entry.is_dir() or entry.is_symlink() or entry.name.startswith(".") or entry.name in _BROWSE_SKIP:
                continue
            entries.append(
                {
                    "name": entry.name,
                    "path": str(entry),
                    "is_project": (entry / ".git").exists() or (entry / ".serena").exists(),
                    "indexed": (entry / ".serena" / "codescope" / "index.db").exists(),
                }
            )
    except PermissionError:
        pass

    parent = str(current.parent) if current != home else None
    return {"path": str(current), "parent": parent, "home": str(home), "entries": entries}


def _files_payload(root: str, path: str | None = None) -> dict:
    """Every indexed file, or the symbols inside one of them."""
    from codescope.index.indexer import default_db_path
    from codescope.index.store import IndexStore

    db = default_db_path(root)
    if not db.exists():
        return {"files": [], "symbols": [], "path": path}

    with IndexStore(db) as store:
        if path:
            symbols = [
                {"name": r[0], "kind": r[1], "path": path, "start_line": r[2], "end_line": r[3], "signature": r[4] or ""}
                for r in store.conn.execute(
                    "SELECT name, kind, start_line, end_line, signature FROM symbols WHERE path=? ORDER BY start_line",
                    (path,),
                )
            ]
            return {"files": [], "symbols": symbols, "path": path}
        files = [
            {"path": r[0], "lang": r[1], "size": r[2], "symbols": r[3], "refs": r[4]}
            for r in store.conn.execute(
                "SELECT fi.path, fi.lang, fi.size, "
                "  (SELECT COUNT(*) FROM symbols s WHERE s.path = fi.path), "
                "  (SELECT COUNT(*) FROM refs r WHERE r.path = fi.path) "
                "FROM files AS fi ORDER BY fi.path"
            )
        ]
    return {"files": files, "symbols": [], "path": None}


def _clones_payload(root: str, min_lines: int = 8, similarity: float = 0.92) -> dict:
    """Near-duplicate symbol clusters, or why they cannot be computed."""
    from codescope.index.search import SearchEngine

    try:
        groups = SearchEngine(root).find_duplicate_code(min_lines=min_lines, similarity=similarity, limit=40)
    except RuntimeError as e:
        return {"error": str(e), "groups": []}
    return {
        "min_lines": min_lines,
        "similarity": similarity,
        "groups": [
            {
                "similarity": group.similarity,
                "min_similarity": group.min_similarity,
                "members": [asdict(member) for member in group.members],
            }
            for group in groups
        ],
    }


def _symbol_payload(root: str, name: str, path: str | None) -> dict:
    """The code of one symbol, for the detail pane."""
    from codescope.index.indexer import default_db_path
    from codescope.index.store import IndexStore

    if not name:
        return {"error": "name is required"}
    with IndexStore(default_db_path(root)) as store:
        sql = (
            "SELECT s.id, s.name, s.kind, s.path, s.start_line, s.end_line, COALESCE(s.signature,''), f.body, fi.lang "
            "FROM symbols AS s "
            "LEFT JOIN symbols_fts AS f ON f.rowid = s.id "
            "JOIN files AS fi ON fi.path = s.path "
            "WHERE s.name = ?"
        )
        params: list[object] = [name]
        if path:
            sql += " AND s.path = ?"
            params.append(path)
        row = store.conn.execute(sql + " ORDER BY s.path, s.start_line LIMIT 1", params).fetchone()
    if row is None:
        return {"error": f"No indexed symbol named {name!r}"}
    return {
        "symbol_id": row[0],
        "name": row[1],
        "kind": row[2],
        "path": row[3],
        "start_line": row[4],
        "end_line": row[5],
        "signature": row[6],
        "code": row[7] or "",
        "lang": row[8] or "",
    }


#: Paths that make the "most referenced" list useless: a bundled jQuery or a
#: language test fixture out-references the actual project many times over.
_OVERVIEW_EXCLUDE = ("*.min.js", "*.min.css", "*/vendor/*", "*/node_modules/*", "*/third_party/*")


def _overview_payload(root: str, limit: int = 40) -> dict:
    """Enough of a starting point that the page is useful before any search."""
    from codescope.index.indexer import Indexer, default_db_path
    from codescope.index.search import SearchFilter, _filter_sql
    from codescope.index.store import IndexStore

    db = default_db_path(root)
    if not db.exists():
        return {"indexed": False, "mode": "none", "health": Indexer(root).health(), "languages": [], "busiest": [], "files": []}

    with IndexStore(db) as store:
        languages = [
            {"lang": lang, "files": files}
            for lang, files in store.conn.execute("SELECT lang, COUNT(*) FROM files GROUP BY lang ORDER BY 2 DESC")
        ]
        # "Busiest" = most referenced, i.e. where a change would ripple from.
        # Tests and vendored bundles are excluded: they are not what someone
        # opening the page wants to see first.
        where, params = _filter_sql(SearchFilter(exclude_tests=True))
        where += " AND NOT s.path GLOB ?" * len(_OVERVIEW_EXCLUDE)
        busiest = [
            {"name": r[0], "kind": r[1], "path": r[2], "start_line": r[3], "callers": r[4]}
            for r in store.conn.execute(
                "SELECT s.name, s.kind, s.path, s.start_line, COUNT(DISTINCT r.owner_id) AS callers "
                "FROM symbols AS s "
                "JOIN files AS fi ON fi.path = s.path "
                "JOIN refs AS r ON r.name = s.name AND r.owner_id IS NOT NULL "
                f"WHERE 1=1{where} "
                "GROUP BY s.name ORDER BY callers DESC, s.name LIMIT ?",
                (*params, *_OVERVIEW_EXCLUDE, limit),
            )
        ]
        # A small or freshly indexed project may have no references at all.
        # An empty panel there reads like "nothing is indexed", so fall back
        # to simply listing what was found.
        starting_mode = "referenced" if busiest else "symbols"
        if not busiest:
            busiest = [
                {"name": r[0], "kind": r[1], "path": r[2], "start_line": r[3], "callers": 0}
                for r in store.conn.execute(
                    "SELECT s.name, s.kind, s.path, s.start_line FROM symbols AS s "
                    "JOIN files AS fi ON fi.path = s.path "
                    f"WHERE 1=1{where} ORDER BY s.path, s.start_line LIMIT ?",
                    (*params, *_OVERVIEW_EXCLUDE, limit),
                )
            ]
        files = [
            {"path": r[0], "lang": r[1], "symbols": r[2]}
            for r in store.conn.execute(
                "SELECT fi.path, fi.lang, COUNT(s.id) FROM files AS fi "
                "LEFT JOIN symbols AS s ON s.path = fi.path "
                "GROUP BY fi.path ORDER BY 3 DESC, fi.path LIMIT ?",
                (limit,),
            )
        ]
    return {
        "indexed": True,
        "mode": starting_mode,
        "health": Indexer(root).health(),
        "languages": languages,
        "busiest": busiest,
        "files": files,
    }


# -- lifecycle --------------------------------------------------------------


class ExplorerServer:
    """The in-process Flask server, when this process is the one hosting it."""

    def __init__(self, registry: ProjectRegistry, port: int, host: str) -> None:
        self.registry = registry
        self.port = port
        self.host = host
        self.url = f"http://{host}:{port}/"
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        from flask import cli

        cli.show_server_banner = lambda *args, **kwargs: None  # type: ignore[assignment]
        app = build_app(self.registry)
        self._thread = threading.Thread(
            target=lambda: app.run(host=self.host, port=self.port, debug=False, use_reloader=False, threaded=True),
            name="codescope-explorer",
            daemon=True,
        )
        self._thread.start()


def stop_this_process(delay_s: float = 0.5) -> None:
    """Terminate the explorer process shortly after the response is sent.

    A separate, patchable function on purpose: it signals the *current*
    process, which any caller that is not a real server (a test, an embedded
    use) must be able to stand in for.
    """
    import signal

    def stop() -> None:
        time.sleep(delay_s)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=stop, name="codescope-explorer-stop", daemon=True).start()


def hosted_registry() -> ProjectRegistry | None:
    """The registry of an explorer hosted by this process, if any."""
    server = _hosted.get("server")
    return server.registry if server else None


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _attach(root: str | Path, port: int, host: str) -> bool:
    """Register ``root`` with an explorer that is already running."""
    import json
    import os
    import urllib.error
    import urllib.request

    payload = json.dumps({"root": str(Path(root).resolve()), "pid": os.getpid()}).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/api/projects",
        data=payload,
        headers={"Content-Type": "application/json", _TOKEN_HEADER: explorer_token()},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_ATTACH_TIMEOUT_S) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def ensure_explorer(
    project_root: str | Path,
    *,
    open_browser: bool = True,
    port: int = EXPLORER_PORT,
    host: str = EXPLORER_HOST,
) -> tuple[str, bool]:
    """Make sure the explorer is running and knows about ``project_root``.

    :return: ``(url, started_here)``. ``started_here`` is False when another
        process was already hosting it, in which case this call only
        registered the project and no second browser tab is opened.
    """
    url = f"http://{host}:{port}/"

    with _server_lock:
        running = _hosted.get("server")
        if running is not None:
            running.registry.register(project_root)
            return running.url, False

        if _attach(project_root, port, host):
            log.info("Explorer already running at %s; attached this project.", url)
            return url, False

        if not _port_is_free(host, port):
            # The port is taken but did not accept a registration. Usually
            # that is an explorer still starting up in another process, so
            # give it a moment before concluding it is something else.
            for _ in range(_ATTACH_RETRIES):
                time.sleep(_ATTACH_RETRY_DELAY_S)
                if _attach(project_root, port, host):
                    log.info("Explorer at %s accepted the project on retry.", url)
                    return url, False
                if _port_is_free(host, port):
                    break
            else:
                log.warning("Port %d is in use but is not a Codescope explorer; not starting one.", port)
                return url, False

        registry = ProjectRegistry()
        registry.register(project_root, os.getpid())
        server = ExplorerServer(registry, port, host)
        server.start()
        _hosted["server"] = server

    if not _wait_until_serving(host, port):
        log.warning("Explorer did not come up on %s", url)
        return url, True
    log.info("Codescope explorer running at %s", url)
    if open_browser:
        _open_browser(url)
    return url, True


def _wait_until_serving(host: str, port: int, timeout_s: float = 5.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def _open_browser(url: str) -> None:
    """Open the UI without writing anything to stdout.

    A stdio MCP server shares stdout with the protocol, and some browser
    launchers print to it, so the launch happens in a separate process.
    """
    import subprocess
    import sys

    try:
        process = subprocess.Popen(
            [sys.executable, "-c", f"import webbrowser; webbrowser.open({url!r})"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        threading.Thread(target=process.wait, daemon=True).start()
    except Exception as e:  # pragma: no cover - platform dependent
        log.warning("Could not open the explorer in a browser: %s", e)
