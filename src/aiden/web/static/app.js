/* AIDEN Explorer.
 *
 * No framework and no bundler: the page is served by a local Flask thread and
 * has to work offline, so everything here is plain DOM.
 */

const state = {
  projectId: null,
  projectsVersion: -1,
  focus: null,          // {name, path}
  view: "symbol",       // symbol | project
  history: [],
  searchToken: 0,
  tab: "search",
  /* How to rebuild the left panel exactly as it stands now -- including a
   * drilled-into file or a typed query -- so a live refresh does not throw the
   * user back to the top of the tab. */
  reload: null,
  indexRevision: null,
  pendingRevision: null,
  refreshing: false,
};

const el = (id) => document.getElementById(id);

const dom = {
  projectSelect: el("project-select"),
  search: el("search-input"),
  excludeTests: el("exclude-tests"),
  themeToggle: el("theme-toggle"),
  resultList: el("result-list"),
  resultsTitle: el("results-title"),
  resultsCount: el("results-count"),
  resultsEmpty: el("results-empty"),
  graphCanvas: el("graph-canvas"),
  graphTitle: el("graph-title"),
  graphNote: el("graph-note"),
  graphBack: el("graph-back"),
  detailBody: el("detail-body"),
  detailTitle: el("detail-title"),
  detailKind: el("detail-kind"),
  statusIndex: el("status-index"),
  statusEmbed: el("status-embed"),
  statusAdvice: el("status-advice"),
  statusProjects: el("status-projects"),
  statusProgress: el("status-progress"),
  progressFill: el("progress-fill"),
  progressLabel: el("progress-label"),
  statusWatch: el("status-watch"),
  statusWatchLabel: el("status-watch-label"),
  watchToggle: el("watch-toggle"),
  tip: el("tip"),
};

/* ---------- helpers ---------- */

async function api(path, params) {
  const url = new URL(path, location.origin);
  Object.entries(params || {}).forEach(([k, v]) => {
    if (v !== null && v !== undefined && v !== "") url.searchParams.set(k, v);
  });
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

/* Handed to the page in its own HTML, which a cross-origin caller cannot
 * read. Every state-changing request carries it. */
const TOKEN = document.querySelector('meta[name="aiden-token"]')?.content || "";

function post(path, body) {
  return fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-AIDEN-Token": TOKEN },
    body: JSON.stringify(body || {}),
  });
}

function projectPath(suffix) {
  return `/api/projects/${state.projectId}/${suffix}`;
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function shortPath(path) {
  return path.length > 46 ? "…" + path.slice(-45) : path;
}

/* ---------- projects ---------- */

async function refreshProjects() {
  let data;
  try {
    data = await api("/api/projects");
  } catch {
    return;
  }
  renderProjectCount(data.projects.length);

  if (data.version === state.projectsVersion) return;
  state.projectsVersion = data.version;

  const previous = state.projectId;
  dom.projectSelect.replaceChildren(
    ...data.projects.map((project) => {
      const option = node("option", null, project.name);
      option.value = project.id;
      option.title = project.root;
      return option;
    })
  );
  const stillThere = data.projects.some((p) => p.id === previous);
  dom.projectSelect.value = stillThere ? previous : (data.projects[0]?.id ?? "");
  if (dom.projectSelect.value && dom.projectSelect.value !== previous) selectProject(dom.projectSelect.value);
}

async function selectProject(projectId) {
  state.projectId = projectId;
  state.focus = null;
  state.history = [];
  state.indexRevision = null;
  state.pendingRevision = null;
  dom.graphBack.disabled = true;
  dom.search.value = "";
  clearGraph("Pick a symbol to see what calls it and what it calls.");
  dom.detailBody.replaceChildren(node("p", "empty", "Select a symbol."));
  dom.detailTitle.textContent = "Details";
  dom.detailKind.textContent = "";
  await Promise.all([loadOverview(), loadStatus()]);
}

/* ---------- status bar ---------- */

async function loadStatus() {
  let health;
  try {
    health = await api(projectPath("status"));
  } catch {
    return;
  }
  const dot = (cls) => {
    const span = node("span", `status-dot${cls ? " " + cls : ""}`);
    return span;
  };

  dom.statusIndex.replaceChildren(
    dot(health.indexed ? "" : "bad"),
    node("span", null, health.indexed ? `${health.symbols} symbols · ${health.files} files` : "no index yet")
  );

  if (!health.indexed) {
    dom.statusEmbed.textContent = "";
  } else if (health.semantic_search_ready) {
    dom.statusEmbed.replaceChildren(dot(), node("span", null, "semantic search ready"));
    dom.statusEmbed.className = "status-item";
  } else {
    dom.statusEmbed.replaceChildren(dot("warn"), node("span", null, "lexical only"));
    dom.statusEmbed.className = "status-item warn";
  }

  const advice = (health.advice || [])[0] || "";
  dom.statusAdvice.textContent = advice;
  dom.statusAdvice.dataset.tip = (health.advice || []).join("\n");
  dom.statusAdvice.className = advice ? "status-item warn" : "status-item";

  renderLiveState(health);
}

/* ---------- live index state ---------- */

/* The index run and the watcher are the two things that move on their own, so
 * they are polled separately from the (much heavier) health payload. */

const PHASE_LABEL = {
  starting: "starting",
  scanning: "scanning files",
  parsing: "indexing",
  pruning: "pruning",
  embedding: "embedding",
  done: "indexed",
  failed: "failed",
};

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) return "";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds % 60)}s`;
}

function renderLiveState(live) {
  renderProgress(live.progress);
  renderWatcher(live.watcher, live.running_actions || []);
}

function renderProgress(run) {
  if (!run || (!run.running && run.phase === "done" && run.elapsed_s > 20)) {
    dom.statusProgress.hidden = true;
    return;
  }
  dom.statusProgress.hidden = false;
  const phase = PHASE_LABEL[run.phase] || run.phase;
  const bar = dom.progressFill.parentElement;
  const known = run.percent !== null && run.percent !== undefined;

  bar.classList.toggle("indeterminate", run.running && !known);
  dom.progressFill.style.width = known ? `${run.percent}%` : "";
  dom.statusProgress.className = `status-item progress-item${run.running ? "" : " " + run.phase}`;

  let label;
  if (!run.running && run.error) {
    label = `${run.operation} failed`;
  } else if (!run.running) {
    label = `${run.operation} finished in ${formatDuration(run.elapsed_s)}`;
  } else if (run.total) {
    label = `${phase} ${run.done}/${run.total}`;
    if (run.eta_s !== null && run.eta_s !== undefined) label += ` · ${formatDuration(run.eta_s)} left`;
  } else {
    label = phase;
  }
  dom.progressLabel.textContent = label;

  const lines = [`${run.operation}: ${phase}`];
  if (run.total) lines.push(`${run.done} of ${run.total}`);
  if (run.detail) lines.push(run.detail);
  lines.push(`running for ${formatDuration(run.elapsed_s)}`);
  if (run.error) lines.push(run.error);
  dom.statusProgress.dataset.tip = lines.join("\n");
}

function renderWatcher(watcher, runningActions) {
  const running = Boolean(watcher && watcher.running);
  dom.statusWatch.hidden = !running;
  if (running) {
    const parts = [`${watcher.files_reindexed} file(s) reindexed in ${watcher.batches} batch(es)`];
    if (watcher.uptime_s) parts.push(`up for ${formatDuration(watcher.uptime_s)}`);
    if (watcher.error) parts.push(`last error: ${watcher.error}`);
    dom.statusWatchLabel.textContent = "watching";
    dom.statusWatch.dataset.tip = `A watcher keeps this index fresh.\n${parts.join("\n")}`;
  }

  // One button, two states: pressing it again stops the watcher rather than
  // trying (and failing) to start a second one.
  if (dom.watchToggle) {
    dom.watchToggle.classList.toggle("is-active", running);
    dom.watchToggle.setAttribute("aria-pressed", running ? "true" : "false");
    dom.watchToggle.textContent = running ? "Watching" : "Watch";
    dom.watchToggle.dataset.action = running ? "watch_stop" : "watch_start";
    dom.watchToggle.disabled = runningActions.includes("watch_start") || runningActions.includes("watch_stop");
    dom.watchToggle.dataset.tip = running
      ? "A watcher is keeping this index fresh. Click to stop it."
      : "Keep the index fresh in the background: every saved file is reindexed automatically.";
  }

  document.querySelectorAll('.actions [data-action="sync"], .actions [data-action="reindex"]').forEach((button) => {
    const busy = runningActions.includes("reindex") || runningActions.includes("sync");
    button.disabled = busy;
    if (busy) button.dataset.tip = "An index run is already in progress for this project.";
  });
}

async function loadLiveState() {
  if (!state.projectId) return;
  let live;
  try {
    live = await api(projectPath("progress"));
  } catch {
    return; /* the server may be shutting down; the next tick will tell us */
  }
  renderLiveState(live);
  noteIndexRevision(live);
}

/* ---------- keeping the panels live ---------- */

/* The index changes under the page all the time: the watcher reindexes saved
 * files, an agent calls reindex over MCP, another instance syncs. None of that
 * went through this page, so nothing here used to notice -- the results,
 * the graph and the detail pane kept showing whatever was true when they were
 * last drawn. They now follow the index revision, which is a stat of the
 * database and therefore catches every writer.
 *
 * A run in flight writes continuously, so a refresh waits until the revision
 * has held still for a poll tick and no run is active. Otherwise a full
 * reindex would redraw the panels once a second for several minutes. */

function noteIndexRevision(live) {
  const revision = live.index_revision;
  if (revision === null || revision === undefined) return;

  if (state.indexRevision === null) {
    state.indexRevision = revision;
    return;
  }
  if (revision === state.indexRevision) {
    state.pendingRevision = null;
    return;
  }
  if (live.progress && live.progress.running) {
    state.pendingRevision = null; // settle once the run is over
    return;
  }
  if (state.pendingRevision !== revision) {
    state.pendingRevision = revision; // seen once; confirm on the next tick
    return;
  }
  state.indexRevision = revision;
  state.pendingRevision = null;
  refreshPanels();
}

async function refreshPanels() {
  if (state.refreshing) return;
  state.refreshing = true;
  const scrollTop = dom.resultList.scrollTop;
  try {
    await Promise.all([
      state.reload ? state.reload() : loadOverview(),
      state.focus ? focusSymbol(state.focus.name, state.focus.path) : Promise.resolve(),
      loadStatus(),
    ]);
    dom.resultList.scrollTop = scrollTop;
  } catch (error) {
    /* A refresh nobody asked for must not replace the panel with an error;
     * the next change will try again. */
    console.warn("Live refresh failed:", error);
  } finally {
    state.refreshing = false;
  }
}

/* ---------- results ---------- */

async function loadOverview() {
  state.reload = loadOverview;
  const data = await api(projectPath("overview"));
  dom.resultsTitle.textContent = data.mode === "symbols" ? "Symbols" : "Most referenced";
  const rows = (data.busiest || []).map((entry) => ({
    name: entry.name,
    kind: entry.kind,
    path: entry.path,
    start_line: entry.start_line,
    callers: entry.callers,
  }));
  const empty = data.indexed
    ? "This project is indexed but has no symbols outside test files."
    : "No index for this project yet. Run reindex (or `aiden index build`), then reload.";
  renderResults(rows, empty);
}

async function runSearch(query) {
  const token = ++state.searchToken;
  if (!query.trim()) {
    await loadOverview();
    return;
  }
  state.reload = () => runSearch(query);
  const data = await api(projectPath("search"), {
    q: query,
    limit: 30,
    exclude_tests: dom.excludeTests.checked ? "1" : "",
  });
  if (token !== state.searchToken) return; // a newer search already answered
  dom.resultsTitle.textContent = "Results";
  renderResults(data.hits || [], "Nothing matched. Try fewer words, or check the index status below.");
}

function renderResults(rows, emptyMessage) {
  dom.resultsCount.textContent = rows.length ? String(rows.length) : "";
  dom.resultsEmpty.textContent = emptyMessage;
  dom.resultsEmpty.hidden = rows.length > 0;
  dom.resultList.replaceChildren(
    ...rows.map((row) => {
      const item = node("li");
      const button = node("button", "result");
      button.type = "button";
      button.dataset.name = row.name;
      button.dataset.path = row.path;

      const line = node("div", "result-name");
      line.append(node("b", null, row.name), node("span", "kind", row.kind || "symbol"));
      if (row.callers) line.append(node("span", "callers-badge", `${row.callers} in`));
      button.append(line, node("div", "result-path", `${shortPath(row.path)}:${row.start_line}`));
      button.dataset.tip = [
        `${row.kind || "symbol"} ${row.name}`,
        `${row.path}:${row.start_line}`,
        row.callers ? `${row.callers} caller(s) in this project` : "",
        "Click to see what calls it and what it calls.",
      ]
        .filter(Boolean)
        .join("\n");
      button.addEventListener("click", () => focusSymbol(row.name, row.path, { push: true }));
      item.append(button);
      return item;
    })
  );
}

function markCurrentResult() {
  dom.resultList.querySelectorAll(".result").forEach((button) => {
    const current = state.focus && button.dataset.name === state.focus.name;
    button.setAttribute("aria-current", current ? "true" : "false");
  });
}

/* ---------- graph ---------- */

function clearGraph(message) {
  dom.graphCanvas.replaceChildren(node("p", "empty centered", message));
  dom.graphTitle.textContent = "Call graph";
  dom.graphNote.hidden = true;
}

async function focusSymbol(name, path, options = {}) {
  if (options.push && state.focus) state.history.push(state.focus);
  state.focus = { name, path };
  dom.graphBack.disabled = state.history.length === 0;
  markCurrentResult();

  const [graph, detail] = await Promise.all([
    api(projectPath("graph"), { symbol: name }),
    api(projectPath("symbol"), { name, path }),
  ]);
  renderGraph(graph);
  renderDetail(detail, graph);
}

function renderGraph(graph) {
  dom.graphTitle.textContent = graph.symbol;

  const callers = graph.callers || [];
  const callees = graph.callees || [];
  if (!callers.length && !callees.length) {
    clearGraph(`Nothing references ${graph.symbol}, and it references nothing indexed.`);
    dom.graphTitle.textContent = graph.symbol;
  } else {
    dom.graphCanvas.replaceChildren(buildGraphSvg(graph.symbol, callers, callees));
  }

  if (graph.ambiguous) {
    const places = (graph.definitions || []).map((d) => `${d.path}:${d.start_line}`).join(", ");
    dom.graphNote.textContent =
      `${graph.symbol} is defined ${graph.definitions.length} times (${places}). ` +
      "Edges are matched by name, so they may belong to any of them.";
    dom.graphNote.hidden = false;
  } else {
    dom.graphNote.hidden = true;
  }
}

/* A hub symbol can have hundreds of callers. Drawing them all produces a
 * wall of boxes nobody can read and a multi-thousand-pixel SVG, so each side
 * is capped and the remainder is reported as a count. */
const MAX_NODES_PER_SIDE = 12;
const MAX_CHIPS = 24;

const NODE_W = 208;
const NODE_H = 42;
const GAP_Y = 12;
const COL_GAP = 128;

function buildGraphSvg(focusName, allCallers, allCallees) {
  const svgNS = "http://www.w3.org/2000/svg";
  const callers = allCallers.slice(0, MAX_NODES_PER_SIDE);
  const callees = allCallees.slice(0, MAX_NODES_PER_SIDE);
  const hiddenIn = allCallers.length - callers.length;
  const hiddenOut = allCallees.length - callees.length;
  const rows = Math.max(callers.length, callees.length, 1);
  const height = Math.max(rows * (NODE_H + GAP_Y) + 60, 220);
  const width = NODE_W * 3 + COL_GAP * 2 + 24;

  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const columnX = [12, 12 + NODE_W + COL_GAP, 12 + (NODE_W + COL_GAP) * 2];
  const centerY = height / 2 - NODE_H / 2;

  const label = (text, x) => {
    const node = document.createElementNS(svgNS, "text");
    node.setAttribute("class", "column-label");
    node.setAttribute("x", String(x));
    node.setAttribute("y", "18");
    node.textContent = text;
    return node;
  };
  if (callers.length) svg.append(label(`calls in (${allCallers.length})`, columnX[0]));
  svg.append(label("symbol", columnX[1]));
  if (callees.length) svg.append(label(`calls out (${allCallees.length})`, columnX[2]));

  const stack = (items, x) =>
    items.map((item, index) => ({
      item,
      x,
      y: height / 2 - (items.length * (NODE_H + GAP_Y) - GAP_Y) / 2 + index * (NODE_H + GAP_Y),
    }));

  const left = stack(callers, columnX[0]);
  const right = stack(callees, columnX[2]);

  for (const { y } of left) svg.append(edge(svgNS, columnX[0] + NODE_W, y + NODE_H / 2, columnX[1], centerY + NODE_H / 2, "in"));
  for (const { y } of right) svg.append(edge(svgNS, columnX[1] + NODE_W, centerY + NODE_H / 2, columnX[2], y + NODE_H / 2, "out"));

  for (const { item, x, y } of left) svg.append(graphNode(svgNS, item, x, y, "in"));
  for (const { item, x, y } of right) svg.append(graphNode(svgNS, item, x, y, "out"));
  svg.append(graphNode(svgNS, { name: focusName, path: "", kind: "" }, columnX[1], centerY, "focus"));

  const more = (count, x) => {
    if (!count) return;
    const text = document.createElementNS(svgNS, "text");
    text.setAttribute("class", "column-label");
    text.setAttribute("x", String(x));
    text.setAttribute("y", String(height - 10));
    text.textContent = `+ ${count} more (listed on the right)`;
    svg.append(text);
  };
  more(hiddenIn, columnX[0]);
  more(hiddenOut, columnX[2]);

  return svg;
}

function edge(svgNS, x1, y1, x2, y2, kind) {
  const path = document.createElementNS(svgNS, "path");
  const mid = (x1 + x2) / 2;
  path.setAttribute("d", `M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`);
  path.setAttribute("class", `edge ${kind}`);
  return path;
}

function graphNode(svgNS, item, x, y, kind) {
  const group = document.createElementNS(svgNS, "g");
  group.setAttribute("class", `node ${kind}`);
  group.setAttribute("transform", `translate(${x},${y})`);

  const box = document.createElementNS(svgNS, "rect");
  box.setAttribute("class", "node-box");
  box.setAttribute("width", String(NODE_W));
  box.setAttribute("height", String(NODE_H));
  group.append(box);

  const name = document.createElementNS(svgNS, "text");
  name.setAttribute("class", "node-label");
  name.setAttribute("x", "11");
  name.setAttribute("y", item.path ? "18" : "26");
  name.textContent = truncate(item.name, 24);
  group.append(name);

  if (item.path) {
    const sub = document.createElementNS(svgNS, "text");
    sub.setAttribute("class", "node-sub");
    sub.setAttribute("x", "11");
    sub.setAttribute("y", "32");
    sub.textContent = truncate(item.path.split("/").pop() + ":" + item.start_line, 30);
    group.append(sub);
  }

  const title = document.createElementNS(svgNS, "title");
  title.textContent = item.path ? `${item.name} — ${item.path}:${item.start_line}` : item.name;
  group.append(title);

  if (kind !== "focus") {
    group.addEventListener("click", () => focusSymbol(item.name, item.path, { push: true }));
  }
  return group;
}

function truncate(text, max) {
  return text.length > max ? text.slice(0, max - 1) + "…" : text;
}

/* ---------- detail ---------- */

function renderDetail(detail, graph) {
  if (detail.error) {
    dom.detailBody.replaceChildren(node("p", "empty", detail.error));
    return;
  }
  dom.detailTitle.textContent = "Details";
  dom.detailKind.textContent = detail.kind || "";

  const parts = [
    node("p", "detail-name", detail.name),
    node("p", "detail-loc", `${detail.path}:${detail.start_line}–${detail.end_line}${detail.lang ? " · " + detail.lang : ""}`),
  ];
  if (detail.code) parts.push(Object.assign(node("pre", "code"), { textContent: detail.code }));

  const section = (title, items, kind) => {
    if (!items.length) return null;
    const wrapper = node("div", "detail-section");
    wrapper.append(node("h3", null, title));
    const row = node("div", "chip-row");
    let shown = MAX_CHIPS;

    const fill = () => {
      row.replaceChildren();
      items.slice(0, shown).forEach((item) => {
        const chip = node("button", `chip ${kind}`, item.name);
        chip.type = "button";
        chip.title = `${item.path}:${item.start_line}`;
        chip.addEventListener("click", () => focusSymbol(item.name, item.path, { push: true }));
        row.append(chip);
      });
      if (items.length > shown) {
        const more = node("button", "chip more", `+${items.length - shown} more`);
        more.type = "button";
        more.addEventListener("click", () => {
          shown = items.length;
          fill();
        });
        row.append(more);
      }
    };

    fill();
    wrapper.append(row);
    return wrapper;
  };

  const callers = section(`Called by (${graph.callers.length})`, graph.callers, "in");
  const callees = section(`Calls (${graph.callees.length})`, graph.callees, "out");
  if (callers) parts.push(callers);
  if (callees) parts.push(callees);

  dom.detailBody.replaceChildren(...parts);
}

/* ---------- wiring ---------- */

let searchTimer = null;
dom.search.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    document.querySelectorAll(".tab").forEach((tab) => tab.setAttribute("aria-selected", String(tab.dataset.tab === "search")));
    runSearch(dom.search.value);
  }, 160);
});

dom.excludeTests.addEventListener("change", () => runSearch(dom.search.value));
dom.projectSelect.addEventListener("change", (event) => selectProject(event.target.value));

dom.graphBack.addEventListener("click", () => {
  const previous = state.history.pop();
  dom.graphBack.disabled = state.history.length === 0;
  if (previous) focusSymbol(previous.name, previous.path);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "/" && document.activeElement !== dom.search) {
    event.preventDefault();
    dom.search.focus();
    dom.search.select();
  }
  if (event.key === "Escape" && document.activeElement === dom.search) dom.search.blur();
});

dom.themeToggle.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem("aiden-theme", next);
  } catch {
    /* private mode: the choice simply does not persist */
  }
});

try {
  const saved = localStorage.getItem("aiden-theme");
  if (saved) document.documentElement.dataset.theme = saved;
  else if (window.matchMedia("(prefers-color-scheme: light)").matches) document.documentElement.dataset.theme = "light";
} catch {
  /* ignore */
}

refreshProjects();
setInterval(refreshProjects, 4000);
setInterval(() => state.projectId && loadStatus(), 15000);
setInterval(loadLiveState, 1000);

/* ---------- project count ---------- */

function renderProjectCount(count) {
  document.getElementById("project-count-value").textContent = String(count);
  dom.statusProjects.textContent = count === 1 ? "1 project attached" : `${count} projects attached`;
  if (projectsDialog?.open) renderKnownProjects();
}

/* ---------- panel resizing ---------- */

/* Grid columns are driven by two CSS custom properties, so a drag is a
 * single style write and the browser handles the rest. Widths are stored
 * per browser, not per project: it is a preference about this screen. */
function setupResizers() {
  const layout = document.querySelector(".layout");
  const MIN = 200;

  const load = () => {
    try {
      const saved = JSON.parse(localStorage.getItem("aiden-columns") || "null");
      if (saved?.left) layout.style.setProperty("--col-left", saved.left);
      if (saved?.right) layout.style.setProperty("--col-right", saved.right);
    } catch {
      /* ignore */
    }
  };

  const save = () => {
    try {
      localStorage.setItem(
        "aiden-columns",
        JSON.stringify({
          left: layout.style.getPropertyValue("--col-left"),
          right: layout.style.getPropertyValue("--col-right"),
        })
      );
    } catch {
      /* ignore */
    }
  };

  const widthOf = (which) => {
    const panel = document.querySelector(which === "left" ? ".results" : ".detail");
    return panel.getBoundingClientRect().width;
  };

  const apply = (which, px) => {
    const total = layout.getBoundingClientRect().width;
    const other = widthOf(which === "left" ? "right" : "left");
    const clamped = Math.max(MIN, Math.min(px, total - other - MIN - 40));
    layout.style.setProperty(which === "left" ? "--col-left" : "--col-right", `${Math.round(clamped)}px`);
  };

  layout.querySelectorAll(".resizer").forEach((handle) => {
    const which = handle.dataset.resize;

    handle.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      handle.setPointerCapture(event.pointerId);
      handle.dataset.dragging = "true";
      document.body.dataset.resizing = "true";
      const startX = event.clientX;
      const startWidth = widthOf(which);

      const move = (moveEvent) => {
        const delta = moveEvent.clientX - startX;
        apply(which, which === "left" ? startWidth + delta : startWidth - delta);
      };
      const up = () => {
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", up);
        delete handle.dataset.dragging;
        delete document.body.dataset.resizing;
        save();
      };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", up);
    });

    // Keyboard: the handles are focusable separators, so they must move too.
    handle.addEventListener("keydown", (event) => {
      const step = event.shiftKey ? 48 : 16;
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const delta = event.key === "ArrowLeft" ? -step : step;
      apply(which, which === "left" ? widthOf(which) + delta : widthOf(which) - delta);
      save();
    });
  });

  load();
}

/* ---------- actions ---------- */

let lastEventId = 0;
let toastTimer = null;

function toast(message, kind) {
  const box = document.getElementById("action-toast");
  box.textContent = message;
  box.className = `action-toast ${kind || ""}`.trim();
  box.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (box.hidden = true), 6000);
}

async function runAction(action, button) {
  if (!state.projectId || button.disabled) return;
  button.disabled = true;
  try {
    const response = await post(projectPath(`actions/${action}`), {});
    const body = await response.json().catch(() => ({}));
    if (response.status === 409) {
      // Nothing was started, and saying "started…" here is how you end up
      // pressing Watch four times and believing you have four watchers.
      toast(body.detail || `${action} is already running`, "skipped");
    } else if (!response.ok) {
      toast(body.error || `${action} could not be started`, "failed");
    } else {
      toast(`${action} started…`);
    }
  } catch (error) {
    toast(`${action} could not be started: ${error.message}`, "failed");
  } finally {
    await loadLiveState();
    if (button.id !== "watch-toggle") setTimeout(() => (button.disabled = false), 600);
  }
}

async function pollEvents() {
  let data;
  try {
    data = await api("/api/events", { since: lastEventId });
  } catch {
    return;
  }
  for (const event of data.events || []) {
    lastEventId = Math.max(lastEventId, event.id);
    if (event.status === "running") continue;
    toast(`${event.action} on ${event.project}: ${event.detail}`, event.status);
    loadLiveState();
    if (event.status === "done" && event.action !== "watch_stop") {
      refreshPanels();
    }
  }
}

document.querySelectorAll(".actions [data-action]").forEach((button) => {
  button.addEventListener("click", () => runAction(button.dataset.action, button));
});

/* The watch toggle re-enables itself from the polled watcher state, not from a
 * timer, so it never re-arms while the watcher is still coming up. */

/* ---------- whole-project graph ---------- */

const viewButtons = { symbol: document.getElementById("view-symbol"), project: document.getElementById("view-project") };

function setView(view) {
  state.view = view;
  viewButtons.symbol.setAttribute("aria-pressed", String(view === "symbol"));
  viewButtons.project.setAttribute("aria-pressed", String(view === "project"));
  const perSymbol = view === "symbol";
  document.getElementById("legend-in").hidden = !perSymbol;
  document.getElementById("legend-out").hidden = !perSymbol;
  dom.graphBack.hidden = !perSymbol;
  if (view === "project") loadProjectGraph();
  else if (state.focus) focusSymbol(state.focus.name, state.focus.path);
  else clearGraph("Pick a symbol to see what calls it and what it calls.");
}

viewButtons.symbol.addEventListener("click", () => setView("symbol"));
viewButtons.project.addEventListener("click", () => setView("project"));

async function loadProjectGraph() {
  const data = await api(projectPath("project-graph"), { level: "dir" });
  dom.graphTitle.textContent = "Whole project";
  if (!data.nodes.length) {
    clearGraph("No cross-directory references indexed yet.");
    dom.graphTitle.textContent = "Whole project";
    return;
  }
  dom.graphCanvas.replaceChildren(buildProjectSvg(data));
  dom.graphNote.textContent = data.truncated
    ? "Showing the most connected areas only. Edges count references whose target name is defined exactly once, so ambiguous names are left out."
    : "Edges count references whose target name is defined exactly once, so ambiguous names are left out.";
  dom.graphNote.hidden = false;
}

function buildProjectSvg(data) {
  const svgNS = "http://www.w3.org/2000/svg";
  const size = Math.max(560, Math.min(920, 220 + data.nodes.length * 34));
  const cx = size / 2;
  const cy = size / 2;
  const radius = size / 2 - 96;

  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("viewBox", `0 0 ${size} ${size}`);

  // A ring keeps every node visible and every edge a readable chord; a
  // force layout of this many hubs collapses into a hairball.
  const placed = new Map();
  data.nodes.forEach((entry, index) => {
    const angle = (index / data.nodes.length) * Math.PI * 2 - Math.PI / 2;
    placed.set(entry.id, { ...entry, x: cx + Math.cos(angle) * radius, y: cy + Math.sin(angle) * radius, angle });
  });

  const maxWeight = Math.max(...data.edges.map((e) => e.weight), 1);
  data.edges.forEach((edge) => {
    const from = placed.get(edge.source);
    const to = placed.get(edge.target);
    if (!from || !to) return;
    const path = document.createElementNS(svgNS, "path");
    path.setAttribute("d", `M${from.x},${from.y} Q${cx},${cy} ${to.x},${to.y}`);
    path.setAttribute("class", "pedge");
    path.setAttribute("stroke-width", String(0.6 + (edge.weight / maxWeight) * 4));
    const title = document.createElementNS(svgNS, "title");
    title.textContent = `${edge.source} → ${edge.target} (${edge.weight} references)`;
    path.append(title);
    svg.append(path);
  });

  const maxSymbols = Math.max(...data.nodes.map((n) => n.symbols), 1);
  placed.forEach((entry) => {
    const group = document.createElementNS(svgNS, "g");
    group.setAttribute("class", "pnode");
    group.setAttribute("transform", `translate(${entry.x},${entry.y})`);

    const circle = document.createElementNS(svgNS, "circle");
    circle.setAttribute("r", String(7 + Math.sqrt(entry.symbols / maxSymbols) * 13));
    group.append(circle);

    const onLeft = Math.cos(entry.angle) < 0;
    const label = document.createElementNS(svgNS, "text");
    label.setAttribute("x", onLeft ? "-26" : "26");
    label.setAttribute("y", "0");
    label.setAttribute("text-anchor", onLeft ? "end" : "start");
    label.textContent = entry.id;
    group.append(label);

    const count = document.createElementNS(svgNS, "text");
    count.setAttribute("class", "pnode-count");
    count.setAttribute("x", onLeft ? "-26" : "26");
    count.setAttribute("y", "13");
    count.setAttribute("text-anchor", onLeft ? "end" : "start");
    count.textContent = `${entry.symbols} symbols`;
    group.append(count);

    const title = document.createElementNS(svgNS, "title");
    title.textContent = `${entry.id} — ${entry.symbols} symbols, ${entry.degree} crossing references`;
    group.append(title);

    // Clicking an area filters the result list to it, which is the natural
    // way to go from "where is the weight" to "what is in there".
    group.addEventListener("click", () => {
      dom.search.value = "";
      filterByArea(entry.id);
    });
    svg.append(group);
  });

  return svg;
}

async function filterByArea(area) {
  const data = await api(projectPath("search"), { q: area.split("/").pop(), limit: 30, path_glob: `${area}/*` });
  dom.resultsTitle.textContent = area;
  renderResults(data.hits || [], `Nothing indexed under ${area}.`);
}

setupResizers();
setView("symbol");
setInterval(pollEvents, 2500);

/* ---------- left-panel tabs ---------- */

/* Each tab is a different question about the same index, so they share the
 * result list rather than each getting its own panel. */
const tabs = {
  async search() {
    dom.search.disabled = false;
    await runSearch(dom.search.value);
  },

  async files() {
    dom.search.disabled = true;
    state.reload = tabs.files;
    const data = await api(projectPath("files"));
    dom.resultsTitle.textContent = "Files";
    setCrumbs(null);
    renderRows(
      data.files.map((file) => ({
        title: file.path.split("/").pop(),
        subtitle: file.path,
        badge: `${file.symbols} sym`,
        kind: file.lang,
        onClick: () => openFile(file.path),
      })),
      "No files indexed."
    );
  },

  async clones() {
    dom.search.disabled = true;
    state.reload = tabs.clones;
    dom.resultsTitle.textContent = "Clones";
    setCrumbs(null);
    renderRows([], "Looking for duplicated code…");
    const data = await api(projectPath("clones"));
    if (data.error) {
      renderRows([], data.error);
      return;
    }
    const rows = [];
    data.groups.forEach((group, index) => {
      group.members.forEach((member) => {
        rows.push({
          title: member.name,
          subtitle: `${member.path}:${member.start_line} · group ${index + 1} · ${member.lines} lines`,
          badge: `${(group.similarity * 100).toFixed(0)}%`,
          kind: member.kind,
          onClick: () => focusSymbol(member.name, member.path, { push: true }),
        });
      });
    });
    renderRows(rows, `No clusters above ${Math.round(data.similarity * 100)}% similarity with at least ${data.min_lines} lines.`);
  },

  async health() {
    dom.search.disabled = true;
    state.reload = tabs.health;
    dom.resultsTitle.textContent = "Health";
    setCrumbs(null);
    const health = await api(projectPath("status"));
    renderLiveState(health);
    const list = node("dl", "health-list");
    const add = (key, value, tip) => {
      const row = node("div", "health-row");
      row.append(node("dt", null, key), node("dd", null, String(value)));
      if (tip) row.dataset.tip = tip;
      list.append(row);
    };

    const run = health.progress;
    if (run) {
      const phase = PHASE_LABEL[run.phase] || run.phase;
      const counts = run.total ? ` ${run.done}/${run.total}` : "";
      add(
        run.running ? "index run" : "last index run",
        run.running
          ? `${run.operation}: ${phase}${counts}`
          : `${run.operation} ${run.error ? "failed" : "finished"} after ${formatDuration(run.elapsed_s)}`,
        run.error || run.detail || "The index run this server most recently started."
      );
    }
    add(
      "watcher",
      health.watcher && health.watcher.running ? "running" : "not running",
      health.watcher && health.watcher.running
        ? `${health.watcher.files_reindexed} file(s) reindexed in ${health.watcher.batches} batch(es)`
        : "Press Watch to keep this index fresh automatically."
    );

    add("indexed", health.indexed ? "yes" : "no", "Whether this project has an index at all.");
    if (health.indexed) {
      add("files", health.files, "Source files currently represented in the index.");
      add("symbols", health.symbols, "Definitions (functions, classes, methods …) extracted from those files.");
      add("references", health.refs, "Call and usage sites, which is what the call graph is built from.");
      add("vectors", health.vectors, "Embedded symbols. Semantic search and clone detection need these.");
      add("symbols without a vector", health.symbols_missing_vectors, "Reindexing resumes the backfill where it stopped.");
      add("embedder", health.embedder || "none", "The embedding model this index was built with. Changing it forces a full re-embed.");
      add("semantic search", health.semantic_search_ready ? "ready" : "unavailable", "Searching by meaning rather than by name.");
      add("clone detection", health.clone_detection_ready ? "ready" : "unavailable", "Finding near-duplicate code needs real vectors.");
      const languages = Object.entries(health.languages || {}).sort((a, b) => b[1] - a[1]);
      add(
        "languages",
        languages.map(([name]) => name).join(", ") || "none",
        languages.map(([name, count]) => `${name}: ${count} file(s)`).join("\n") ||
          "No language was recognised in this project."
      );
      add("uncommitted changes", health.uncommitted_changes ?? "not a git repo", "Changed source files git knows about but the index does not.");
      add(
        "HEAD moved since index",
        health.head_moved_since_index === null ? "unknown" : String(health.head_moved_since_index),
        "A clean tree can still be stale: committing and switching branch leaves no dirty files."
      );
      add("database", `${(health.db_size_bytes / 1048576).toFixed(1)} MB`, health.db_path || "");
    }
    dom.resultList.replaceChildren();
    dom.resultsEmpty.hidden = true;
    dom.resultsCount.textContent = "";
    const wrapper = node("li");
    wrapper.append(list);
    (health.advice || []).forEach((line) => wrapper.append(node("div", "health-advice", line)));
    dom.resultList.append(wrapper);
  },
};

function setCrumbs(element) {
  const existing = document.querySelector(".crumbs");
  if (existing) existing.remove();
  if (element) dom.resultList.before(element);
}

/* Generic row renderer shared by the tabs that are not symbol hits. */
function renderRows(rows, emptyMessage) {
  dom.resultsCount.textContent = rows.length ? String(rows.length) : "";
  dom.resultsEmpty.textContent = emptyMessage;
  dom.resultsEmpty.hidden = rows.length > 0;
  dom.resultList.replaceChildren(
    ...rows.map((row) => {
      const item = node("li");
      const button = node("button", "result");
      button.type = "button";
      const line = node("div", "result-name");
      line.append(node("b", null, row.title));
      if (row.kind) line.append(node("span", "kind", row.kind));
      if (row.badge) line.append(node("span", "result-meta", row.badge));
      button.append(line);
      if (row.subtitle) button.append(node("div", "result-path", shortPath(row.subtitle)));
      if (row.onClick) button.addEventListener("click", row.onClick);
      if (row.disabled) button.disabled = true;
      item.append(button);
      return item;
    })
  );
}

async function openFile(path) {
  state.reload = () => openFile(path);
  const data = await api(projectPath("files"), { path });
  dom.resultsTitle.textContent = path.split("/").pop();
  const crumbs = node("div", "crumbs");
  const back = node("button", null, "← all files");
  back.type = "button";
  back.addEventListener("click", () => tabs.files());
  crumbs.append(back, node("span", null, path));
  setCrumbs(crumbs);
  renderRows(
    data.symbols.map((symbol) => ({
      title: symbol.name,
      subtitle: symbol.signature || `${symbol.start_line}–${symbol.end_line}`,
      badge: `L${symbol.start_line}`,
      kind: symbol.kind,
      onClick: () => focusSymbol(symbol.name, path, { push: true }),
    })),
    "No symbols were extracted from this file."
  );
}

/* ---------- projects dialog ---------- */

/* Kept out of the left panel on purpose: everything there answers a question
 * about the repository you are currently in, and this answers one about all
 * of them. */

const projectsDialog = document.getElementById("projects-dialog");

async function openProjects() {
  await renderKnownProjects();
  if (!projectsDialog.open) projectsDialog.showModal();
}

async function renderKnownProjects() {
  const body = document.getElementById("projects-body");
  const data = await api("/api/known");
  if (!data.projects.length) {
    body.replaceChildren(node("p", "empty", "AIDEN has not been used in any project yet."));
    return;
  }

  body.replaceChildren(
    ...data.projects.map((project) => {
      const row = node("div", "project-row");
      const who = node("div", "who");
      who.append(node("b", null, project.name), node("span", null, project.root));
      row.append(who);

      if (!project.exists) row.append(node("span", "pill missing", "folder is gone"));
      else if (project.attached) row.append(node("span", "pill live", `running · pid ${project.pid ?? "?"}`));
      else row.append(node("span", "pill cold", "not running"));

      if (project.indexed) {
        row.append(node("span", "pill", `${(project.index_size / 1048576).toFixed(0)} MB index`));
      } else if (project.exists) {
        row.append(node("span", "pill cold", "no index"));
      }

      if (project.exists) {
        const open = node("button", "ghost-button", project.attached ? "Show" : "Start here");
        open.type = "button";
        open.title = project.attached
          ? "Switch the explorer to this project"
          : "Attach this project to the explorer so it can be searched and indexed";
        open.addEventListener("click", () => startProject(project.root));
        row.append(open);
      }

      const forget = node("button", "ghost-button", "Forget");
      forget.type = "button";
      forget.title = "Remove it from this list. The index on disk is left alone.";
      forget.addEventListener("click", async () => {
        await post("/api/known/forget", { root: project.root });
        renderKnownProjects();
      });
      row.append(forget);
      return row;
    })
  );
}

async function startProject(root) {
  const response = await post("/api/projects", { root });
  const data = await response.json();
  if (!response.ok) {
    toast(data.error || "Could not open that project", "failed");
    return;
  }
  state.projectsVersion = -1;
  await refreshProjects();
  dom.projectSelect.value = data.id;
  await selectProject(data.id);
  projectsDialog.close();
  toast(`Switched to ${data.name}.`, "done");
}

async function browseForProject(path) {
  const body = document.getElementById("projects-body");
  const data = await api("/api/browse", { path });

  const header = node("div", "project-row");
  const who = node("div", "who");
  who.append(node("b", null, "Add a folder"), node("span", null, data.path));
  header.append(who);
  if (data.parent) {
    const up = node("button", "ghost-button", "Up");
    up.type = "button";
    up.addEventListener("click", () => browseForProject(data.parent));
    header.append(up);
  }
  const here = node("button", "ghost-button", "Use this folder");
  here.type = "button";
  here.addEventListener("click", () => startProject(data.path));
  header.append(here);
  const back = node("button", "ghost-button", "Cancel");
  back.type = "button";
  back.addEventListener("click", renderKnownProjects);
  header.append(back);

  const rows = data.entries.map((entry) => {
    const row = node("div", "project-row");
    const who2 = node("div", "who");
    who2.append(node("b", null, entry.name), node("span", null, entry.path));
    row.append(who2);
    if (entry.indexed) row.append(node("span", "pill", "indexed"));
    else if (entry.is_project) row.append(node("span", "pill cold", "repo"));
    const enter = node("button", "ghost-button", "Open");
    enter.type = "button";
    enter.addEventListener("click", () => browseForProject(entry.path));
    row.append(enter);
    return row;
  });

  body.replaceChildren(header, ...rows);
}

document.getElementById("project-count").addEventListener("click", openProjects);
document.getElementById("projects-close").addEventListener("click", () => projectsDialog.close());
document.getElementById("projects-browse").addEventListener("click", () => browseForProject(null));

/* ---------- shutdown ---------- */

document.getElementById("shutdown").addEventListener("click", async () => {
  const count = document.getElementById("project-count-value").textContent;
  if (!confirm(`Shut down AIDEN?\n\nThis stops the ${count} attached instance(s) and this server. Unsaved agent work is not affected, but any agent using AIDEN will lose its tools until it restarts.`)) {
    return;
  }
  try {
    const data = await (await post("/api/shutdown")).json();
    toast(`Stopped ${data.stopped.length} instance(s). This server is shutting down.`, "done");
  } catch {
    toast("The server stopped before it could answer.", "done");
  }
});

/* ---------- tab wiring ---------- */

function activateTab(name) {
  state.tab = name;
  document.querySelectorAll(".tab").forEach((tab) => tab.setAttribute("aria-selected", String(tab.dataset.tab === name)));
  tabs[name]().catch((error) => renderRows([], `Could not load: ${error.message}`));
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => activateTab(tab.dataset.tab));
});

/* ---------- tooltips ---------- */

/* One floating element for the whole page, driven by `data-tip`. Native
 * title= is unusable here: it takes a second to appear, cannot be styled, and
 * silently truncates the multi-line explanations the status bar needs. */

const TIP_DELAY_MS = 260;
let tipTimer = null;
let tipTarget = null;

function tipText(element) {
  const text = element.dataset.tip;
  return text && text.trim() ? text : null;
}

function placeTip(element) {
  const tip = dom.tip;
  const box = element.getBoundingClientRect();
  const size = tip.getBoundingClientRect();
  const margin = 8;

  let left = box.left + box.width / 2 - size.width / 2;
  left = Math.max(margin, Math.min(left, window.innerWidth - size.width - margin));

  // Prefer below; flip above when the element sits near the bottom edge
  // (the status bar, where most of these live).
  let top = box.bottom + 6;
  if (top + size.height > window.innerHeight - margin) top = box.top - size.height - 6;
  tip.style.left = `${Math.round(left)}px`;
  tip.style.top = `${Math.round(Math.max(margin, top))}px`;
}

function showTip(element) {
  const text = tipText(element);
  if (!text) return;
  tipTarget = element;
  dom.tip.replaceChildren(
    ...text.split("\n").map((line, index) => node("span", index === 0 ? null : "tip-sub", line))
  );
  dom.tip.hidden = false;
  placeTip(element);
  requestAnimationFrame(() => dom.tip.classList.add("shown"));
}

function hideTip() {
  clearTimeout(tipTimer);
  tipTarget = null;
  dom.tip.classList.remove("shown");
  dom.tip.hidden = true;
}

function scheduleTip(element) {
  clearTimeout(tipTimer);
  tipTimer = setTimeout(() => showTip(element), TIP_DELAY_MS);
}

document.addEventListener("pointerover", (event) => {
  const element = event.target.closest?.("[data-tip]");
  if (!element || element === tipTarget) return;
  hideTip();
  scheduleTip(element);
});

document.addEventListener("pointerout", (event) => {
  const element = event.target.closest?.("[data-tip]");
  if (element && element === tipTarget) hideTip();
  else if (element) clearTimeout(tipTimer);
});

document.addEventListener("focusin", (event) => {
  const element = event.target.closest?.("[data-tip]");
  if (element) showTip(element);
});

document.addEventListener("focusout", hideTip);
document.addEventListener("pointerdown", hideTip);
window.addEventListener("scroll", hideTip, true);
window.addEventListener("blur", hideTip);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") hideTip();
});
