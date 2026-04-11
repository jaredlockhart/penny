/**
 * Full page — consolidates thoughts, prompt logs, and all settings.
 * Tabs: Thoughts, Prompts, Schedules, Likes, Dislikes, Domains, Config.
 */

import {
  type DomainAllowlist,
  DomainPermission as DP,
  type DomainPermissionEntry,
  type PreferenceItem,
  type PromptLogEntry,
  type PromptLogRun,
  type RuntimeConfigParam,
  type RuntimeMessage,
  RuntimeMessageType,
  type ScheduleItem,
  STORAGE_KEY_DOMAIN_ALLOWLIST,
  STORAGE_KEY_TOOL_USE,
  type ThoughtCard,
} from "../protocol.js";

// --- Top-level state ---

type Tab = "thoughts" | "prompts" | "schedules" | "likes" | "dislikes" | "domains" | "config";

// --- Toast ---

let toastTimer: ReturnType<typeof setTimeout> | null = null;

function showToast(text: string): void {
  const toast = document.getElementById("toast")!;
  toast.textContent = text;
  toast.classList.add("visible");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 2000);
}

// --- Thoughts state ---

const grid = document.getElementById("grid")!;
const thoughtsLoading = document.getElementById("loading")!;
const thoughtsLoadMore = document.getElementById("thoughts-load-more")!;
const thoughtsLoadMoreBtn = document.getElementById("thoughts-load-more-btn")!;
const modal = document.getElementById("modal")!;
const modalImage = document.getElementById("modal-image") as HTMLImageElement;
const modalTitle = document.getElementById("modal-title")!;
const modalDate = document.getElementById("modal-date")!;
const modalText = document.getElementById("modal-text")!;

let unnotifiedThoughts: ThoughtCard[] = [];
let notifiedThoughts: ThoughtCard[] = [];
let notifiedHasMore = false;
let notifiedPages = 1;
let activeThoughtTab: "new" | "archive" = "new";
let modalThought: ThoughtCard | null = null;

// --- Prompts state ---

const runsContainer = document.getElementById("runs")!;
const promptsLoading = document.getElementById("prompts-loading")!;
const promptsLoadMore = document.getElementById("prompts-load-more")!;
const promptsLoadMoreBtn = document.getElementById("prompts-load-more-btn")!;
let activeAgentFilter = "";

const AGENT_LABELS: Record<string, string> = {
  inner_monologue: '<i class="fa-regular fa-lightbulb"></i> Thinking',
  chat: '<i class="fa-solid fa-comment"></i> Chat',
  history: '<i class="fa-solid fa-clock-rotate-left"></i> History',
  notify: '<i class="fa-solid fa-bell"></i> Notify',
  startup: '<i class="fa-solid fa-rocket"></i> Startup',
};

const ACTIVE_TIMEOUT_MS = 60_000;

let allRuns: PromptLogRun[] = [];
let hasMore = false;
const runElements = new Map<string, HTMLElement>();
let activeRunId: string | null = null;
let activeTimer: ReturnType<typeof setTimeout> | null = null;
let promptsLoaded = false;

// --- Config state ---

let pendingConfigSave = false;

// ============================================================
// Init
// ============================================================

function init(): void {
  browser.runtime.onMessage.addListener(handleMessage);

  // Top-level tab switching
  for (const btn of Array.from(document.querySelectorAll(".tab"))) {
    btn.addEventListener("click", () => switchTab(btn.getAttribute("data-tab") as Tab));
  }

  // Load initial data for thoughts tab
  browser.runtime.sendMessage({ type: RuntimeMessageType.ThoughtsRequest });

  // Set up all panel interactions
  setupThoughts();
  setupPrompts();
  setupSchedules();
  setupPreferences("positive", "likes");
  setupPreferences("negative", "dislikes");
  setupDomains();
  setupConfig();
}

function switchTab(tab: Tab): void {
  for (const btn of Array.from(document.querySelectorAll(".tab"))) {
    btn.classList.toggle("active", btn.getAttribute("data-tab") === tab);
  }
  for (const panel of Array.from(document.querySelectorAll(".panel"))) {
    panel.classList.toggle("hidden", panel.id !== `panel-${tab}`);
  }

  // Request data for the activated tab
  if (tab === "thoughts") {
    browser.runtime.sendMessage({ type: RuntimeMessageType.ThoughtsRequest });
  } else if (tab === "prompts" && !promptsLoaded) {
    requestPromptLogs(0);
  } else if (tab === "schedules") {
    browser.runtime.sendMessage({ type: RuntimeMessageType.SchedulesRequest });
  } else if (tab === "likes") {
    browser.runtime.sendMessage({ type: RuntimeMessageType.PreferencesRequest, valence: "positive" });
  } else if (tab === "dislikes") {
    browser.runtime.sendMessage({ type: RuntimeMessageType.PreferencesRequest, valence: "negative" });
  } else if (tab === "domains") {
    loadDomainsFromCache();
  } else if (tab === "config") {
    browser.runtime.sendMessage({ type: RuntimeMessageType.ConfigRequest });
    loadToolUseState();
  }
}

// ============================================================
// Message handler
// ============================================================

function handleMessage(message: RuntimeMessage): void {
  if (message.type === RuntimeMessageType.ThoughtsResponse) {
    unnotifiedThoughts = message.unnotified;
    notifiedThoughts = message.notified;
    notifiedHasMore = message.notified_has_more;
    renderThoughts();
  } else if (message.type === RuntimeMessageType.PromptLogsResponse) {
    promptsLoaded = true;
    if (message.runs.length > 0 && allRuns.length > 0) {
      appendRuns(message.runs);
    } else {
      allRuns = message.runs;
      renderPrompts();
    }
    hasMore = message.has_more;
    promptsLoadMore.classList.toggle("hidden", !hasMore);
  } else if (message.type === RuntimeMessageType.PromptLogUpdate) {
    handlePromptUpdate(message.prompt);
  } else if (message.type === RuntimeMessageType.RunOutcomeUpdate) {
    handleRunOutcome(message.run_id, message.outcome);
  } else if (message.type === RuntimeMessageType.SchedulesResponse) {
    renderSchedules(message.schedules, message.error);
  } else if (message.type === RuntimeMessageType.PreferencesResponse) {
    renderPreferences(message.valence, message.preferences);
  } else if (message.type === RuntimeMessageType.ConfigResponse) {
    renderConfig(message.params);
    if (pendingConfigSave) {
      pendingConfigSave = false;
      showToast("Saved");
    }
  } else if (message.type === RuntimeMessageType.ToolUseState) {
    const toggle = document.getElementById("tool-use-toggle") as HTMLInputElement | null;
    if (toggle) toggle.checked = message.enabled;
  } else if (message.type === RuntimeMessageType.DomainPermissionsSync) {
    renderDomains(message.permissions);
  }
}

// ============================================================
// Thoughts
// ============================================================

function setupThoughts(): void {
  for (const btn of Array.from(document.querySelectorAll(".thought-tab"))) {
    btn.addEventListener("click", () => switchThoughtTab(btn.getAttribute("data-ttab") as "new" | "archive"));
  }

  thoughtsLoadMoreBtn.addEventListener("click", loadNextPage);

  document.getElementById("modal-backdrop")!.addEventListener("click", closeModal);
  document.getElementById("modal-close")!.addEventListener("click", closeModal);
  document.getElementById("modal-thumbs-up")!.addEventListener("click", () => modalReact("\u{1f44d}"));
  document.getElementById("modal-thumbs-down")!.addEventListener("click", () => modalReact("\u{1f44e}"));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });
}

function switchThoughtTab(tab: "new" | "archive"): void {
  activeThoughtTab = tab;

  for (const btn of Array.from(document.querySelectorAll(".thought-tab"))) {
    btn.classList.toggle("active", btn.getAttribute("data-ttab") === tab);
  }

  renderThoughts();
}

function renderThoughts(): void {
  thoughtsLoading.classList.add("hidden");
  grid.innerHTML = "";

  const visible = activeThoughtTab === "new" ? unnotifiedThoughts : notifiedThoughts;

  if (visible.length === 0) {
    thoughtsLoading.textContent = activeThoughtTab === "new"
      ? "No new thoughts yet. Penny is still thinking..."
      : "No archived thoughts yet.";
    thoughtsLoading.classList.remove("hidden");
    thoughtsLoadMore.classList.add("hidden");
    return;
  }

  renderCards(visible);
  const showLoadMore = activeThoughtTab === "archive" && notifiedHasMore;
  thoughtsLoadMore.classList.toggle("hidden", !showLoadMore);
}

function loadNextPage(): void {
  notifiedPages += 1;
  browser.runtime.sendMessage({
    type: RuntimeMessageType.ThoughtsRequest,
    notified_pages: notifiedPages,
  });
}

function renderCards(thoughts: ThoughtCard[]): void {
  for (const thought of thoughts) {
    grid.appendChild(createCard(thought));
  }
}

function createCard(thought: ThoughtCard): HTMLElement {
  const card = document.createElement("div");
  card.className = "card";
  card.dataset.id = String(thought.id);
  card.style.cursor = "pointer";
  card.addEventListener("click", () => openModal(thought));

  if (thought.image) {
    const img = document.createElement("img");
    img.className = "card-image";
    img.src = thought.image;
    img.alt = thought.title;
    img.loading = "lazy";
    img.addEventListener("error", () => {
      img.replaceWith(createPlaceholder());
    });
    card.appendChild(img);
  } else {
    card.appendChild(createPlaceholder());
  }

  const body = document.createElement("div");
  body.className = "card-body";

  const title = document.createElement("div");
  title.className = "card-title";
  title.textContent = thought.title || "Untitled thought";
  body.appendChild(title);

  const byline = buildByline(thought);
  if (byline) {
    const date = document.createElement("div");
    date.className = "card-date";
    date.textContent = byline;
    body.appendChild(date);
  }

  const content = document.createElement("div");
  content.className = "card-content";
  content.innerHTML = thought.content;
  body.appendChild(content);

  card.appendChild(body);

  if (!thought.notified) {
    const actions = document.createElement("div");
    actions.className = "card-actions";

    const thumbsUp = document.createElement("button");
    thumbsUp.className = "reaction-btn thumbs-up";
    thumbsUp.innerHTML = '<i class="fa-solid fa-thumbs-up"></i>';
    thumbsUp.addEventListener("click", (e) => {
      e.stopPropagation();
      reactToThought(thought.id, "\u{1f44d}", card);
    });

    const thumbsDown = document.createElement("button");
    thumbsDown.className = "reaction-btn thumbs-down";
    thumbsDown.innerHTML = '<i class="fa-solid fa-thumbs-down"></i>';
    thumbsDown.addEventListener("click", (e) => {
      e.stopPropagation();
      reactToThought(thought.id, "\u{1f44e}", card);
    });

    actions.appendChild(thumbsUp);
    actions.appendChild(thumbsDown);
    card.appendChild(actions);
  }

  return card;
}

function reactToThought(thoughtId: number, emoji: string, card: HTMLElement | null): void {
  browser.runtime.sendMessage({
    type: RuntimeMessageType.ThoughtReaction,
    thought_id: thoughtId,
    emoji,
  });
  if (card) {
    card.style.transition = "opacity 0.3s, transform 0.3s";
    card.style.opacity = "0";
    card.style.transform = "scale(0.95)";
    setTimeout(() => card.remove(), 300);
  }
}

function createPlaceholder(): HTMLElement {
  const div = document.createElement("div");
  div.className = "card-image-placeholder";
  div.textContent = "\u{1f4ad}";
  return div;
}

function buildByline(thought: ThoughtCard): string {
  const parts: string[] = [];
  if (thought.created_at) parts.push(formatDate(thought.created_at));
  parts.push(thought.seed_topic || "free thought");
  return parts.join(" \u00b7 ");
}

function openModal(thought: ThoughtCard): void {
  modalThought = thought;
  modalImage.src = thought.image || "";
  modalTitle.textContent = thought.title || "Untitled thought";
  modalDate.textContent = buildByline(thought);
  modalText.innerHTML = thought.content;
  const actions = document.getElementById("modal-actions")!;
  actions.classList.toggle("hidden", thought.notified);
  modal.classList.remove("hidden");
}

function closeModal(): void {
  modal.classList.add("hidden");
  modalThought = null;
}

function modalReact(emoji: string): void {
  if (!modalThought) return;
  const card = document.querySelector(`.card[data-id="${modalThought.id}"]`) as HTMLElement | null;
  reactToThought(modalThought.id, emoji, card);
  closeModal();
}

function formatDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return "";
  }
}

// ============================================================
// Prompts
// ============================================================

function setupPrompts(): void {
  for (const btn of Array.from(document.querySelectorAll("#agent-tabs .sub-tab"))) {
    btn.addEventListener("click", () => {
      activeAgentFilter = btn.getAttribute("data-agent") ?? "";
      for (const b of Array.from(document.querySelectorAll("#agent-tabs .sub-tab"))) {
        b.classList.toggle("active", b === btn);
      }
      allRuns = [];
      requestPromptLogs(0);
    });
  }
  promptsLoadMoreBtn.addEventListener("click", () => {
    requestPromptLogs(allRuns.length);
  });
}

function requestPromptLogs(offset: number): void {
  const agentName = activeAgentFilter || undefined;
  browser.runtime.sendMessage({
    type: RuntimeMessageType.PromptLogsRequest,
    agent_name: agentName,
    offset,
  });
}

function handlePromptUpdate(prompt: PromptLogEntry & { run_id: string }): void {
  if (activeAgentFilter && prompt.agent_name !== activeAgentFilter) return;

  const existingRun = allRuns.find((r) => r.run_id === prompt.run_id);
  if (existingRun) {
    updateExistingRun(existingRun, prompt);
  } else {
    insertNewRun(prompt);
  }
}

function updateExistingRun(run: PromptLogRun, prompt: PromptLogEntry): void {
  run.prompts.push(prompt);
  run.prompt_count = run.prompts.length;
  run.ended_at = prompt.timestamp;
  run.total_duration_ms += prompt.duration_ms;
  run.total_input_tokens += prompt.input_tokens;
  run.total_output_tokens += prompt.output_tokens;

  const row = runElements.get(run.run_id);
  if (!row) return;

  const summary = row.querySelector(".run-summary")!;
  const oldHeader = summary.querySelector(".run-header")!;
  const newHeader = createRunHeader(run);
  summary.replaceChild(newHeader, oldHeader);

  const promptsContainer = row.querySelector(".run-prompts")!;
  promptsContainer.appendChild(createPromptRow(prompt, run.prompts.length));

  markRunActive(run.run_id, row);
}

function insertNewRun(prompt: PromptLogEntry & { run_id: string }): void {
  const run: PromptLogRun = {
    run_id: prompt.run_id,
    agent_name: prompt.agent_name,
    prompt_count: 1,
    started_at: prompt.timestamp,
    ended_at: prompt.timestamp,
    total_duration_ms: prompt.duration_ms,
    total_input_tokens: prompt.input_tokens,
    total_output_tokens: prompt.output_tokens,
    run_outcome: null,
    prompts: [prompt],
  };
  allRuns.unshift(run);
  promptsLoading.classList.add("hidden");

  const row = createRunRow(run);
  runsContainer.prepend(row);
  markRunActive(run.run_id, row);
}

function handleRunOutcome(runId: string, outcome: string): void {
  const run = allRuns.find((r) => r.run_id === runId);
  if (!run) return;
  run.run_outcome = outcome;

  const row = runElements.get(runId);
  if (!row) return;

  const summary = row.querySelector(".run-summary");
  if (summary) {
    summary.appendChild(createRunOutcome(outcome));
  }

  // Dismiss spinner — run is complete
  row.classList.remove("active-run");
  if (activeRunId === runId) {
    if (activeTimer) clearTimeout(activeTimer);
    activeRunId = null;
    activeTimer = null;
  }
}

function markRunActive(runId: string, row: HTMLElement): void {
  if (activeRunId && activeRunId !== runId) {
    const previous = runElements.get(activeRunId);
    if (previous) previous.classList.remove("active-run");
  }
  activeRunId = runId;
  row.classList.add("active-run");
  if (activeTimer) clearTimeout(activeTimer);
  activeTimer = setTimeout(() => {
    row.classList.remove("active-run");
    activeRunId = null;
    activeTimer = null;
  }, ACTIVE_TIMEOUT_MS);
}

function renderPrompts(): void {
  promptsLoading.classList.add("hidden");
  runsContainer.innerHTML = "";
  runElements.clear();
  if (activeTimer) clearTimeout(activeTimer);
  activeTimer = null;
  activeRunId = null;

  if (allRuns.length === 0) {
    const label = activeAgentFilter || "any agent";
    promptsLoading.textContent = `No prompt logs for ${label}.`;
    promptsLoading.classList.remove("hidden");
    return;
  }

  for (const run of allRuns) {
    runsContainer.appendChild(createRunRow(run));
  }
}

function appendRuns(newRuns: PromptLogRun[]): void {
  for (const run of newRuns) {
    allRuns.push(run);
    runsContainer.appendChild(createRunRow(run));
  }
}

function createRunRow(run: PromptLogRun): HTMLElement {
  const row = document.createElement("div");
  row.className = "run";
  runElements.set(run.run_id, row);

  const summary = document.createElement("div");
  summary.className = "run-summary";

  const header = createRunHeader(run);
  summary.appendChild(header);

  if (run.run_outcome) {
    summary.appendChild(createRunOutcome(run.run_outcome));
  }

  row.appendChild(summary);

  const promptsContainer = document.createElement("div");
  promptsContainer.className = "run-prompts";
  for (let i = 0; i < run.prompts.length; i++) {
    promptsContainer.appendChild(createPromptRow(run.prompts[i], i + 1));
  }
  row.appendChild(promptsContainer);

  summary.addEventListener("click", () => {
    row.classList.toggle("expanded");
  });

  return row;
}

function createRunOutcome(outcome: string): HTMLElement {
  const el = document.createElement("div");
  el.className = outcome.startsWith("Stored")
    ? "run-outcome run-outcome-stored"
    : "run-outcome run-outcome-discarded";
  el.textContent = outcome;
  return el;
}

function createRunHeader(run: PromptLogRun): HTMLElement {
  const header = document.createElement("div");
  header.className = "run-header";

  const toggle = document.createElement("span");
  toggle.className = "run-toggle";
  toggle.innerHTML = '<i class="fa-solid fa-chevron-right"></i>';
  header.appendChild(toggle);

  const agent = document.createElement("span");
  agent.className = "run-agent";
  agent.innerHTML = AGENT_LABELS[run.agent_name] ?? run.agent_name;
  const spinner = document.createElement("span");
  spinner.className = "run-spinner";
  spinner.innerHTML = ' <i class="fa-solid fa-spinner fa-spin"></i>';
  agent.appendChild(spinner);
  header.appendChild(agent);

  const promptType = extractPromptType(run);
  if (promptType) {
    const typeEl = document.createElement("span");
    typeEl.className = "run-type";
    typeEl.textContent = promptType;
    header.appendChild(typeEl);
  }

  const time = document.createElement("span");
  time.className = "run-time";
  time.textContent = formatDateTime(run.started_at);
  header.appendChild(time);

  const meta = document.createElement("span");
  meta.className = "run-meta";
  const tokPerSec = run.total_duration_ms > 0
    ? ((run.total_output_tokens / run.total_duration_ms) * 1000).toFixed(1)
    : "0";
  meta.innerHTML = `<span><i class="fa-solid fa-layer-group"></i>${run.prompt_count}</span>` +
    `<span><i class="fa-solid fa-arrow-down"></i>${formatTokens(run.total_input_tokens)}</span>` +
    `<span><i class="fa-solid fa-arrow-up"></i>${formatTokens(run.total_output_tokens)}</span>` +
    `<span><i class="fa-solid fa-gauge-high"></i>${tokPerSec} tok/s</span>` +
    `<span><i class="fa-solid fa-clock"></i>${formatDuration(run.total_duration_ms)}</span>`;
  header.appendChild(meta);

  return header;
}

function createPromptRow(prompt: PromptLogEntry, step: number): HTMLElement {
  const row = document.createElement("div");
  row.className = "prompt";

  const header = document.createElement("div");
  header.className = "prompt-header";

  const stepEl = document.createElement("span");
  stepEl.className = "prompt-step";
  stepEl.textContent = String(step);
  header.appendChild(stepEl);

  const iconEl = document.createElement("span");
  iconEl.className = "prompt-tools";
  iconEl.innerHTML = prompt.has_tools
    ? '<i class="fa-solid fa-wrench"></i>'
    : '<i class="fa-solid fa-comment"></i>';
  header.appendChild(iconEl);

  const snippet = extractLastTurnSnippet(prompt);
  if (snippet) {
    const snippetEl = document.createElement("span");
    snippetEl.className = "prompt-snippet";
    snippetEl.textContent = snippet;
    snippetEl.title = snippet;
    header.appendChild(snippetEl);
  }

  const meta = document.createElement("span");
  meta.className = "prompt-meta";
  const promptTokPerSec = prompt.duration_ms > 0
    ? ((prompt.output_tokens / prompt.duration_ms) * 1000).toFixed(1)
    : "0";
  meta.innerHTML =
    `<span></span>` +
    `<span><i class="fa-solid fa-arrow-down"></i>${formatTokens(prompt.input_tokens)}</span>` +
    `<span><i class="fa-solid fa-arrow-up"></i>${formatTokens(prompt.output_tokens)}</span>` +
    `<span><i class="fa-solid fa-gauge-high"></i>${promptTokPerSec} tok/s</span>` +
    `<span><i class="fa-solid fa-clock"></i>${formatDuration(prompt.duration_ms)}</span>`;
  header.appendChild(meta);

  row.appendChild(header);

  const detail = createPromptDetail(prompt);
  row.appendChild(detail);

  header.addEventListener("click", () => {
    row.classList.toggle("expanded");
  });

  return row;
}

function createPromptDetail(prompt: PromptLogEntry): HTMLElement {
  const detail = document.createElement("div");
  detail.className = "prompt-detail";

  for (const message of prompt.messages) {
    const role = String(message.role ?? "unknown");
    const content = extractMessageContent(message);
    detail.appendChild(createPromptSection(role, content));
  }

  if (prompt.thinking) {
    detail.appendChild(createPromptSection("thinking", prompt.thinking));
  }

  detail.appendChild(createPromptSection("response", renderResponse(prompt.response)));

  return detail;
}

function createPromptSection(label: string, content: string): HTMLElement {
  const section = document.createElement("div");
  section.className = "prompt-section";

  const labelEl = document.createElement("div");
  labelEl.className = "prompt-section-label";
  labelEl.dataset.role = label.toLowerCase();
  labelEl.innerHTML = `<i class="fa-solid fa-chevron-right section-toggle-icon"></i> ${label}`;
  section.appendChild(labelEl);

  const contentEl = document.createElement("div");
  contentEl.className = "prompt-section-content";
  contentEl.textContent = content;
  section.appendChild(contentEl);

  labelEl.addEventListener("click", () => {
    section.classList.toggle("expanded");
  });

  return section;
}

function extractMessageContent(message: Record<string, unknown>): string {
  const parts: string[] = [];

  if (typeof message.content === "string" && message.content) {
    parts.push(prettyJson(message.content));
  } else if (Array.isArray(message.content)) {
    const text = message.content.map((part: Record<string, unknown>) => {
      if (part.type === "text") return String(part.text ?? "");
      if (part.type === "image_url") return "[image]";
      return JSON.stringify(part);
    }).join("\n");
    if (text) parts.push(text);
  }

  if (Array.isArray(message.tool_calls)) {
    const calls = message.tool_calls as Record<string, unknown>[];
    for (const call of calls) {
      const fn = call.function as Record<string, unknown> | undefined;
      if (fn) {
        parts.push(`tool_call: ${fn.name}(${prettyJson(String(fn.arguments ?? ""))})`);
      } else {
        parts.push(JSON.stringify(call, null, 2));
      }
    }
  }

  return parts.length > 0 ? parts.join("\n") : prettyJson(JSON.stringify(message.content ?? ""));
}

function renderResponse(response: Record<string, unknown>): string {
  const choices = response.choices as Record<string, unknown>[] | undefined;
  if (!choices || choices.length === 0) {
    return JSON.stringify(response, null, 2);
  }

  const choice = choices[0];
  const message = choice.message as Record<string, unknown> | undefined;
  if (!message) {
    return JSON.stringify(choice, null, 2);
  }

  return extractMessageContent(message);
}


const SNIPPET_MAX_CHARS = 80;

function extractLastTurnSnippet(prompt: PromptLogEntry): string {
  const response = prompt.response as Record<string, unknown>;
  const choices = response.choices as Record<string, unknown>[] | undefined;
  if (!choices || choices.length === 0) return "";
  const message = choices[0].message as Record<string, unknown> | undefined;
  if (!message) return "";

  // Check for tool calls first
  const toolCalls = message.tool_calls as Record<string, unknown>[] | undefined;
  if (toolCalls && toolCalls.length > 0) {
    const names = toolCalls.map((tc) => {
      const fn = tc.function as Record<string, unknown> | undefined;
      return fn?.name ?? "tool";
    });
    const args = toolCalls.map((tc) => {
      const fn = tc.function as Record<string, unknown> | undefined;
      const raw = fn?.arguments;
      if (typeof raw === "string") {
        try {
          const parsed = JSON.parse(raw);
          return parsed.queries ? parsed.queries.join(", ") : raw;
        } catch { return raw; }
      }
      if (typeof raw === "object" && raw !== null) {
        const obj = raw as Record<string, unknown>;
        return obj.queries ? (obj.queries as string[]).join(", ") : JSON.stringify(raw);
      }
      return "";
    });
    return normalizeSnippet(names.map((n, i) => `${n}(${args[i]})`).join(", "));
  }

  return normalizeSnippet(message.content as string | null);
}

function normalizeSnippet(content: string | null | undefined): string {
  if (typeof content !== "string" || content.length === 0) return "";
  const text = content.replace(/\s+/g, " ").trim();
  if (text.length <= SNIPPET_MAX_CHARS) return text;
  return text.slice(0, SNIPPET_MAX_CHARS) + "…";
}

function extractPromptType(run: PromptLogRun): string {
  for (const prompt of run.prompts) {
    if (!prompt.prompt_type) continue;
    if (prompt.prompt_type === "user_message") {
      const userText = extractLastUserMessage(prompt);
      if (userText) return userText;
    }
    return prompt.prompt_type;
  }
  return "";
}

function extractLastUserMessage(prompt: PromptLogEntry): string {
  for (let i = prompt.messages.length - 1; i >= 0; i--) {
    const message = prompt.messages[i];
    if (message.role !== "user") continue;
    const snippet = normalizeSnippet(message.content as string | null);
    if (snippet) return snippet;
  }
  return "";
}

function formatDateTime(iso: string): string {
  try {
    const date = new Date(iso);
    return date.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function prettyJson(value: string): string {
  try {
    return JSON.stringify(JSON.parse(value), null, 2);
  } catch {
    return value;
  }
}

function formatTokens(count: number): string {
  if (count >= 1000) return `${(count / 1000).toFixed(1)}k`;
  return String(count);
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const seconds = (ms / 1000).toFixed(1);
  return `${seconds}s`;
}

// ============================================================
// Preferences
// ============================================================

function renderPreferences(valence: string, prefs: PreferenceItem[]): void {
  const listEl = document.getElementById(valence === "positive" ? "likes-list" : "dislikes-list")!;
  listEl.innerHTML = "";

  if (prefs.length === 0) {
    const empty = document.createElement("div");
    empty.className = "prefs-empty";
    empty.textContent = valence === "positive" ? "No likes yet." : "No dislikes yet.";
    listEl.appendChild(empty);
    return;
  }

  for (const pref of prefs) {
    const row = document.createElement("div");
    row.className = "pref-row";

    const name = document.createElement("span");
    name.className = "pref-name";
    name.textContent = pref.content;

    const meta = document.createElement("span");
    meta.className = "pref-meta";
    const sourceLabel = pref.source === "manual" ? "manual" : "extracted";
    const sourceIcon = pref.source === "manual" ? "fa-hand" : "fa-robot";
    meta.innerHTML =
      `<span><i class="fa-solid ${sourceIcon}"></i>${sourceLabel}</span>` +
      `<span><i class="fa-solid fa-comment"></i>${pref.mention_count} mentions</span>`;

    const del = document.createElement("button");
    del.className = "pref-delete";
    del.innerHTML = '<i class="fa-solid fa-xmark"></i>';
    del.setAttribute("aria-label", `Remove ${pref.content}`);
    del.addEventListener("click", () => {
      browser.runtime.sendMessage({ type: RuntimeMessageType.PreferenceDelete, preference_id: pref.id });
    });

    row.appendChild(name);
    row.appendChild(meta);
    row.appendChild(del);
    listEl.appendChild(row);
  }
}

function setupPreferences(valence: string, prefix: string): void {
  const input = document.getElementById(`${prefix}-input`) as HTMLInputElement;
  const btn = document.getElementById(`${prefix}-add-btn`) as HTMLButtonElement;

  function add(): void {
    const content = input.value.trim();
    if (!content) return;
    browser.runtime.sendMessage({ type: RuntimeMessageType.PreferenceAdd, valence, content });
    input.value = "";
    showToast(`Added ${valence === "positive" ? "like" : "dislike"}: ${content}`);
  }

  btn.addEventListener("click", add);
  input.addEventListener("keydown", (e: KeyboardEvent) => {
    if (e.key === "Enter") add();
  });
}

// ============================================================
// Schedules
// ============================================================

function setScheduleAddEnabled(enabled: boolean): void {
  const input = document.getElementById("schedules-input") as HTMLInputElement | null;
  const btn = document.getElementById("schedules-add-btn") as HTMLButtonElement | null;
  if (input) input.disabled = !enabled;
  if (btn) btn.disabled = !enabled;
}

function renderSchedules(schedules: ScheduleItem[], error: string | null): void {
  const listEl = document.getElementById("schedules-list")!;
  listEl.innerHTML = "";
  setScheduleAddEnabled(true);

  if (error) {
    const errEl = document.createElement("div");
    errEl.className = "schedule-error";
    errEl.textContent = error;
    listEl.appendChild(errEl);
  }

  if (schedules.length === 0 && !error) {
    const empty = document.createElement("div");
    empty.className = "schedules-empty";
    empty.textContent = "No scheduled tasks yet.";
    listEl.appendChild(empty);
    return;
  }

  for (const schedule of schedules) {
    listEl.appendChild(createScheduleRow(schedule));
  }
}

function createScheduleRow(schedule: ScheduleItem): HTMLElement {
  const row = document.createElement("div");
  row.className = "schedule-row";

  const header = document.createElement("div");
  header.className = "schedule-header";

  const timing = document.createElement("span");
  timing.className = "schedule-timing";
  timing.textContent = schedule.timing_description;

  const prompt = document.createElement("span");
  prompt.className = "schedule-prompt";
  prompt.textContent = schedule.prompt_text;

  const cron = document.createElement("span");
  cron.className = "schedule-cron-inline";
  cron.textContent = schedule.cron_expression;

  const del = document.createElement("button");
  del.className = "schedule-delete";
  del.innerHTML = '<i class="fa-solid fa-xmark"></i>';
  del.setAttribute("aria-label", `Delete schedule: ${schedule.prompt_text}`);
  del.addEventListener("click", (e) => {
    e.stopPropagation();
    browser.runtime.sendMessage({
      type: RuntimeMessageType.ScheduleDelete,
      schedule_id: schedule.id,
    });
  });

  header.appendChild(timing);
  header.appendChild(prompt);
  header.appendChild(cron);
  header.appendChild(del);

  const detail = document.createElement("div");
  detail.className = "schedule-detail";

  const editInput = document.createElement("textarea");
  editInput.className = "schedule-edit-input";
  editInput.value = schedule.prompt_text;
  editInput.rows = 2;

  const saveBtn = document.createElement("button");
  saveBtn.className = "schedule-save";
  saveBtn.textContent = "Save";
  saveBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const newText = editInput.value.trim();
    if (newText && newText !== schedule.prompt_text) {
      browser.runtime.sendMessage({
        type: RuntimeMessageType.ScheduleUpdate,
        schedule_id: schedule.id,
        prompt_text: newText,
      });
    }
  });

  detail.appendChild(editInput);
  detail.appendChild(saveBtn);

  row.appendChild(header);
  row.appendChild(detail);

  header.addEventListener("click", () => {
    row.classList.toggle("expanded");
  });

  return row;
}

function createSkeletonRow(): HTMLElement {
  const row = document.createElement("div");
  row.className = "schedule-row schedule-skeleton";

  const header = document.createElement("div");
  header.className = "schedule-header";

  const timing = document.createElement("span");
  timing.className = "skeleton-block skeleton-timing";

  const prompt = document.createElement("span");
  prompt.className = "skeleton-block skeleton-prompt";

  header.appendChild(timing);
  header.appendChild(prompt);
  row.appendChild(header);
  return row;
}

function setupSchedules(): void {
  const input = document.getElementById("schedules-input") as HTMLInputElement;
  const btn = document.getElementById("schedules-add-btn")!;

  function add(): void {
    const command = input.value.trim();
    if (!command) return;
    browser.runtime.sendMessage({ type: RuntimeMessageType.ScheduleAdd, command });
    input.value = "";
    setScheduleAddEnabled(false);
    showToast(`Adding schedule: ${command}`);

    const listEl = document.getElementById("schedules-list")!;
    const empty = listEl.querySelector(".schedules-empty");
    if (empty) empty.remove();
    listEl.appendChild(createSkeletonRow());
  }

  btn.addEventListener("click", add);
  input.addEventListener("keydown", (e: KeyboardEvent) => {
    if (e.key === "Enter") add();
  });
}

// ============================================================
// Domains
// ============================================================

async function loadDomainsFromCache(): Promise<void> {
  const stored = await browser.storage.local.get(STORAGE_KEY_DOMAIN_ALLOWLIST);
  const allowlist: DomainAllowlist = (stored[STORAGE_KEY_DOMAIN_ALLOWLIST] as DomainAllowlist) ?? {};
  const permissions = Object.entries(allowlist).map(([domain, permission]) => ({ domain, permission }));
  renderDomains(permissions);
}

function renderDomains(permissions: DomainPermissionEntry[]): void {
  const listEl = document.getElementById("domains-list")!;
  listEl.innerHTML = "";

  const sorted = [...permissions].sort((a, b) => a.domain.localeCompare(b.domain));

  if (sorted.length === 0) {
    const empty = document.createElement("div");
    empty.className = "prefs-empty";
    empty.textContent = "No domains saved yet.";
    listEl.appendChild(empty);
    return;
  }

  for (const { domain, permission } of sorted) {
    const row = document.createElement("div");
    row.className = "domain-row";

    const name = document.createElement("span");
    name.className = "domain-name";
    name.textContent = domain;

    const status = document.createElement("button");
    status.className = `domain-status ${permission}`;
    status.textContent = permission === DP.Allowed ? "Allowed" : "Blocked";
    status.title = "Click to toggle";
    status.addEventListener("click", () => {
      const next = permission === DP.Allowed ? DP.Blocked : DP.Allowed;
      browser.runtime.sendMessage({ type: RuntimeMessageType.DomainUpdate, domain, permission: next });
    });

    const del = document.createElement("button");
    del.className = "pref-delete";
    del.innerHTML = '<i class="fa-solid fa-xmark"></i>';
    del.setAttribute("aria-label", `Remove ${domain}`);
    del.addEventListener("click", () => {
      browser.runtime.sendMessage({ type: RuntimeMessageType.DomainDelete, domain });
    });

    row.appendChild(name);
    row.appendChild(status);
    row.appendChild(del);
    listEl.appendChild(row);
  }
}

function setupDomains(): void {
  const input = document.getElementById("domains-input") as HTMLInputElement;
  const select = document.getElementById("domains-permission") as HTMLSelectElement;
  const btn = document.getElementById("domains-add-btn")!;

  function add(): void {
    const raw = input.value.trim().toLowerCase();
    if (!raw) return;
    const domain = raw.replace(/^https?:\/\//, "").replace(/\/.*$/, "");
    if (!domain) return;
    browser.runtime.sendMessage({
      type: RuntimeMessageType.DomainUpdate,
      domain,
      permission: select.value,
    });
    input.value = "";
    const label = select.value === "allowed" ? "Allowed" : "Blocked";
    showToast(`${label}: ${domain}`);
  }

  btn.addEventListener("click", add);
  input.addEventListener("keydown", (e: KeyboardEvent) => {
    if (e.key === "Enter") add();
  });
}

// ============================================================
// Config
// ============================================================

async function loadToolUseState(): Promise<void> {
  const stored = await browser.storage.local.get(STORAGE_KEY_TOOL_USE);
  const enabled = (stored[STORAGE_KEY_TOOL_USE] as boolean) ?? false;
  const toggle = document.getElementById("tool-use-toggle") as HTMLInputElement | null;
  if (toggle) toggle.checked = enabled;
}

function setupConfig(): void {
  const toggle = document.getElementById("tool-use-toggle") as HTMLInputElement;
  toggle.addEventListener("change", () => {
    browser.runtime.sendMessage({ type: RuntimeMessageType.ToolUseToggle, enabled: toggle.checked });
  });
}

function renderConfig(params: RuntimeConfigParam[]): void {
  const panel = document.getElementById("config-list")!;
  panel.innerHTML = "";

  const groups = new Map<string, RuntimeConfigParam[]>();
  for (const param of params) {
    if (!groups.has(param.group)) groups.set(param.group, []);
    groups.get(param.group)!.push(param);
  }

  for (const [group, groupParams] of groups) {
    const groupEl = document.createElement("div");
    groupEl.className = "config-group";

    const title = document.createElement("div");
    title.className = "config-group-title";
    title.textContent = group;
    groupEl.appendChild(title);

    for (const param of groupParams) {
      groupEl.appendChild(createConfigItem(param));
    }
    panel.appendChild(groupEl);
  }
}

function createConfigItem(param: RuntimeConfigParam): HTMLElement {
  const item = document.createElement("div");
  item.className = "config-item";

  const header = document.createElement("div");
  header.className = "config-header";

  const label = document.createElement("label");
  label.className = "config-label";
  label.textContent = param.description;
  label.htmlFor = `config-${param.key}`;

  const key = document.createElement("span");
  key.className = "config-key";
  key.textContent = param.key;

  const defaultVal = document.createElement("span");
  defaultVal.className = "config-default";
  defaultVal.textContent = `default: ${param.default}`;

  header.appendChild(label);
  header.appendChild(key);
  header.appendChild(defaultVal);

  const input = document.createElement("input");
  input.id = `config-${param.key}`;
  input.className = "config-input";
  input.type = param.type === "str" ? "text" : "number";
  if (param.type === "int") input.step = "1";
  if (param.type === "float") input.step = "any";
  input.value = param.value;
  input.placeholder = param.default;
  if (param.value !== param.default) input.classList.add("modified");

  input.addEventListener("change", () => {
    pendingConfigSave = true;
    browser.runtime.sendMessage({
      type: RuntimeMessageType.ConfigUpdate,
      key: param.key,
      value: input.value,
    });
  });

  item.appendChild(header);
  item.appendChild(input);
  return item;
}

// ============================================================
// Boot
// ============================================================

init();
