/**
 * Prompts page — displays prompt logs grouped by agent run.
 * Each run row can be expanded to show individual prompts,
 * and each prompt can be expanded to show messages and response.
 */

import {
  type PromptLogEntry,
  type PromptLogRun,
  type RuntimeMessage,
  RuntimeMessageType,
} from "../protocol.js";

const runsContainer = document.getElementById("runs")!;
const loading = document.getElementById("loading")!;
const agentFilter = document.getElementById("agent-filter") as HTMLSelectElement;

const loadMore = document.getElementById("load-more")!;
const loadMoreBtn = document.getElementById("load-more-btn")!;

let allRuns: PromptLogRun[] = [];
let hasMore = false;
const runElements = new Map<string, HTMLElement>();
const ACTIVE_TIMEOUT_MS = 10_000;

// --- Init ---

function init(): void {
  browser.runtime.onMessage.addListener(handleMessage);
  requestRuns(0);
  agentFilter.addEventListener("change", () => {
    allRuns = [];
    requestRuns(0);
  });
  loadMoreBtn.addEventListener("click", () => {
    requestRuns(allRuns.length);
  });
}

function requestRuns(offset: number): void {
  const agentName = agentFilter.value || undefined;
  browser.runtime.sendMessage({
    type: RuntimeMessageType.PromptLogsRequest,
    agent_name: agentName,
    offset,
  });
}

function handleMessage(message: RuntimeMessage): void {
  if (message.type === RuntimeMessageType.PromptLogsResponse) {
    if (message.runs.length > 0 && allRuns.length > 0) {
      appendRuns(message.runs);
    } else {
      allRuns = message.runs;
      render();
    }
    hasMore = message.has_more;
    loadMore.classList.toggle("hidden", !hasMore);
    populateFilter(message.agent_names);
  } else if (message.type === RuntimeMessageType.PromptLogUpdate) {
    handlePromptUpdate(message.prompt);
  }
}

// --- Filter ---

function populateFilter(agentNames: string[]): void {
  const previous = agentFilter.value;
  agentFilter.innerHTML = '<option value="">All agents</option>';
  for (const agent of agentNames) {
    const option = document.createElement("option");
    option.value = agent;
    option.textContent = agent;
    agentFilter.appendChild(option);
  }
  agentFilter.value = previous;
}

// --- Real-time updates ---

function handlePromptUpdate(prompt: PromptLogEntry & { run_id: string }): void {
  const filter = agentFilter.value;
  if (filter && prompt.agent_name !== filter) return;

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

  const oldHeader = row.querySelector(".run-header")!;
  const newHeader = createRunHeader(run);
  row.replaceChild(newHeader, oldHeader);
  newHeader.addEventListener("click", () => row.classList.toggle("expanded"));

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
  loading.classList.add("hidden");

  const row = createRunRow(run);
  runsContainer.prepend(row);
  markRunActive(run.run_id, row);
}

function markRunActive(runId: string, row: HTMLElement): void {
  row.classList.add("active-run");
  setTimeout(() => {
    const latestRun = allRuns.find((r) => r.run_id === runId);
    if (!latestRun) return;
    const lastPromptTime = new Date(latestRun.ended_at).getTime();
    if (Date.now() - lastPromptTime >= ACTIVE_TIMEOUT_MS) {
      row.classList.remove("active-run");
    }
  }, ACTIVE_TIMEOUT_MS);
}

// --- Rendering ---

function render(): void {
  loading.classList.add("hidden");
  runsContainer.innerHTML = "";
  runElements.clear();

  if (allRuns.length === 0) {
    const label = agentFilter.value || "any agent";
    loading.textContent = `No prompt logs for ${label}.`;
    loading.classList.remove("hidden");
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

  const header = createRunHeader(run);
  row.appendChild(header);

  if (run.run_outcome) {
    row.appendChild(createRunOutcome(run.run_outcome));
  }

  const promptsContainer = document.createElement("div");
  promptsContainer.className = "run-prompts";
  for (let i = 0; i < run.prompts.length; i++) {
    promptsContainer.appendChild(createPromptRow(run.prompts[i], i + 1));
  }
  row.appendChild(promptsContainer);

  header.addEventListener("click", () => {
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
  agent.textContent = run.agent_name;
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

  if (prompt.prompt_type) {
    const typeEl = document.createElement("span");
    typeEl.className = "prompt-type";
    typeEl.textContent = prompt.prompt_type;
    header.appendChild(typeEl);
  }

  if (prompt.has_tools) {
    const toolsEl = document.createElement("span");
    toolsEl.className = "prompt-tools";
    toolsEl.innerHTML = '<i class="fa-solid fa-wrench"></i>';
    header.appendChild(toolsEl);
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
    detail.appendChild(createSection(role, content));
  }

  detail.appendChild(createSection("response", renderResponse(prompt.response)));

  if (prompt.thinking) {
    detail.appendChild(createSection("thinking", prompt.thinking));
  }

  return detail;
}

function createSection(label: string, content: string): HTMLElement {
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

// --- Helpers ---

function extractPromptType(run: PromptLogRun): string {
  for (const prompt of run.prompts) {
    if (prompt.prompt_type) return prompt.prompt_type;
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

// --- Boot ---

init();
