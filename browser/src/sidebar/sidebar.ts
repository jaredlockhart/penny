/**
 * Sidebar script — pure UI layer.
 * Communicates with the background script via browser.runtime messaging.
 * No direct WebSocket connection.
 */

import {
  type ConnectionState,
  ConnectionState as CS,
  type DomainAllowlist,
  DomainPermission as DP,
  MAX_STORED_MESSAGES,
  type MessageSender,
  MessageSender as MS,
  type PreferenceItem,
  type RuntimeConfigParam,
  type RuntimeMessage,
  RuntimeMessageType,
  STORAGE_KEY_CHAT_HISTORY,
  STORAGE_KEY_DEVICE_LABEL,
  STORAGE_KEY_DOMAIN_ALLOWLIST,
  type StoredMessage,
  TEXTAREA_LINE_HEIGHT,
  TEXTAREA_MAX_ROWS,
} from "../protocol.js";

// DOM refs (resolved after init)
let messagesEl: HTMLElement;
let inputEl: HTMLTextAreaElement;
let sendBtn: HTMLButtonElement;
let statusEl: HTMLElement;

// Page context for decorating the next response
let pendingPageRef: { title: string; url: string; image: string } | null = null;
let currentPageImage = "";

// Toast
let toastTimer: ReturnType<typeof setTimeout> | null = null;

function showToast(text: string): void {
  const toast = document.getElementById("toast")!;
  toast.textContent = text;
  toast.classList.add("visible");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 2000);
}

// View state
type View = "register" | "chat" | "settings";
type SettingsTab = "likes" | "dislikes" | "domains" | "config";

let activeView: View = "register";
let activeSettingsTab: SettingsTab = "likes";
let pendingConfigSave = false;
let lastPageInfo = { title: "", url: "", favicon: "", image: "", available: false };

// --- Registration ---

async function init(): Promise<void> {
  statusEl = document.getElementById("status")!;
  const stored = await browser.storage.local.get(STORAGE_KEY_DEVICE_LABEL);
  if (stored[STORAGE_KEY_DEVICE_LABEL]) {
    showChat();
  } else {
    showRegister();
  }
}

function showView(view: View): void {
  document.getElementById("register")!.classList.toggle("hidden", view !== "register");
  document.getElementById("chat")!.classList.toggle("hidden", view !== "chat");
  document.getElementById("settings")!.classList.toggle("hidden", view !== "settings");
  document.getElementById("nav-settings")?.classList.toggle("active", view === "settings");
  activeView = view;
}

function showRegister(): void {
  showView("register");

  const labelInput = document.getElementById("device-label") as HTMLInputElement;
  const registerBtn = document.getElementById("register-btn")!;

  registerBtn.addEventListener("click", () => saveLabel(labelInput));
  labelInput.addEventListener("keydown", (e: KeyboardEvent) => {
    if (e.key === "Enter") saveLabel(labelInput);
  });

  labelInput.focus();
}

async function saveLabel(labelInput: HTMLInputElement): Promise<void> {
  const label = labelInput.value.trim();
  if (!label) return;

  await browser.storage.local.set({ [STORAGE_KEY_DEVICE_LABEL]: label });
  showChat();
}

async function showChat(): Promise<void> {
  showView("chat");

  messagesEl = document.getElementById("messages")!;
  inputEl = document.getElementById("input") as HTMLTextAreaElement;
  sendBtn = document.getElementById("send") as HTMLButtonElement;

  sendBtn.addEventListener("click", send);
  inputEl.addEventListener("keydown", (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  inputEl.addEventListener("input", autoResize);

  document.getElementById("nav-thoughts")!.addEventListener("click", () => {
    browser.tabs.create({ url: browser.runtime.getURL("feed/feed.html") });
  });
  document.getElementById("nav-settings")!.addEventListener("click", () => {
    showView("settings");
    activateSettingsTab(activeSettingsTab);
  });

  document.getElementById("settings-back")!.addEventListener("click", () => showView("chat"));

  for (const btn of Array.from(document.querySelectorAll(".settings-tab"))) {
    btn.addEventListener("click", () => {
      activateSettingsTab(btn.getAttribute("data-stab") as SettingsTab);
    });
  }

  setupPrefsAdd("positive", "likes");
  setupPrefsAdd("negative", "dislikes");
  setupDomainsAdd();

  await rehydrateHistory();
  listenToBackground();
}

// --- Settings tab switching ---

function activateSettingsTab(tab: SettingsTab): void {
  activeSettingsTab = tab;

  for (const btn of Array.from(document.querySelectorAll(".settings-tab"))) {
    btn.classList.toggle("active", btn.getAttribute("data-stab") === tab);
  }

  document.getElementById("stab-likes")!.classList.toggle("hidden", tab !== "likes");
  document.getElementById("stab-dislikes")!.classList.toggle("hidden", tab !== "dislikes");
  document.getElementById("stab-domains")!.classList.toggle("hidden", tab !== "domains");
  document.getElementById("stab-config")!.classList.toggle("hidden", tab !== "config");

  if (tab === "likes") {
    browser.runtime.sendMessage({ type: RuntimeMessageType.PreferencesRequest, valence: "positive" });
  } else if (tab === "dislikes") {
    browser.runtime.sendMessage({ type: RuntimeMessageType.PreferencesRequest, valence: "negative" });
  } else if (tab === "domains") {
    loadAndRenderDomains();
  } else if (tab === "config") {
    browser.runtime.sendMessage({ type: RuntimeMessageType.ConfigRequest });
  }
}

// --- Preferences UI ---

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

    const count = document.createElement("span");
    count.className = "pref-count";
    count.textContent = `(${pref.mention_count})`;

    const del = document.createElement("button");
    del.className = "pref-delete";
    del.innerHTML = '<i class="fa-solid fa-xmark"></i>';
    del.setAttribute("aria-label", `Remove ${pref.content}`);
    del.addEventListener("click", () => {
      browser.runtime.sendMessage({ type: RuntimeMessageType.PreferenceDelete, preference_id: pref.id });
    });

    row.appendChild(name);
    row.appendChild(count);
    row.appendChild(del);
    listEl.appendChild(row);
  }
}

function setupPrefsAdd(valence: string, prefix: string): void {
  const input = document.getElementById(`${prefix}-input`) as HTMLInputElement;
  const btn = document.getElementById(`${prefix}-add-btn`) as HTMLButtonElement;

  function add(): void {
    const content = input.value.trim();
    if (!content) return;
    browser.runtime.sendMessage({ type: RuntimeMessageType.PreferenceAdd, valence, content });
    input.value = "";
  }

  btn.addEventListener("click", add);
  input.addEventListener("keydown", (e: KeyboardEvent) => {
    if (e.key === "Enter") add();
  });
}

// --- Domains UI ---

async function loadAndRenderDomains(): Promise<void> {
  const stored = await browser.storage.local.get(STORAGE_KEY_DOMAIN_ALLOWLIST);
  const allowlist: DomainAllowlist = (stored[STORAGE_KEY_DOMAIN_ALLOWLIST] as DomainAllowlist) ?? {};
  renderDomains(allowlist);
}

function renderDomains(allowlist: DomainAllowlist): void {
  const listEl = document.getElementById("domains-list")!;
  listEl.innerHTML = "";

  const entries = Object.entries(allowlist).sort(([a], [b]) => a.localeCompare(b));

  if (entries.length === 0) {
    const empty = document.createElement("div");
    empty.className = "prefs-empty";
    empty.textContent = "No domains saved yet.";
    listEl.appendChild(empty);
    return;
  }

  for (const [domain, permission] of entries) {
    const row = document.createElement("div");
    row.className = "domain-row";

    const name = document.createElement("span");
    name.className = "domain-name";
    name.textContent = domain;

    const status = document.createElement("button");
    status.className = `domain-status ${permission}`;
    status.textContent = permission === DP.Allowed ? "Allowed" : "Blocked";
    status.title = "Click to toggle";
    status.addEventListener("click", async () => {
      const next = permission === DP.Allowed ? DP.Blocked : DP.Allowed;
      await updateDomainPermission(domain, next);
    });

    const del = document.createElement("button");
    del.className = "pref-delete";
    del.innerHTML = '<i class="fa-solid fa-xmark"></i>';
    del.setAttribute("aria-label", `Remove ${domain}`);
    del.addEventListener("click", async () => {
      await deleteDomainPermission(domain);
    });

    row.appendChild(name);
    row.appendChild(status);
    row.appendChild(del);
    listEl.appendChild(row);
  }
}

async function updateDomainPermission(domain: string, permission: string): Promise<void> {
  const stored = await browser.storage.local.get(STORAGE_KEY_DOMAIN_ALLOWLIST);
  const allowlist: DomainAllowlist = (stored[STORAGE_KEY_DOMAIN_ALLOWLIST] as DomainAllowlist) ?? {};
  allowlist[domain] = permission as DomainAllowlist[string];
  await browser.storage.local.set({ [STORAGE_KEY_DOMAIN_ALLOWLIST]: allowlist });
  await loadAndRenderDomains();
}

async function deleteDomainPermission(domain: string): Promise<void> {
  const stored = await browser.storage.local.get(STORAGE_KEY_DOMAIN_ALLOWLIST);
  const allowlist: DomainAllowlist = (stored[STORAGE_KEY_DOMAIN_ALLOWLIST] as DomainAllowlist) ?? {};
  delete allowlist[domain];
  await browser.storage.local.set({ [STORAGE_KEY_DOMAIN_ALLOWLIST]: allowlist });
  await loadAndRenderDomains();
}

function setupDomainsAdd(): void {
  const input = document.getElementById("domains-input") as HTMLInputElement;
  const select = document.getElementById("domains-permission") as HTMLSelectElement;
  const btn = document.getElementById("domains-add-btn")!;

  async function add(): Promise<void> {
    const raw = input.value.trim().toLowerCase();
    if (!raw) return;
    // Strip protocol and path — store just the hostname
    const domain = raw.replace(/^https?:\/\//, "").replace(/\/.*$/, "");
    if (!domain) return;
    await updateDomainPermission(domain, select.value);
    input.value = "";
  }

  btn.addEventListener("click", add);
  input.addEventListener("keydown", (e: KeyboardEvent) => {
    if (e.key === "Enter") add();
  });
}

// --- Config UI ---

function renderConfig(params: RuntimeConfigParam[]): void {
  const panel = document.getElementById("stab-config")!;
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

  const label = document.createElement("label");
  label.className = "config-label";
  label.textContent = param.description;
  label.htmlFor = `config-${param.key}`;

  const key = document.createElement("div");
  key.className = "config-key";
  key.textContent = param.key;

  const input = document.createElement("input");
  input.id = `config-${param.key}`;
  input.className = "config-input";
  input.type = param.type === "str" ? "text" : "number";
  if (param.type === "int") input.step = "1";
  if (param.type === "float") input.step = "any";
  input.value = param.value;
  if (param.value !== param.default) input.classList.add("modified");

  input.addEventListener("change", () => {
    pendingConfigSave = true;
    browser.runtime.sendMessage({
      type: RuntimeMessageType.ConfigUpdate,
      key: param.key,
      value: input.value,
    });
  });

  item.appendChild(label);
  item.appendChild(key);
  item.appendChild(input);
  return item;
}

// --- Background communication ---

function listenToBackground(): void {
  const port = browser.runtime.connect({ name: "sidebar" });
  port.onMessage.addListener((message: object) => {
    handleBackgroundMessage(message as RuntimeMessage);
  });
  port.onDisconnect.addListener(() => {
    setStatus(CS.Disconnected);
  });

  browser.runtime.onMessage.addListener(handleBackgroundMessage);
}

function handleBackgroundMessage(message: RuntimeMessage): void {
  if (message.type === RuntimeMessageType.ConnectionState) {
    setStatus(message.state);
  } else if (message.type === RuntimeMessageType.ChatMessage) {
    if (activeView === "settings") showView("chat");
    setTyping(false);
    let content = message.content;
    if (pendingPageRef) {
      content = buildPageHeader(pendingPageRef) + content.replace(/<img[^>]*><br>/g, "");
      pendingPageRef = null;
    }
    addMessage(content, MS.Penny);
    setInputEnabled(true);
  } else if (message.type === RuntimeMessageType.Typing) {
    setTyping(message.active, message.content);
  } else if (message.type === RuntimeMessageType.PermissionRequest) {
    if (activeView === "settings") showView("chat");
    showPermissionDialog(message.request_id, message.domain, message.url);
  } else if (message.type === RuntimeMessageType.ThoughtCount) {
    const countEl = document.getElementById("nav-thoughts-count");
    if (countEl) countEl.textContent = message.count > 0 ? ` (${message.count})` : "";
  } else if (message.type === RuntimeMessageType.PageInfo) {
    updatePageContextBar(message.title, message.url, message.favicon, message.image, message.available);
  } else if (message.type === RuntimeMessageType.PreferencesResponse) {
    renderPreferences(message.valence, message.preferences);
  } else if (message.type === RuntimeMessageType.ConfigResponse) {
    renderConfig(message.params);
    if (pendingConfigSave) {
      pendingConfigSave = false;
      showToast("Saved");
    }
  }
}

// --- Page context bar ---

function updatePageContextBar(
  title: string, url: string, favicon: string, image: string, available: boolean,
): void {
  lastPageInfo = { title, url, favicon, image, available };

  const bar = document.getElementById("page-context-bar")!;
  const titleEl = document.getElementById("page-context-title")!;
  const faviconEl = document.getElementById("page-context-favicon") as HTMLImageElement;
  const toggle = document.getElementById("page-context-toggle") as HTMLInputElement;

  currentPageImage = image;

  if (!available || !title || activeView !== "chat") {
    bar.classList.add("hidden");
    if (!available || !title) toggle.checked = false;
    return;
  }

  const urlChanged = bar.dataset.url !== url;
  titleEl.textContent = title;
  faviconEl.src = favicon || "";
  bar.dataset.url = url;
  if (urlChanged) {
    toggle.checked = true;
  }
  bar.classList.remove("hidden");
}

// --- Permission dialog ---

function showPermissionDialog(requestId: string, domain: string, url: string): void {
  const dialog = document.getElementById("permission-dialog")!;
  document.getElementById("permission-domain")!.textContent = domain;
  document.getElementById("permission-url")!.textContent = url;
  dialog.classList.remove("hidden");

  const allowBtn = document.getElementById("permission-allow")!;
  const denyBtn = document.getElementById("permission-deny")!;

  function respond(allowed: boolean): void {
    browser.runtime.sendMessage({
      type: RuntimeMessageType.PermissionResponse,
      request_id: requestId,
      allowed,
    });
    dialog.classList.add("hidden");
    allowBtn.removeEventListener("click", onAllow);
    denyBtn.removeEventListener("click", onDeny);
  }

  function onAllow(): void { respond(true); }
  function onDeny(): void { respond(false); }

  allowBtn.addEventListener("click", onAllow);
  denyBtn.addEventListener("click", onDeny);
}

function send(): void {
  const text = inputEl.value.trim();
  if (!text) return;

  const toggle = document.getElementById("page-context-toggle") as HTMLInputElement;
  const includePage = toggle?.checked ?? false;
  const bar = document.getElementById("page-context-bar")!;
  const titleEl = document.getElementById("page-context-title")!;

  if (includePage && titleEl.textContent) {
    pendingPageRef = {
      title: titleEl.textContent,
      url: bar.dataset.url ?? "",
      image: currentPageImage,
    };
  } else {
    pendingPageRef = null;
  }

  addMessage(text, MS.User);
  setInputEnabled(false);
  browser.runtime.sendMessage({
    type: RuntimeMessageType.SendChat,
    content: text,
    include_page: includePage,
  });
  inputEl.value = "";
  inputEl.rows = 1;
}

function setInputEnabled(enabled: boolean): void {
  inputEl.disabled = !enabled;
  sendBtn.disabled = !enabled;
  const toggle = document.getElementById("page-context-toggle") as HTMLInputElement;
  if (toggle) toggle.disabled = !enabled;
}

function buildPageHeader(ref: { title: string; url: string; image: string }): string {
  const img = ref.image
    ? `<img src="${ref.image}" alt="${ref.title}">`
    : "";
  const link = `<a href="${ref.url}" target="_blank">${ref.title}</a>`;
  return `<div class="page-header">${img}<div class="page-header-label">In response to ${link}</div></div>`;
}

// --- Chat history persistence ---

async function rehydrateHistory(): Promise<void> {
  const stored = await browser.storage.local.get(STORAGE_KEY_CHAT_HISTORY);
  const messages: StoredMessage[] = stored[STORAGE_KEY_CHAT_HISTORY] ?? [];
  for (const msg of messages) {
    renderMessage(msg.text, msg.sender, false);
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function persistMessage(text: string, sender: MessageSender): Promise<void> {
  const stored = await browser.storage.local.get(STORAGE_KEY_CHAT_HISTORY);
  const messages: StoredMessage[] = stored[STORAGE_KEY_CHAT_HISTORY] ?? [];
  messages.push({ text, sender });
  if (messages.length > MAX_STORED_MESSAGES) {
    messages.splice(0, messages.length - MAX_STORED_MESSAGES);
  }
  await browser.storage.local.set({ [STORAGE_KEY_CHAT_HISTORY]: messages });
}

// --- Chat UI ---

function renderMessage(text: string, sender: MessageSender, animate = true): void {
  const div = document.createElement("div");
  div.className = `message ${sender}`;
  if (sender === MS.Penny) {
    div.innerHTML = text;
  } else {
    div.textContent = text;
  }
  messagesEl.appendChild(div);
  if (!animate) return;

  scrollToMessage(div);

  const images = Array.from(div.querySelectorAll("img"));
  for (const img of images) {
    img.addEventListener("load", () => scrollToMessage(div), { once: true });
  }
}

function scrollToMessage(div: HTMLElement): void {
  const fitsInView = div.offsetHeight <= messagesEl.clientHeight;
  div.scrollIntoView({ block: fitsInView ? "end" : "start", behavior: "smooth" });
}

function addMessage(text: string, sender: MessageSender): void {
  renderMessage(text, sender);
  persistMessage(text, sender);
}

function setStatus(state: ConnectionState): void {
  statusEl.className = state;
  if (sendBtn) sendBtn.disabled = state !== CS.Connected;
}

function typingHTML(text: string): string {
  return `${text}<span class="typing-dots"><span>.</span><span>.</span><span>.</span></span>`;
}

function setTyping(active: boolean, content?: string): void {
  const text = content ?? "Penny is thinking";
  let indicator = document.getElementById("typing");
  if (active && !indicator) {
    indicator = document.createElement("div");
    indicator.id = "typing";
    indicator.className = "typing";
    indicator.innerHTML = typingHTML(text);
    messagesEl.appendChild(indicator);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } else if (active && indicator && content) {
    indicator.innerHTML = typingHTML(text);
  } else if (!active && indicator) {
    indicator.remove();
  }
}

function autoResize(): void {
  inputEl.rows = 1;
  const lines = Math.ceil(inputEl.scrollHeight / TEXTAREA_LINE_HEIGHT);
  inputEl.rows = Math.min(lines, TEXTAREA_MAX_ROWS);
}

// --- Boot ---

init();
