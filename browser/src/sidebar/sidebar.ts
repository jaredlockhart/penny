/**
 * Sidebar script — pure UI layer.
 * Communicates with the background script via browser.runtime messaging.
 * No direct WebSocket connection.
 */

import {
  type ConnectionState,
  ConnectionState as CS,
  MAX_STORED_MESSAGES,
  type MessageSender,
  MessageSender as MS,
  type PreferenceItem,
  type RuntimeMessage,
  RuntimeMessageType,
  STORAGE_KEY_CHAT_HISTORY,
  STORAGE_KEY_DEVICE_LABEL,
  type StoredMessage,
  TEXTAREA_LINE_HEIGHT,
  TEXTAREA_MAX_ROWS,
  TYPING_INDICATOR_TEXT,
} from "../protocol.js";

// DOM refs (resolved after init)
let messagesEl: HTMLElement;
let inputEl: HTMLTextAreaElement;
let sendBtn: HTMLButtonElement;
let statusEl: HTMLElement;

// Page context for decorating the next response
let pendingPageRef: { title: string; url: string; image: string } | null = null;
let currentPageImage = "";

// Tab state
type TabName = "chat" | "likes" | "dislikes";
let activeTab: TabName = "chat";
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

function showRegister(): void {
  document.getElementById("register")!.classList.remove("hidden");
  document.getElementById("chat")!.classList.add("hidden");

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
  document.getElementById("register")!.classList.add("hidden");
  document.getElementById("chat")!.classList.remove("hidden");

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

  document.getElementById("nav-chat")!.addEventListener("click", () => activateTab("chat"));
  document.getElementById("nav-likes")!.addEventListener("click", () => activateTab("likes"));
  document.getElementById("nav-dislikes")!.addEventListener("click", () => activateTab("dislikes"));
  document.getElementById("nav-thoughts")!.addEventListener("click", () => {
    browser.tabs.create({ url: browser.runtime.getURL("feed/feed.html") });
  });

  setupPrefsAdd("positive", "likes");
  setupPrefsAdd("negative", "dislikes");

  await rehydrateHistory();
  listenToBackground();
}

// --- Tab switching ---

function activateTab(tab: TabName): void {
  activeTab = tab;

  document.getElementById("nav-chat")!.classList.toggle("active", tab === "chat");
  document.getElementById("nav-likes")!.classList.toggle("active", tab === "likes");
  document.getElementById("nav-dislikes")!.classList.toggle("active", tab === "dislikes");

  document.getElementById("messages-wrapper")!.classList.toggle("hidden", tab !== "chat");
  document.getElementById("input-area")!.classList.toggle("hidden", tab !== "chat");

  if (tab !== "chat") {
    document.getElementById("page-context-bar")!.classList.add("hidden");
    document.getElementById("permission-dialog")!.classList.add("hidden");
  } else {
    updatePageContextBar(
      lastPageInfo.title, lastPageInfo.url, lastPageInfo.favicon,
      lastPageInfo.image, lastPageInfo.available,
    );
  }

  document.getElementById("likes-panel")!.classList.toggle("hidden", tab !== "likes");
  document.getElementById("dislikes-panel")!.classList.toggle("hidden", tab !== "dislikes");

  if (tab === "likes") {
    browser.runtime.sendMessage({ type: RuntimeMessageType.PreferencesRequest, valence: "positive" });
  } else if (tab === "dislikes") {
    browser.runtime.sendMessage({ type: RuntimeMessageType.PreferencesRequest, valence: "negative" });
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
    del.textContent = "×";
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

// --- Background communication ---

function listenToBackground(): void {
  // Get current connection state from background
  const port = browser.runtime.connect({ name: "sidebar" });
  port.onMessage.addListener((message: object) => {
    handleBackgroundMessage(message as RuntimeMessage);
  });
  port.onDisconnect.addListener(() => {
    setStatus(CS.Disconnected);
  });

  // Listen for ongoing broadcasts
  browser.runtime.onMessage.addListener(handleBackgroundMessage);
}

function handleBackgroundMessage(message: RuntimeMessage): void {
  if (message.type === RuntimeMessageType.ConnectionState) {
    setStatus(message.state);
  } else if (message.type === RuntimeMessageType.ChatMessage) {
    setTyping(false);
    let content = message.content;
    if (pendingPageRef) {
      content = buildPageHeader(pendingPageRef) + content.replace(/<img[^>]*><br>/g, "");
      pendingPageRef = null;
    }
    addMessage(content, MS.Penny);
    setInputEnabled(true);
  } else if (message.type === RuntimeMessageType.Typing) {
    setTyping(message.active);
  } else if (message.type === RuntimeMessageType.PermissionRequest) {
    showPermissionDialog(message.request_id, message.domain, message.url);
  } else if (message.type === RuntimeMessageType.ThoughtCount) {
    const countEl = document.getElementById("nav-thoughts-count");
    if (countEl) countEl.textContent = message.count > 0 ? `(${message.count})` : "";
  } else if (message.type === RuntimeMessageType.PageInfo) {
    updatePageContextBar(message.title, message.url, message.favicon, message.image, message.available);
  } else if (message.type === RuntimeMessageType.PreferencesResponse) {
    renderPreferences(message.valence, message.preferences);
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

  if (!available || !title || activeTab !== "chat") {
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

  // Re-scroll after images load (dimensions unknown until rendered)
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

function setTyping(active: boolean): void {
  let indicator = document.getElementById("typing");
  if (active && !indicator) {
    indicator = document.createElement("div");
    indicator.id = "typing";
    indicator.className = "typing";
    indicator.textContent = TYPING_INDICATOR_TEXT;
    messagesEl.appendChild(indicator);
    messagesEl.scrollTop = messagesEl.scrollHeight;
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
