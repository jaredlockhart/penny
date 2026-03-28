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

  await rehydrateHistory();
  listenToBackground();
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
    addMessage(message.content, MS.Penny);
  } else if (message.type === RuntimeMessageType.Typing) {
    setTyping(message.active);
  } else if (message.type === RuntimeMessageType.PermissionRequest) {
    showPermissionDialog(message.request_id, message.domain, message.url);
  }
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

  addMessage(text, MS.User);
  browser.runtime.sendMessage({ type: RuntimeMessageType.SendChat, content: text });
  inputEl.value = "";
  inputEl.rows = 1;
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
