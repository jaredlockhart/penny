import {
  type ConnectionState,
  ConnectionState as CS,
  type IncomingPayload,
  IncomingType,
  type MessageSender,
  MessageSender as MS,
  OutgoingType,
  RECONNECT_DELAY_MS,
  SERVER_URL,
  STORAGE_KEY_DEVICE_LABEL,
  TEXTAREA_LINE_HEIGHT,
  TEXTAREA_MAX_ROWS,
  TYPING_INDICATOR_TEXT,
} from "../protocol.js";

let ws: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let deviceLabel: string | null = null;

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
    deviceLabel = stored[STORAGE_KEY_DEVICE_LABEL] as string;
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

  deviceLabel = label;
  await browser.storage.local.set({ [STORAGE_KEY_DEVICE_LABEL]: label });
  showChat();
}

function showChat(): void {
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

  setStatus(CS.Disconnected);
  connect();
}

// --- Chat UI ---

function addMessage(text: string, sender: MessageSender): void {
  const div = document.createElement("div");
  div.className = `message ${sender}`;
  if (sender === MS.Penny) {
    div.innerHTML = text;
  } else {
    div.textContent = text;
  }
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
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

// --- WebSocket ---

function connect(): void {
  if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
    return;
  }

  ws = new WebSocket(SERVER_URL);

  ws.addEventListener("open", () => {
    console.log("Connected to Penny server");
  });

  ws.addEventListener("message", (event: MessageEvent) => {
    const data: IncomingPayload = JSON.parse(event.data);

    if (data.type === IncomingType.Status && data.connected) {
      setStatus(CS.Connected);
    } else if (data.type === IncomingType.Message) {
      setTyping(false);
      addMessage(data.content, MS.Penny);
    } else if (data.type === IncomingType.Typing) {
      setTyping(data.active);
    }
  });

  ws.addEventListener("close", () => {
    setStatus(CS.Reconnecting);
    setTyping(false);
    scheduleReconnect();
  });

  ws.addEventListener("error", () => {
    // Error fires before close — close handler will reconnect
  });
}

function scheduleReconnect(): void {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, RECONNECT_DELAY_MS);
}

function send(): void {
  const text = inputEl.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN || !deviceLabel) return;

  addMessage(text, MS.User);
  ws.send(JSON.stringify({ type: OutgoingType.Message, content: text, sender: deviceLabel }));
  inputEl.value = "";
  inputEl.rows = 1;
}

// --- Boot ---

init();
