const SERVER_URL = "ws://localhost:9090";

let ws = null;
let reconnectTimer = null;
let deviceLabel = null;

// --- DOM refs (resolved after init) ---
let messagesEl, inputEl, sendBtn, statusEl;

// --- Registration ---

async function init() {
  statusEl = document.getElementById("status");
  const stored = await browser.storage.local.get("deviceLabel");
  if (stored.deviceLabel) {
    deviceLabel = stored.deviceLabel;
    showChat();
  } else {
    showRegister();
  }
}

function showRegister() {
  document.getElementById("register").classList.remove("hidden");
  document.getElementById("chat").classList.add("hidden");

  const labelInput = document.getElementById("device-label");
  const registerBtn = document.getElementById("register-btn");

  registerBtn.addEventListener("click", () => saveLabel(labelInput));
  labelInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") saveLabel(labelInput);
  });

  labelInput.focus();
}

async function saveLabel(labelInput) {
  const label = labelInput.value.trim();
  if (!label) return;

  deviceLabel = label;
  await browser.storage.local.set({ deviceLabel: label });
  showChat();
}

function showChat() {
  document.getElementById("register").classList.add("hidden");
  document.getElementById("chat").classList.remove("hidden");

  messagesEl = document.getElementById("messages");
  inputEl = document.getElementById("input");
  sendBtn = document.getElementById("send");

  sendBtn.addEventListener("click", send);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  inputEl.addEventListener("input", autoResize);

  setStatus("disconnected");
  connect();
}

// --- Chat UI ---

function addMessage(text, sender) {
  const div = document.createElement("div");
  div.className = `message ${sender}`;
  if (sender === "penny") {
    div.innerHTML = text;
  } else {
    div.textContent = text;
  }
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setStatus(state) {
  statusEl.textContent = state;
  statusEl.className = state;
  if (sendBtn) sendBtn.disabled = state !== "connected";
}

function setTyping(active) {
  let indicator = document.getElementById("typing");
  if (active && !indicator) {
    indicator = document.createElement("div");
    indicator.id = "typing";
    indicator.className = "typing";
    indicator.textContent = "Penny is thinking...";
    messagesEl.appendChild(indicator);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } else if (!active && indicator) {
    indicator.remove();
  }
}

function autoResize() {
  inputEl.rows = 1;
  const lines = Math.ceil(inputEl.scrollHeight / 20);
  inputEl.rows = Math.min(lines, 4);
}

// --- WebSocket ---

function connect() {
  if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
    return;
  }

  ws = new WebSocket(SERVER_URL);

  ws.addEventListener("open", () => {
    console.log("Connected to Penny server");
  });

  ws.addEventListener("message", (event) => {
    const data = JSON.parse(event.data);

    if (data.type === "status" && data.connected) {
      setStatus("connected");
    } else if (data.type === "message") {
      setTyping(false);
      addMessage(data.content, "penny");
    } else if (data.type === "typing") {
      setTyping(data.active);
    }
  });

  ws.addEventListener("close", () => {
    setStatus("reconnecting");
    setTyping(false);
    scheduleReconnect();
  });

  ws.addEventListener("error", () => {
    // Error fires before close — close handler will reconnect
  });
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, 3000);
}

function send() {
  const text = inputEl.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

  addMessage(text, "user");
  ws.send(JSON.stringify({ type: "message", content: text, sender: deviceLabel }));
  inputEl.value = "";
  inputEl.rows = 1;
}

// --- Boot ---

init();
