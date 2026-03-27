const messages = document.getElementById("messages");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const status = document.getElementById("status");

const SERVER_URL = "ws://localhost:9090";

let ws = null;
let reconnectTimer = null;

function addMessage(text, sender) {
  const div = document.createElement("div");
  div.className = `message ${sender}`;
  div.textContent = text;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

function setStatus(connected) {
  status.textContent = connected ? "connected" : "disconnected";
  status.className = connected ? "connected" : "disconnected";
  sendBtn.disabled = !connected;
}

function setTyping(active) {
  let indicator = document.getElementById("typing");
  if (active && !indicator) {
    indicator = document.createElement("div");
    indicator.id = "typing";
    indicator.className = "typing";
    indicator.textContent = "Penny is thinking...";
    messages.appendChild(indicator);
    messages.scrollTop = messages.scrollHeight;
  } else if (!active && indicator) {
    indicator.remove();
  }
}

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
      setStatus(true);
    } else if (data.type === "message") {
      setTyping(false);
      addMessage(data.content, "penny");
    } else if (data.type === "typing") {
      setTyping(data.active);
    }
  });

  ws.addEventListener("close", () => {
    setStatus(false);
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
  const text = input.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

  addMessage(text, "user");
  ws.send(JSON.stringify({ type: "message", content: text }));
  input.value = "";
  input.rows = 1;
}

sendBtn.addEventListener("click", send);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

// Auto-resize textarea
input.addEventListener("input", () => {
  input.rows = 1;
  const lines = Math.ceil(input.scrollHeight / 20);
  input.rows = Math.min(lines, 4);
});

setStatus(false);
connect();
