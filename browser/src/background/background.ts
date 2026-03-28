/**
 * Background script — owns the WebSocket connection to Penny server.
 * The sidebar communicates with this script via browser.runtime messaging.
 */

import {
  type ConnectionState,
  ConnectionState as CS,
  RECONNECT_DELAY_MS,
  type RuntimeMessage,
  RuntimeMessageType,
  SERVER_URL,
  STORAGE_KEY_DEVICE_LABEL,
  type WsIncomingPayload,
  WsIncomingType,
  WsOutgoingType,
} from "../protocol.js";

let ws: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let deviceLabel: string | null = null;
let connectionState: ConnectionState = CS.Disconnected;

// --- Lifecycle ---

async function init(): Promise<void> {
  const stored = await browser.storage.local.get(STORAGE_KEY_DEVICE_LABEL);
  deviceLabel = (stored[STORAGE_KEY_DEVICE_LABEL] as string) ?? null;

  browser.runtime.onMessage.addListener(handleRuntimeMessage);
  browser.storage.onChanged.addListener(handleStorageChange);

  if (deviceLabel) {
    connect();
  }
}

function handleStorageChange(
  changes: Record<string, browser.storage.StorageChange>,
): void {
  if (changes[STORAGE_KEY_DEVICE_LABEL]?.newValue) {
    deviceLabel = changes[STORAGE_KEY_DEVICE_LABEL].newValue as string;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      connect();
    }
  }
}

// --- Runtime messaging (sidebar ↔ background) ---

function handleRuntimeMessage(message: RuntimeMessage): void {
  if (message.type === RuntimeMessageType.SendChat) {
    sendToServer(message.content);
  }
}

function broadcastToSidebar(message: RuntimeMessage): void {
  browser.runtime.sendMessage(message).catch(() => {
    // Sidebar not open — ignore
  });
}

function setConnectionState(state: ConnectionState): void {
  connectionState = state;
  broadcastToSidebar({ type: RuntimeMessageType.ConnectionState, state });
}

// --- WebSocket ---

function connect(): void {
  if (!deviceLabel) return;
  if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
    return;
  }

  ws = new WebSocket(SERVER_URL);

  ws.addEventListener("open", () => {
    console.log("Background: connected to Penny server");
  });

  ws.addEventListener("message", (event: MessageEvent) => {
    const data: WsIncomingPayload = JSON.parse(event.data);

    if (data.type === WsIncomingType.Status && data.connected) {
      setConnectionState(CS.Connected);
    } else if (data.type === WsIncomingType.Message) {
      broadcastToSidebar({ type: RuntimeMessageType.ChatMessage, content: data.content });
    } else if (data.type === WsIncomingType.Typing) {
      broadcastToSidebar({ type: RuntimeMessageType.Typing, active: data.active });
    }
  });

  ws.addEventListener("close", () => {
    setConnectionState(CS.Reconnecting);
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

function sendToServer(content: string): void {
  if (!ws || ws.readyState !== WebSocket.OPEN || !deviceLabel) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.Message, content, sender: deviceLabel }));
}

// --- Public state (sidebar can query on open) ---

browser.runtime.onConnect.addListener((port) => {
  if (port.name === "sidebar") {
    port.postMessage({ type: RuntimeMessageType.ConnectionState, state: connectionState });
  }
});

// --- Boot ---

init();
