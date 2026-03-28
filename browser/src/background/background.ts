/**
 * Background script — owns the WebSocket connection to Penny server.
 * Handles tool requests, permission prompts, and active tab tracking.
 * The sidebar communicates with this script via browser.runtime messaging.
 */

import {
  type ConnectionState,
  ConnectionState as CS,
  DomainPermission as DP,
  MAX_PAGE_CONTEXT_CHARS,
  type PageContext,
  RECONNECT_DELAY_MS,
  type RuntimeMessage,
  RuntimeMessageType,
  SERVER_URL,
  STORAGE_KEY_DEVICE_LABEL,
  type WsIncomingPayload,
  WsIncomingType,
  type WsIncomingToolRequestPayload,
  WsOutgoingType,
} from "../protocol.js";
import {
  checkDomainPermission,
  extractDomain,
  requestPermissionFromUser,
  storeDomainPermission,
} from "./permissions.js";
import { browseUrl } from "./tools/browse_url.js";

let ws: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let deviceLabel: string | null = null;
let connectionState: ConnectionState = CS.Disconnected;
let currentPageContext: PageContext | null = null;

// URLs we should never try to extract from
const SKIP_URL_PREFIXES = ["about:", "moz-extension:", "chrome:", "data:", "file:"];

// --- Lifecycle ---

async function init(): Promise<void> {
  const stored = await browser.storage.local.get(STORAGE_KEY_DEVICE_LABEL);
  deviceLabel = (stored[STORAGE_KEY_DEVICE_LABEL] as string) ?? null;

  browser.runtime.onMessage.addListener(handleRuntimeMessage);
  browser.storage.onChanged.addListener(handleStorageChange);
  browser.tabs.onActivated.addListener(handleTabActivated);
  browser.tabs.onUpdated.addListener(handleTabUpdated);

  if (deviceLabel) {
    connect();
  }

  extractFromActiveTab();
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

// --- Active tab tracking ---

function handleTabActivated(): void {
  extractFromActiveTab();
}

function handleTabUpdated(
  _tabId: number,
  changeInfo: browser.tabs._OnUpdatedChangeInfo,
): void {
  if (changeInfo.status === "complete") {
    extractFromActiveTab();
  }
}

async function extractFromActiveTab(): Promise<void> {
  let favicon = "";
  try {
    const tabs = await browser.tabs.query({ active: true, currentWindow: true });
    const tab = tabs[0];
    if (!tab?.id || !tab.url || SKIP_URL_PREFIXES.some((p) => tab.url!.startsWith(p))) {
      currentPageContext = null;
      broadcastPageInfo("", "", "", "", false);
      return;
    }

    favicon = tab.favIconUrl ?? "";

    const results = await browser.tabs.executeScript(tab.id, {
      file: "/dist/content/extract_text.js",
      runAt: "document_idle",
    });

    if (results?.[0]) {
      const data = results[0] as { title: string; url: string; text: string; image: string };
      currentPageContext = {
        title: data.title,
        url: data.url,
        text: data.text.slice(0, MAX_PAGE_CONTEXT_CHARS),
        image: data.image,
      };
      broadcastPageInfo(data.title, data.url, favicon, data.image, true);
    } else {
      currentPageContext = null;
      broadcastPageInfo("", "", "", "", false);
    }
  } catch {
    currentPageContext = null;
    broadcastPageInfo("", "", "", "", false);
  }
}

function broadcastPageInfo(
  title: string, url: string, favicon: string, image: string, available: boolean,
): void {
  broadcastToSidebar({
    type: RuntimeMessageType.PageInfo,
    title,
    url,
    favicon,
    image,
    available,
  });
}

// --- Runtime messaging (sidebar ↔ background) ---

function handleRuntimeMessage(message: RuntimeMessage): void {
  if (message.type === RuntimeMessageType.SendChat) {
    sendChatToServer(message.content, message.include_page);
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
    } else if (data.type === WsIncomingType.ToolRequest) {
      handleToolRequest(data);
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

function sendChatToServer(content: string, includePage: boolean): void {
  if (!ws || ws.readyState !== WebSocket.OPEN || !deviceLabel) return;
  const payload: Record<string, unknown> = {
    type: WsOutgoingType.Message,
    content,
    sender: deviceLabel,
  };
  if (includePage && currentPageContext) {
    payload.page_context = currentPageContext;
  }
  ws.send(JSON.stringify(payload));
}

function sendToolResponse(requestId: string, result?: string, error?: string): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({
    type: WsOutgoingType.ToolResponse,
    request_id: requestId,
    result,
    error,
  }));
}

// --- Tool request handling ---

async function handleToolRequest(request: WsIncomingToolRequestPayload): Promise<void> {
  const { request_id, tool, arguments: args } = request;

  try {
    if (tool === "browse_url") {
      const result = await executeBrowseUrl(request_id, args);
      sendToolResponse(request_id, result);
    } else {
      sendToolResponse(request_id, undefined, `Unknown tool: ${tool}`);
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    sendToolResponse(request_id, undefined, message);
  }
}

async function executeBrowseUrl(
  requestId: string,
  args: Record<string, unknown>,
): Promise<string> {
  const url = args.url as string;
  if (!url) throw new Error("Missing required argument: url");

  const domain = extractDomain(url);
  const permission = await checkDomainPermission(domain);

  if (permission === "blocked") {
    throw new Error(`Domain ${domain} is blocked by user`);
  }

  if (permission === "unknown") {
    const allowed = await requestPermissionFromUser(requestId, domain, url);
    await storeDomainPermission(domain, allowed ? DP.Allowed : DP.Blocked);
    if (!allowed) {
      throw new Error(`User denied access to ${domain}`);
    }
  }

  return await browseUrl(url);
}

// --- Sidebar state sync ---

browser.runtime.onConnect.addListener((port) => {
  if (port.name === "sidebar") {
    port.postMessage({ type: RuntimeMessageType.ConnectionState, state: connectionState });
    // Send current page info so the toggle is populated immediately
    if (currentPageContext) {
      port.postMessage({
        type: RuntimeMessageType.PageInfo,
        title: currentPageContext.title,
        url: currentPageContext.url,
        favicon: "",
        image: currentPageContext.image,
        available: true,
      });
    }
  }
});

// --- Boot ---

init();
