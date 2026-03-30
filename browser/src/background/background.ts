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
  THOUGHTS_POLL_INTERVAL_MS,
  type RuntimeMessage,
  RuntimeMessageType,
  SERVER_URL,
  STORAGE_KEY_DEVICE_LABEL,
  STORAGE_KEY_TOOL_USE,
  type WsIncomingPayload,
  WsIncomingType,
  type WsIncomingToolRequestPayload,
  WsIncomingType as WsIn,
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
      sendHeartbeat();
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
  } else if (message.type === RuntimeMessageType.ThoughtsRequest) {
    requestThoughts();
  } else if (message.type === RuntimeMessageType.ThoughtReaction) {
    sendThoughtReaction(message.thought_id, message.emoji);
  } else if (message.type === RuntimeMessageType.PreferencesRequest) {
    requestPreferences(message.valence);
  } else if (message.type === RuntimeMessageType.PreferenceAdd) {
    sendPreferenceAdd(message.valence, message.content);
  } else if (message.type === RuntimeMessageType.PreferenceDelete) {
    sendPreferenceDelete(message.preference_id);
  } else if (message.type === RuntimeMessageType.ConfigRequest) {
    requestConfig();
  } else if (message.type === RuntimeMessageType.ConfigUpdate) {
    sendConfigUpdate(message.key, message.value);
  } else if (message.type === RuntimeMessageType.ToolUseToggle) {
    setToolUse(message.enabled);
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

  setConnectionState(CS.Reconnecting);
  ws = new WebSocket(SERVER_URL);

  ws.addEventListener("open", () => {
    console.log("Background: connected to Penny server");
  });

  ws.addEventListener("message", (event: MessageEvent) => {
    const data: WsIncomingPayload = JSON.parse(event.data);

    if (data.type === WsIncomingType.Status && data.connected) {
      setConnectionState(CS.Connected);
      sendRegister();
      sendCapabilities();
      requestThoughts();
      setInterval(requestThoughts, THOUGHTS_POLL_INTERVAL_MS);
    } else if (data.type === WsIncomingType.Message) {
      broadcastToSidebar({ type: RuntimeMessageType.ChatMessage, content: data.content });
    } else if (data.type === WsIncomingType.Typing) {
      broadcastToSidebar({ type: RuntimeMessageType.Typing, active: data.active, content: data.content });
    } else if (data.type === WsIncomingType.ToolRequest) {
      handleToolRequest(data);
    } else if (data.type === WsIn.ThoughtsResponse) {
      broadcastToSidebar({ type: RuntimeMessageType.ThoughtsResponse, thoughts: data.thoughts });
      const unnotified = data.thoughts.filter((t: { notified: boolean }) => !t.notified).length;
      broadcastToSidebar({ type: RuntimeMessageType.ThoughtCount, count: unnotified });
    } else if (data.type === WsIn.PreferencesResponse) {
      broadcastToSidebar({
        type: RuntimeMessageType.PreferencesResponse,
        valence: data.valence,
        preferences: data.preferences,
      });
    } else if (data.type === WsIn.ConfigResponse) {
      broadcastToSidebar({ type: RuntimeMessageType.ConfigResponse, params: data.params });
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

function sendRegister(): void {
  if (!ws || ws.readyState !== WebSocket.OPEN || !deviceLabel) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.Register, sender: deviceLabel }));
}

function sendHeartbeat(): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.Heartbeat }));
}

function requestThoughts(): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.ThoughtsRequest }));
}

function sendThoughtReaction(thoughtId: number, emoji: string): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.ThoughtReaction, thought_id: thoughtId, emoji }));
}

function requestPreferences(valence: string): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.PreferencesRequest, valence }));
}

function sendPreferenceAdd(valence: string, content: string): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.PreferenceAdd, valence, content }));
}

function sendPreferenceDelete(preferenceId: number): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.PreferenceDelete, preference_id: preferenceId }));
}

function requestConfig(): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.ConfigRequest }));
}

function sendConfigUpdate(key: string, value: string): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.ConfigUpdate, key, value }));
}

async function sendCapabilities(): Promise<void> {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const stored = await browser.storage.local.get(STORAGE_KEY_TOOL_USE);
  const enabled = (stored[STORAGE_KEY_TOOL_USE] as boolean) ?? false;
  ws.send(JSON.stringify({ type: WsOutgoingType.CapabilitiesUpdate, tool_use_enabled: enabled }));
}

async function setToolUse(enabled: boolean): Promise<void> {
  await browser.storage.local.set({ [STORAGE_KEY_TOOL_USE]: enabled });
  await sendCapabilities();
  broadcastToSidebar({ type: RuntimeMessageType.ToolUseState, enabled });
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

// Serialize permission prompts so concurrent tool requests don't clobber the dialog
let permissionQueue: Promise<void> = Promise.resolve();

async function checkOrPromptPermission(
  requestId: string, domain: string, url: string,
): Promise<void> {
  // Re-check inside the queue — a prior prompt may have already resolved this domain
  const permission = await checkDomainPermission(domain);
  if (permission === "allowed") return;
  if (permission === "blocked") {
    throw new Error(`Domain ${domain} is blocked by user`);
  }

  const allowed = await requestPermissionFromUser(requestId, domain, url);
  await storeDomainPermission(domain, allowed ? DP.Allowed : DP.Blocked);
  if (!allowed) {
    throw new Error(`User denied access to ${domain}`);
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
    // Queue the prompt so only one dialog shows at a time
    const prompt = permissionQueue.then(() => checkOrPromptPermission(requestId, domain, url));
    permissionQueue = prompt.catch(() => {});
    await prompt;
  }

  return await browseUrl(url);
}

// --- Sidebar state sync ---

browser.runtime.onConnect.addListener(async (port) => {
  if (port.name === "sidebar") {
    port.postMessage({ type: RuntimeMessageType.ConnectionState, state: connectionState });
    const stored = await browser.storage.local.get(STORAGE_KEY_TOOL_USE);
    const enabled = (stored[STORAGE_KEY_TOOL_USE] as boolean) ?? false;
    port.postMessage({ type: RuntimeMessageType.ToolUseState, enabled });
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

// --- Debug: test browse_url from background console ---
// Usage: debugBrowseUrl("https://example.com")

// @ts-expect-error -- exposed for console debugging
globalThis.debugBrowseUrl = (url: string): void => {
  browseUrl(url).then(
    (result) => {
      console.log(`[debug] ${result.length} chars`);
      console.log(result);
    },
    (err) => console.error("[debug] ERROR:", err),
  );
};

// --- Boot ---

init();
