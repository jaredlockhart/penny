/**
 * WebSocket protocol types shared between sidebar and background scripts.
 * Mirrors penny/penny/channels/browser/models.py.
 */

// --- Connection ---

export const SERVER_URL = "ws://localhost:9090";
export const RECONNECT_DELAY_MS = 3000;

export type ConnectionState = "connected" | "disconnected" | "reconnecting";
export const ConnectionState = {
  Connected: "connected",
  Disconnected: "disconnected",
  Reconnecting: "reconnecting",
} as const satisfies Record<string, ConnectionState>;

// --- WebSocket: outgoing (browser → server) ---

export type WsOutgoingType =
  | "message"
  | "tool_response"
  | "thoughts_request"
  | "thought_reaction"
  | "preferences_request"
  | "preference_add"
  | "preference_delete"
  | "heartbeat"
  | "config_request"
  | "config_update"
  | "register";
export const WsOutgoingType = {
  Message: "message",
  ToolResponse: "tool_response",
  ThoughtsRequest: "thoughts_request",
  ThoughtReaction: "thought_reaction",
  PreferencesRequest: "preferences_request",
  PreferenceAdd: "preference_add",
  PreferenceDelete: "preference_delete",
  Heartbeat: "heartbeat",
  ConfigRequest: "config_request",
  ConfigUpdate: "config_update",
  Register: "register",
} as const satisfies Record<string, WsOutgoingType>;

export interface WsOutgoingThoughtReaction {
  type: typeof WsOutgoingType.ThoughtReaction;
  thought_id: number;
  emoji: string;
}

export interface WsOutgoingMessage {
  type: typeof WsOutgoingType.Message;
  content: string;
  sender: string;
  page_context?: PageContext;
}

export interface WsOutgoingToolResponse {
  type: typeof WsOutgoingType.ToolResponse;
  request_id: string;
  result?: string;
  error?: string;
}

export interface WsOutgoingPreferencesRequest {
  type: typeof WsOutgoingType.PreferencesRequest;
  valence: string;
}

export interface WsOutgoingPreferenceAdd {
  type: typeof WsOutgoingType.PreferenceAdd;
  valence: string;
  content: string;
}

export interface WsOutgoingPreferenceDelete {
  type: typeof WsOutgoingType.PreferenceDelete;
  preference_id: number;
}

export interface WsOutgoingHeartbeat {
  type: typeof WsOutgoingType.Heartbeat;
}

export type WsOutgoing =
  | WsOutgoingMessage
  | WsOutgoingToolResponse
  | WsOutgoingPreferencesRequest
  | WsOutgoingPreferenceAdd
  | WsOutgoingPreferenceDelete
  | WsOutgoingHeartbeat;

// --- WebSocket: incoming (server → browser) ---

export type WsIncomingType =
  | "message"
  | "typing"
  | "status"
  | "tool_request"
  | "thoughts_response"
  | "preferences_response"
  | "config_response";
export const WsIncomingType = {
  Message: "message",
  Typing: "typing",
  Status: "status",
  ToolRequest: "tool_request",
  ThoughtsResponse: "thoughts_response",
  PreferencesResponse: "preferences_response",
  ConfigResponse: "config_response",
} as const satisfies Record<string, WsIncomingType>;

export interface WsIncomingMessagePayload {
  type: typeof WsIncomingType.Message;
  content: string;
}

export interface WsIncomingTypingPayload {
  type: typeof WsIncomingType.Typing;
  active: boolean;
  content?: string;
}

export interface WsIncomingStatusPayload {
  type: typeof WsIncomingType.Status;
  connected: boolean;
}

export interface WsIncomingToolRequestPayload {
  type: typeof WsIncomingType.ToolRequest;
  request_id: string;
  tool: string;
  arguments: Record<string, unknown>;
}

export interface ThoughtCard {
  id: number;
  title: string;
  content: string;
  image_url: string;
  created_at: string;
  notified: boolean;
  seed_topic: string;
}

export interface WsIncomingThoughtsPayload {
  type: typeof WsIncomingType.ThoughtsResponse;
  thoughts: ThoughtCard[];
}

export interface PreferenceItem {
  id: number;
  content: string;
  mention_count: number;
}

export interface WsIncomingPreferencesPayload {
  type: typeof WsIncomingType.PreferencesResponse;
  valence: string;
  preferences: PreferenceItem[];
}

export interface RuntimeConfigParam {
  key: string;
  value: string;
  default: string;
  description: string;
  type: "int" | "float" | "str";
  group: string;
}

export interface WsIncomingConfigPayload {
  type: typeof WsIncomingType.ConfigResponse;
  params: RuntimeConfigParam[];
}

export type WsIncomingPayload =
  | WsIncomingMessagePayload
  | WsIncomingTypingPayload
  | WsIncomingStatusPayload
  | WsIncomingToolRequestPayload
  | WsIncomingThoughtsPayload
  | WsIncomingPreferencesPayload
  | WsIncomingConfigPayload;

// --- Runtime messages (sidebar ↔ background) ---

export type RuntimeMessageType =
  | "send_chat"
  | "chat_message"
  | "typing"
  | "connection_state"
  | "permission_request"
  | "permission_response"
  | "page_info"
  | "thoughts_request"
  | "thoughts_response"
  | "thought_reaction"
  | "thought_count"
  | "preferences_request"
  | "preferences_response"
  | "preference_add"
  | "preference_delete"
  | "config_request"
  | "config_response"
  | "config_update";

export const RuntimeMessageType = {
  SendChat: "send_chat",
  ChatMessage: "chat_message",
  Typing: "typing",
  ConnectionState: "connection_state",
  PermissionRequest: "permission_request",
  PermissionResponse: "permission_response",
  PageInfo: "page_info",
  ThoughtsRequest: "thoughts_request",
  ThoughtsResponse: "thoughts_response",
  ThoughtReaction: "thought_reaction",
  ThoughtCount: "thought_count",
  PreferencesRequest: "preferences_request",
  PreferencesResponse: "preferences_response",
  PreferenceAdd: "preference_add",
  PreferenceDelete: "preference_delete",
  ConfigRequest: "config_request",
  ConfigResponse: "config_response",
  ConfigUpdate: "config_update",
} as const satisfies Record<string, RuntimeMessageType>;

/** Sidebar → background: user typed a chat message */
export interface RuntimeSendChat {
  type: typeof RuntimeMessageType.SendChat;
  content: string;
  include_page: boolean;
}

/** Background → sidebar: incoming message from Penny */
export interface RuntimeChatMessage {
  type: typeof RuntimeMessageType.ChatMessage;
  content: string;
}

/** Background → sidebar: typing indicator */
export interface RuntimeTyping {
  type: typeof RuntimeMessageType.Typing;
  active: boolean;
  content?: string;
}

/** Background → sidebar: connection state changed */
export interface RuntimeConnectionState {
  type: typeof RuntimeMessageType.ConnectionState;
  state: ConnectionState;
}

/** Background → sidebar: ask user to allow/deny a domain */
export interface RuntimePermissionRequest {
  type: typeof RuntimeMessageType.PermissionRequest;
  request_id: string;
  domain: string;
  url: string;
}

/** Sidebar → background: user's permission decision */
export interface RuntimePermissionResponse {
  type: typeof RuntimeMessageType.PermissionResponse;
  request_id: string;
  allowed: boolean;
}

/** Background → sidebar: current page info for the context toggle */
export interface RuntimePageInfo {
  type: typeof RuntimeMessageType.PageInfo;
  title: string;
  url: string;
  favicon: string;
  image: string;      // og:image or similar meta image
  available: boolean;  // false if extraction failed or on a privileged page
}

/** Feed page → background: request thoughts */
export interface RuntimeThoughtsRequest {
  type: typeof RuntimeMessageType.ThoughtsRequest;
}

/** Background → feed page: thoughts data */
export interface RuntimeThoughtsResponse {
  type: typeof RuntimeMessageType.ThoughtsResponse;
  thoughts: ThoughtCard[];
}

/** Feed page → background: react to a thought */
export interface RuntimeThoughtReaction {
  type: typeof RuntimeMessageType.ThoughtReaction;
  thought_id: number;
  emoji: string;
}

/** Background → sidebar: unnotified thought count */
export interface RuntimeThoughtCount {
  type: typeof RuntimeMessageType.ThoughtCount;
  count: number;
}

/** Sidebar → background: request preferences by valence */
export interface RuntimePreferencesRequest {
  type: typeof RuntimeMessageType.PreferencesRequest;
  valence: string;
}

/** Background → sidebar: preferences list for a valence */
export interface RuntimePreferencesResponse {
  type: typeof RuntimeMessageType.PreferencesResponse;
  valence: string;
  preferences: PreferenceItem[];
}

/** Sidebar → background: add a preference */
export interface RuntimePreferenceAdd {
  type: typeof RuntimeMessageType.PreferenceAdd;
  valence: string;
  content: string;
}

/** Sidebar → background: delete a preference */
export interface RuntimePreferenceDelete {
  type: typeof RuntimeMessageType.PreferenceDelete;
  preference_id: number;
}

/** Sidebar → background: request all config params */
export interface RuntimeConfigRequest {
  type: typeof RuntimeMessageType.ConfigRequest;
}

/** Background → sidebar: all config params with current values */
export interface RuntimeConfigResponse {
  type: typeof RuntimeMessageType.ConfigResponse;
  params: RuntimeConfigParam[];
}

/** Sidebar → background: update one config param */
export interface RuntimeConfigUpdate {
  type: typeof RuntimeMessageType.ConfigUpdate;
  key: string;
  value: string;
}

export type RuntimeMessage =
  | RuntimeSendChat
  | RuntimeChatMessage
  | RuntimeTyping
  | RuntimeConnectionState
  | RuntimePermissionRequest
  | RuntimePermissionResponse
  | RuntimePageInfo
  | RuntimeThoughtsRequest
  | RuntimeThoughtsResponse
  | RuntimeThoughtReaction
  | RuntimeThoughtCount
  | RuntimePreferencesRequest
  | RuntimePreferencesResponse
  | RuntimePreferenceAdd
  | RuntimePreferenceDelete
  | RuntimeConfigRequest
  | RuntimeConfigResponse
  | RuntimeConfigUpdate;

// --- Domain permissions ---

export type DomainPermission = "allowed" | "blocked";
export const DomainPermission = {
  Allowed: "allowed",
  Blocked: "blocked",
} as const satisfies Record<string, DomainPermission>;

/** Map of domain → permission stored in browser.storage.local */
export type DomainAllowlist = Record<string, DomainPermission>;

// --- Page context ---

export interface PageContext {
  title: string;
  url: string;
  text: string;
  image: string;
}

export const MAX_PAGE_CONTEXT_CHARS = 5_000;

// --- Tool constants ---

export const THOUGHTS_POLL_INTERVAL_MS = 300_000;
export const TOOL_TIMEOUT_MS = 60_000;
export const TAB_LOAD_TIMEOUT_MS = 60_000;
export const MAX_EXTRACTED_CHARS = 50_000;

// --- Chat UI ---

export type MessageSender = "user" | "penny";
export const MessageSender = {
  User: "user",
  Penny: "penny",
} as const satisfies Record<string, MessageSender>;

// --- Chat history ---

export interface StoredMessage {
  text: string;
  sender: MessageSender;
}

export const MAX_STORED_MESSAGES = 200;

// --- Storage keys ---

export const STORAGE_KEY_DEVICE_LABEL = "deviceLabel";
export const STORAGE_KEY_CHAT_HISTORY = "chatHistory";
export const STORAGE_KEY_DOMAIN_ALLOWLIST = "domainAllowlist";

// --- UI constants ---

export const TEXTAREA_LINE_HEIGHT = 20;
export const TEXTAREA_MAX_ROWS = 4;
export const TYPING_INDICATOR_TEXT = "Penny is thinking...";
