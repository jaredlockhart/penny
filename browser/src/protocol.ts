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
  | "register"
  | "capabilities_update"
  | "domain_update"
  | "domain_delete"
  | "permission_decision"
  | "schedules_request"
  | "schedule_add"
  | "schedule_update"
  | "schedule_delete"
  | "prompt_logs_request";
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
  CapabilitiesUpdate: "capabilities_update",
  DomainUpdate: "domain_update",
  DomainDelete: "domain_delete",
  PermissionDecision: "permission_decision",
  SchedulesRequest: "schedules_request",
  ScheduleAdd: "schedule_add",
  ScheduleUpdate: "schedule_update",
  ScheduleDelete: "schedule_delete",
  PromptLogsRequest: "prompt_logs_request",
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

export interface WsOutgoingCapabilitiesUpdate {
  type: typeof WsOutgoingType.CapabilitiesUpdate;
  tool_use_enabled: boolean;
}

export interface WsOutgoingSchedulesRequest {
  type: typeof WsOutgoingType.SchedulesRequest;
}

export interface WsOutgoingScheduleAdd {
  type: typeof WsOutgoingType.ScheduleAdd;
  command: string;
}

export interface WsOutgoingScheduleUpdate {
  type: typeof WsOutgoingType.ScheduleUpdate;
  schedule_id: number;
  prompt_text: string;
}

export interface WsOutgoingScheduleDelete {
  type: typeof WsOutgoingType.ScheduleDelete;
  schedule_id: number;
}

export type WsOutgoing =
  | WsOutgoingMessage
  | WsOutgoingToolResponse
  | WsOutgoingPreferencesRequest
  | WsOutgoingPreferenceAdd
  | WsOutgoingPreferenceDelete
  | WsOutgoingHeartbeat
  | WsOutgoingCapabilitiesUpdate
  | WsOutgoingSchedulesRequest
  | WsOutgoingScheduleAdd
  | WsOutgoingScheduleUpdate
  | WsOutgoingScheduleDelete;

// --- WebSocket: incoming (server → browser) ---

export type WsIncomingType =
  | "message"
  | "typing"
  | "status"
  | "tool_request"
  | "thoughts_response"
  | "preferences_response"
  | "config_response"
  | "domain_permissions_sync"
  | "permission_prompt"
  | "permission_dismiss"
  | "schedules_response"
  | "prompt_logs_response"
  | "prompt_log_update"
  | "run_outcome_update";
export const WsIncomingType = {
  Message: "message",
  Typing: "typing",
  Status: "status",
  ToolRequest: "tool_request",
  ThoughtsResponse: "thoughts_response",
  PreferencesResponse: "preferences_response",
  ConfigResponse: "config_response",
  DomainPermissionsSync: "domain_permissions_sync",
  PermissionPrompt: "permission_prompt",
  PermissionDismiss: "permission_dismiss",
  SchedulesResponse: "schedules_response",
  PromptLogsResponse: "prompt_logs_response",
  PromptLogUpdate: "prompt_log_update",
  RunOutcomeUpdate: "run_outcome_update",
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
  image: string;
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
  source: string;
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

export interface DomainPermissionEntry {
  domain: string;
  permission: DomainPermission;
}

export interface WsIncomingDomainPermissionsPayload {
  type: typeof WsIncomingType.DomainPermissionsSync;
  permissions: DomainPermissionEntry[];
}

export interface WsIncomingPermissionPromptPayload {
  type: typeof WsIncomingType.PermissionPrompt;
  request_id: string;
  domain: string;
  url: string;
}

export interface WsIncomingPermissionDismissPayload {
  type: typeof WsIncomingType.PermissionDismiss;
  request_id: string;
}

export interface ScheduleItem {
  id: number;
  timing_description: string;
  prompt_text: string;
  cron_expression: string;
}

export interface WsIncomingSchedulesPayload {
  type: typeof WsIncomingType.SchedulesResponse;
  schedules: ScheduleItem[];
  error: string | null;
}

export interface PromptLogEntry {
  id: number;
  timestamp: string;
  model: string;
  agent_name: string;
  prompt_type: string;
  duration_ms: number;
  input_tokens: number;
  output_tokens: number;
  messages: Record<string, unknown>[];
  response: Record<string, unknown>;
  thinking: string;
  has_tools: boolean;
}

export interface PromptLogRun {
  run_id: string;
  agent_name: string;
  prompt_count: number;
  started_at: string;
  ended_at: string;
  total_duration_ms: number;
  total_input_tokens: number;
  total_output_tokens: number;
  run_outcome: string | null;
  prompts: PromptLogEntry[];
}

export interface WsIncomingPromptLogsPayload {
  type: typeof WsIncomingType.PromptLogsResponse;
  agent_names: string[];
  runs: PromptLogRun[];
  has_more: boolean;
}

export interface WsIncomingPromptLogUpdatePayload {
  type: typeof WsIncomingType.PromptLogUpdate;
  prompt: PromptLogEntry & { run_id: string };
}

export interface WsIncomingRunOutcomePayload {
  type: typeof WsIncomingType.RunOutcomeUpdate;
  run_id: string;
  outcome: string;
}

export type WsIncomingPayload =
  | WsIncomingMessagePayload
  | WsIncomingTypingPayload
  | WsIncomingStatusPayload
  | WsIncomingToolRequestPayload
  | WsIncomingThoughtsPayload
  | WsIncomingPreferencesPayload
  | WsIncomingConfigPayload
  | WsIncomingDomainPermissionsPayload
  | WsIncomingPermissionPromptPayload
  | WsIncomingPermissionDismissPayload
  | WsIncomingSchedulesPayload
  | WsIncomingPromptLogsPayload
  | WsIncomingPromptLogUpdatePayload
  | WsIncomingRunOutcomePayload;

// --- Runtime messages (sidebar ↔ background) ---

export type RuntimeMessageType =
  | "send_chat"
  | "chat_message"
  | "typing"
  | "connection_state"
  | "permission_request"
  | "permission_response"
  | "permission_dismiss"
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
  | "config_update"
  | "tool_use_toggle"
  | "tool_use_state"
  | "domain_update"
  | "domain_delete"
  | "domain_permissions_sync"
  | "schedules_request"
  | "schedules_response"
  | "schedule_add"
  | "schedule_update"
  | "schedule_delete"
  | "prompt_logs_request"
  | "prompt_logs_response"
  | "prompt_log_update"
  | "run_outcome_update";

export const RuntimeMessageType = {
  SendChat: "send_chat",
  ChatMessage: "chat_message",
  Typing: "typing",
  ConnectionState: "connection_state",
  PermissionRequest: "permission_request",
  PermissionResponse: "permission_response",
  PermissionDismiss: "permission_dismiss",
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
  ToolUseToggle: "tool_use_toggle",
  ToolUseState: "tool_use_state",
  DomainUpdate: "domain_update",
  DomainDelete: "domain_delete",
  DomainPermissionsSync: "domain_permissions_sync",
  SchedulesRequest: "schedules_request",
  SchedulesResponse: "schedules_response",
  ScheduleAdd: "schedule_add",
  ScheduleUpdate: "schedule_update",
  ScheduleDelete: "schedule_delete",
  PromptLogsRequest: "prompt_logs_request",
  PromptLogsResponse: "prompt_logs_response",
  PromptLogUpdate: "prompt_log_update",
  RunOutcomeUpdate: "run_outcome_update",
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

/** Background → sidebar: dismiss any open permission dialog */
export interface RuntimePermissionDismiss {
  type: typeof RuntimeMessageType.PermissionDismiss;
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

/** Background → page: thoughts data */
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

/** Sidebar → background: toggle tool-use capability */
export interface RuntimeToolUseToggle {
  type: typeof RuntimeMessageType.ToolUseToggle;
  enabled: boolean;
}

/** Background → sidebar: current tool-use state */
export interface RuntimeToolUseState {
  type: typeof RuntimeMessageType.ToolUseState;
  enabled: boolean;
}

/** Sidebar → background: add or update a domain permission */
export interface RuntimeDomainUpdate {
  type: typeof RuntimeMessageType.DomainUpdate;
  domain: string;
  permission: DomainPermission;
}

/** Sidebar → background: delete a domain permission */
export interface RuntimeDomainDelete {
  type: typeof RuntimeMessageType.DomainDelete;
  domain: string;
}

/** Background → sidebar: full domain permissions list */
export interface RuntimeDomainPermissionsSync {
  type: typeof RuntimeMessageType.DomainPermissionsSync;
  permissions: DomainPermissionEntry[];
}

/** Sidebar → background: request all schedules */
export interface RuntimeSchedulesRequest {
  type: typeof RuntimeMessageType.SchedulesRequest;
}

/** Background → sidebar: schedules list */
export interface RuntimeSchedulesResponse {
  type: typeof RuntimeMessageType.SchedulesResponse;
  schedules: ScheduleItem[];
  error: string | null;
}

/** Sidebar → background: add a new schedule */
export interface RuntimeScheduleAdd {
  type: typeof RuntimeMessageType.ScheduleAdd;
  command: string;
}

/** Sidebar → background: update a schedule's prompt text */
export interface RuntimeScheduleUpdate {
  type: typeof RuntimeMessageType.ScheduleUpdate;
  schedule_id: number;
  prompt_text: string;
}

/** Sidebar → background: delete a schedule */
export interface RuntimeScheduleDelete {
  type: typeof RuntimeMessageType.ScheduleDelete;
  schedule_id: number;
}

/** Prompts page → background: request prompt logs */
export interface RuntimePromptLogsRequest {
  type: typeof RuntimeMessageType.PromptLogsRequest;
  agent_name?: string;
  offset?: number;
}

/** Background → prompts page: prompt logs data */
export interface RuntimePromptLogsResponse {
  type: typeof RuntimeMessageType.PromptLogsResponse;
  agent_names: string[];
  runs: PromptLogRun[];
  has_more: boolean;
}

/** Background → prompts page: single prompt logged in real time */
export interface RuntimePromptLogUpdate {
  type: typeof RuntimeMessageType.PromptLogUpdate;
  prompt: PromptLogEntry & { run_id: string };
}

/** Background → prompts page: run outcome set (stored/discarded) */
export interface RuntimeRunOutcomeUpdate {
  type: typeof RuntimeMessageType.RunOutcomeUpdate;
  run_id: string;
  outcome: string;
}

export type RuntimeMessage =
  | RuntimeSendChat
  | RuntimeChatMessage
  | RuntimeTyping
  | RuntimeConnectionState
  | RuntimePermissionRequest
  | RuntimePermissionResponse
  | RuntimePermissionDismiss
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
  | RuntimeConfigUpdate
  | RuntimeToolUseToggle
  | RuntimeToolUseState
  | RuntimeDomainUpdate
  | RuntimeDomainDelete
  | RuntimeDomainPermissionsSync
  | RuntimeSchedulesRequest
  | RuntimeSchedulesResponse
  | RuntimeScheduleAdd
  | RuntimeScheduleUpdate
  | RuntimeScheduleDelete
  | RuntimePromptLogsRequest
  | RuntimePromptLogsResponse
  | RuntimePromptLogUpdate
  | RuntimeRunOutcomeUpdate;

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
export const STORAGE_KEY_TOOL_USE = "toolUseEnabled";

// --- UI constants ---

export const TEXTAREA_LINE_HEIGHT = 20;
export const TEXTAREA_MAX_ROWS = 4;
export const TYPING_INDICATOR_TEXT = "Penny is thinking...";
