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

export type WsOutgoingType = "message";
export const WsOutgoingType = {
  Message: "message",
} as const satisfies Record<string, WsOutgoingType>;

export interface WsOutgoingMessage {
  type: typeof WsOutgoingType.Message;
  content: string;
  sender: string;
}

// --- WebSocket: incoming (server → browser) ---

export type WsIncomingType = "message" | "typing" | "status";
export const WsIncomingType = {
  Message: "message",
  Typing: "typing",
  Status: "status",
} as const satisfies Record<string, WsIncomingType>;

export interface WsIncomingMessagePayload {
  type: typeof WsIncomingType.Message;
  content: string;
}

export interface WsIncomingTypingPayload {
  type: typeof WsIncomingType.Typing;
  active: boolean;
}

export interface WsIncomingStatusPayload {
  type: typeof WsIncomingType.Status;
  connected: boolean;
}

export type WsIncomingPayload =
  | WsIncomingMessagePayload
  | WsIncomingTypingPayload
  | WsIncomingStatusPayload;

// --- Runtime messages (sidebar ↔ background) ---

export type RuntimeMessageType = "send_chat" | "chat_message" | "typing" | "connection_state";
export const RuntimeMessageType = {
  SendChat: "send_chat",
  ChatMessage: "chat_message",
  Typing: "typing",
  ConnectionState: "connection_state",
} as const satisfies Record<string, RuntimeMessageType>;

/** Sidebar → background: user typed a chat message */
export interface RuntimeSendChat {
  type: typeof RuntimeMessageType.SendChat;
  content: string;
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
}

/** Background → sidebar: connection state changed */
export interface RuntimeConnectionState {
  type: typeof RuntimeMessageType.ConnectionState;
  state: ConnectionState;
}

export type RuntimeMessage =
  | RuntimeSendChat
  | RuntimeChatMessage
  | RuntimeTyping
  | RuntimeConnectionState;

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

// --- UI constants ---

export const TEXTAREA_LINE_HEIGHT = 20;
export const TEXTAREA_MAX_ROWS = 4;
export const TYPING_INDICATOR_TEXT = "Penny is thinking...";
