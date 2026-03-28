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

// --- Outgoing messages (browser → server) ---

export type OutgoingType = "message";
export const OutgoingType = {
  Message: "message",
} as const satisfies Record<string, OutgoingType>;

export interface OutgoingMessage {
  type: typeof OutgoingType.Message;
  content: string;
  sender: string;
}

// --- Incoming messages (server → browser) ---

export type IncomingType = "message" | "typing" | "status";
export const IncomingType = {
  Message: "message",
  Typing: "typing",
  Status: "status",
} as const satisfies Record<string, IncomingType>;

export interface IncomingMessagePayload {
  type: typeof IncomingType.Message;
  content: string;
}

export interface IncomingTypingPayload {
  type: typeof IncomingType.Typing;
  active: boolean;
}

export interface IncomingStatusPayload {
  type: typeof IncomingType.Status;
  connected: boolean;
}

export type IncomingPayload =
  | IncomingMessagePayload
  | IncomingTypingPayload
  | IncomingStatusPayload;

// --- Chat UI ---

export type MessageSender = "user" | "penny";
export const MessageSender = {
  User: "user",
  Penny: "penny",
} as const satisfies Record<string, MessageSender>;

// --- Storage keys ---

export const STORAGE_KEY_DEVICE_LABEL = "deviceLabel";

// --- UI constants ---

export const TEXTAREA_LINE_HEIGHT = 20;
export const TEXTAREA_MAX_ROWS = 4;
export const TYPING_INDICATOR_TEXT = "Penny is thinking...";
