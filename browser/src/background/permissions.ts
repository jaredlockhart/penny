/**
 * Domain permission management — checks and prompts for domain access.
 */

import {
  type DomainAllowlist,
  type DomainPermission,
  DomainPermission as DP,
  RuntimeMessageType,
  STORAGE_KEY_DOMAIN_ALLOWLIST,
} from "../protocol.js";

/** Result of a permission check: allowed, blocked, or needs user prompt. */
type PermissionResult = "allowed" | "blocked" | "unknown";

export function extractDomain(url: string): string {
  return new URL(url).hostname;
}

export async function checkDomainPermission(domain: string): Promise<PermissionResult> {
  const allowlist = await loadAllowlist();
  if (domain in allowlist) {
    return allowlist[domain] === DP.Allowed ? "allowed" : "blocked";
  }
  // Check parent domains (e.g., "www.example.com" matches "example.com")
  const parts = domain.split(".");
  for (let i = 1; i < parts.length - 1; i++) {
    const parent = parts.slice(i).join(".");
    if (parent in allowlist) {
      return allowlist[parent] === DP.Allowed ? "allowed" : "blocked";
    }
  }
  return "unknown";
}

export async function storeDomainPermission(
  domain: string,
  permission: DomainPermission,
): Promise<void> {
  const allowlist = await loadAllowlist();
  allowlist[domain] = permission;
  await browser.storage.local.set({ [STORAGE_KEY_DOMAIN_ALLOWLIST]: allowlist });
}

export async function requestPermissionFromUser(
  requestId: string,
  domain: string,
  url: string,
): Promise<boolean> {
  return new Promise((resolve) => {
    function handleResponse(message: { type: string; request_id?: string; allowed?: boolean }): void {
      if (
        message.type === RuntimeMessageType.PermissionResponse &&
        message.request_id === requestId
      ) {
        browser.runtime.onMessage.removeListener(handleResponse);
        resolve(message.allowed ?? false);
      }
    }

    browser.runtime.onMessage.addListener(handleResponse);

    // Ask the sidebar to show a permission dialog
    browser.runtime.sendMessage({
      type: RuntimeMessageType.PermissionRequest,
      request_id: requestId,
      domain,
      url,
    }).catch(() => {
      // Sidebar not open — deny by default
      browser.runtime.onMessage.removeListener(handleResponse);
      resolve(false);
    });
  });
}

async function loadAllowlist(): Promise<DomainAllowlist> {
  const stored = await browser.storage.local.get(STORAGE_KEY_DOMAIN_ALLOWLIST);
  return (stored[STORAGE_KEY_DOMAIN_ALLOWLIST] as DomainAllowlist) ?? {};
}
