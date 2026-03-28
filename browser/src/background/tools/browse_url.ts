/**
 * browse_url tool — opens a URL in a hidden tab, extracts visible text, closes the tab.
 */

import { TAB_LOAD_TIMEOUT_MS } from "../../protocol.js";

interface PageData {
  title: string;
  url: string;
  text: string;
}

export async function browseUrl(url: string): Promise<string> {
  const tab = await openHiddenTab(url);
  try {
    await waitForTabLoad(tab.id!);
    const pageData = await extractPageContent(tab.id!);
    return formatResult(pageData);
  } finally {
    await closeTab(tab.id!);
  }
}

async function openHiddenTab(url: string): Promise<browser.tabs.Tab> {
  const tab = await browser.tabs.create({ url, active: false });
  if (!tab.id) {
    throw new Error("Failed to create tab");
  }
  return tab;
}

function waitForTabLoad(tabId: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      browser.tabs.onUpdated.removeListener(listener);
      reject(new Error(`Tab load timed out after ${TAB_LOAD_TIMEOUT_MS}ms`));
    }, TAB_LOAD_TIMEOUT_MS);

    function listener(
      updatedTabId: number,
      changeInfo: browser.tabs._OnUpdatedChangeInfo,
    ): void {
      if (updatedTabId === tabId && changeInfo.status === "complete") {
        clearTimeout(timeout);
        browser.tabs.onUpdated.removeListener(listener);
        resolve();
      }
    }

    browser.tabs.onUpdated.addListener(listener);
  });
}

async function extractPageContent(tabId: number): Promise<PageData> {
  const results = await browser.tabs.executeScript(tabId, {
    file: "/dist/content/extract_text.js",
    runAt: "document_idle",
  });

  if (!results || !results[0]) {
    throw new Error("Content script returned no results");
  }

  return results[0] as PageData;
}

async function closeTab(tabId: number): Promise<void> {
  try {
    await browser.tabs.remove(tabId);
  } catch {
    // Tab may already be closed
  }
}

function formatResult(data: PageData): string {
  return `Title: ${data.title}\nURL: ${data.url}\n\n${data.text}`;
}
