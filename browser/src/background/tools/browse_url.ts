/**
 * browse_url tool — opens a URL in a hidden tab, extracts visible text, closes the tab.
 */

import { TAB_LOAD_TIMEOUT_MS } from "../../protocol.js";

const BROWSE_MAX_RETRIES = 3;

interface PageData {
  title: string;
  url: string;
  text: string;
}

export async function browseUrl(url: string): Promise<string> {
  for (let attempt = 1; attempt <= BROWSE_MAX_RETRIES; attempt++) {
    console.log(`[browse_url] attempt ${attempt}/${BROWSE_MAX_RETRIES}: ${url}`);
    const tab = await openHiddenTab(url);
    try {
      await waitForTabLoad(tab.id!);
      console.log(`[browse_url] page complete, extracting content`);
      const pageData = await extractPageContent(tab.id!);
      const textLen = pageData.text.trim().length;
      if (textLen > 0) {
        console.log(`[browse_url] extracted ${textLen} chars`);
        return formatResult(pageData);
      }
      console.warn(`[browse_url] empty content on attempt ${attempt}`);
    } catch (err) {
      console.error(`[browse_url] attempt ${attempt} failed:`, err);
    } finally {
      await closeTab(tab.id!);
    }
  }
  console.error(`[browse_url] gave up after ${BROWSE_MAX_RETRIES} attempts: ${url}`);
  return `No content extracted from ${url} after ${BROWSE_MAX_RETRIES} attempts`;
}

async function openHiddenTab(url: string): Promise<browser.tabs.Tab> {
  const tab = await browser.tabs.create({ url, active: false });
  if (!tab.id) {
    throw new Error("Failed to create tab");
  }
  try {
    await browser.tabs.hide(tab.id);
  } catch {
    // tabHide may not be available — tab stays visible but still works
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
