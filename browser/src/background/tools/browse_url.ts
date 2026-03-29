/**
 * browse_url tool — opens a URL in a hidden tab, extracts visible text, closes the tab.
 *
 * After the page reports "complete", polls extraction up to EXTRACT_MAX_RETRIES
 * times to handle JS-rendered content (e.g. Kagi search results, SPAs).
 */

import { TAB_LOAD_TIMEOUT_MS } from "../../protocol.js";

const EXTRACT_MAX_RETRIES = 10;
const EXTRACT_POLL_MS = 1000;
const MIN_CONTENT_LENGTH = 200;

interface PageData {
  title: string;
  url: string;
  text: string;
  ready: boolean;
}

const MAX_TAB_ATTEMPTS = 3;

export async function browseUrl(url: string): Promise<string> {
  for (let attempt = 1; attempt <= MAX_TAB_ATTEMPTS; attempt++) {
    console.log(`[browse_url] opening: ${url} (attempt ${attempt}/${MAX_TAB_ATTEMPTS})`);
    const tab = await openHiddenTab(url);
    try {
      await waitForTabLoad(tab.id!);
      const pageData = await pollForContent(tab.id!);
      return formatResult(pageData);
    } catch (err) {
      console.warn(`[browse_url] attempt ${attempt} failed:`, err);
      if (attempt === MAX_TAB_ATTEMPTS) {
        return `Failed to read ${url}: ${err}`;
      }
    } finally {
      await closeTab(tab.id!);
    }
  }
  return `Failed to read ${url}`;
}

async function pollForContent(tabId: number): Promise<PageData> {
  for (let attempt = 1; attempt <= EXTRACT_MAX_RETRIES; attempt++) {
    const data = await extractPageContent(tabId);
    const textLen = data.text.trim().length;
    if (data.ready && textLen >= MIN_CONTENT_LENGTH) {
      console.log(`[browse_url] extracted ${textLen} chars (attempt ${attempt})`);
      return data;
    }
    const reason = !data.ready ? "not ready" : `only ${textLen} chars`;
    console.log(
      `[browse_url] ${reason}, waiting (attempt ${attempt}/${EXTRACT_MAX_RETRIES})`,
    );
    await new Promise((r) => setTimeout(r, EXTRACT_POLL_MS));
  }
  const final = await extractPageContent(tabId);
  if (!final.ready) {
    throw new Error(`Page not ready after ${EXTRACT_MAX_RETRIES} retries`);
  }
  console.warn(
    `[browse_url] settled on ${final.text.trim().length} chars after ${EXTRACT_MAX_RETRIES} retries`,
  );
  return final;
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
