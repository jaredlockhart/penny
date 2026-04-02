/**
 * browse_url tool — opens a URL in a hidden tab, extracts visible text, closes the tab.
 *
 * After the page reports "complete", polls extraction up to EXTRACT_MAX_RETRIES
 * times to handle JS-rendered content (e.g. Kagi search results, SPAs).
 */

import { TAB_LOAD_TIMEOUT_MS } from "../../protocol.js";

const EXTRACT_MAX_RETRIES = 10;
const EXTRACT_POLL_MS = 500;
const MIN_CONTENT_LENGTH = 200;

interface PageData {
  title: string;
  url: string;
  text: string;
  image: string;
  ready: boolean;
}

const MAX_TAB_ATTEMPTS = 3;

export async function browseUrl(url: string): Promise<BrowseResult> {
  for (let attempt = 1; attempt <= MAX_TAB_ATTEMPTS; attempt++) {
    console.log(`[browse_url] opening: ${url} (attempt ${attempt}/${MAX_TAB_ATTEMPTS})`);
    const tab = await openHiddenTab(url);
    try {
      await waitForTabLoad(tab.id!);
      const pageData = await pollForContent(tab.id!, url);
      return await formatResult(pageData);
    } catch (err) {
      console.warn(`[browse_url] attempt ${attempt} failed:`, err);
      if (attempt === MAX_TAB_ATTEMPTS) {
        return { text: `Failed to read ${url}: ${err}`, image: "" };
      }
    } finally {
      await closeTab(tab.id!);
    }
  }
  return { text: `Failed to read ${url}`, image: "" };
}

async function pollForContent(tabId: number, url: string): Promise<PageData> {
  // Baseline extraction — establishes initial content length for growth detection
  const baseline = await extractPageContent(tabId);
  let previousLength = baseline.text.trim().length;
  console.log(`[browse_url] ${url}: baseline ${previousLength} chars, ready=${baseline.ready}`);

  if (baseline.ready && previousLength >= MIN_CONTENT_LENGTH) {
    // Have content — poll once more to check for growth
    await new Promise((r) => setTimeout(r, EXTRACT_POLL_MS));
    const second = await extractPageContent(tabId);
    const secondLength = second.text.trim().length;
    if (secondLength < previousLength * 2) {
      console.log(
        `[browse_url] ${url}: settled at ${secondLength} chars (prev ${previousLength})`,
      );
      return second;
    }
    console.log(
      `[browse_url] ${url}: ${secondLength} chars (prev ${previousLength}), still growing`,
    );
    previousLength = secondLength;
  }

  for (let attempt = 1; attempt <= EXTRACT_MAX_RETRIES; attempt++) {
    await new Promise((r) => setTimeout(r, EXTRACT_POLL_MS));
    const data = await extractPageContent(tabId);
    const textLen = data.text.trim().length;

    if (!data.ready) {
      console.log(
        `[browse_url] ${url}: not ready, waiting (attempt ${attempt}/${EXTRACT_MAX_RETRIES})`,
      );
      continue;
    }

    if (textLen < MIN_CONTENT_LENGTH) {
      console.log(
        `[browse_url] ${url}: only ${textLen} chars, waiting (attempt ${attempt}/${EXTRACT_MAX_RETRIES})`,
      );
      previousLength = textLen;
      continue;
    }

    if (previousLength > 0 && textLen < previousLength * 2) {
      console.log(
        `[browse_url] ${url}: settled at ${textLen} chars (prev ${previousLength}, attempt ${attempt})`,
      );
      return data;
    }

    console.log(
      `[browse_url] ${url}: ${textLen} chars (prev ${previousLength}), still growing (attempt ${attempt}/${EXTRACT_MAX_RETRIES})`,
    );
    previousLength = textLen;
  }

  const final = await extractPageContent(tabId);
  if (!final.ready) {
    throw new Error(`${url}: page not ready after ${EXTRACT_MAX_RETRIES} retries`);
  }
  console.warn(
    `[browse_url] ${url}: settled on ${final.text.trim().length} chars after ${EXTRACT_MAX_RETRIES} retries`,
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

interface BrowseResult {
  text: string;
  image: string;
}

async function formatResult(data: PageData): Promise<BrowseResult> {
  const image = data.image ? await downloadImageAsDataUri(data.image) : "";
  console.log(`[browse_url] image: ${image ? `${image.length} chars` : "none"}`);
  return {
    text: `Title: ${data.title}\nURL: ${data.url}\n\n${data.text}`,
    image,
  };
}

async function downloadImageAsDataUri(url: string): Promise<string> {
  try {
    const resp = await fetch(url);
    if (!resp.ok) return "";
    const blob = await resp.blob();
    const buffer = await blob.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    let binary = "";
    for (const b of bytes) binary += String.fromCharCode(b);
    const b64 = btoa(binary);
    return `data:${blob.type};base64,${b64}`;
  } catch {
    console.warn("[browse_url] failed to download image:", url);
    return "";
  }
}
