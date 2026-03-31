/**
 * Content script — extracts main content from the current page.
 * Uses Defuddle for article extraction, then Turndown for HTML → Markdown.
 * Bundled with esbuild (not compiled by tsc) since content scripts can't use imports.
 */

import Defuddle from "defuddle";
import TurndownService from "turndown";

const turndown = new TurndownService({ headingStyle: "atx" });

const MIN_CONTENT_LENGTH = 200;
const MAX_CHARS = 50_000;

interface PageData {
  title: string;
  url: string;
  text: string;
  image: string;
  ready: boolean;
}

function extractWithDefuddle(): string | null {
  const clone = document.cloneNode(true) as Document;
  const result = new Defuddle(clone, { url: location.href }).parse();
  if (!result.content) return null;
  const text = turndown.turndown(result.content);
  if (text && text.length >= MIN_CONTENT_LENGTH) {
    return text;
  }
  return null;
}


function extractMetaImage(): string {
  const selectors = [
    'meta[property="og:image"]',
    'meta[name="twitter:image"]',
    'meta[property="og:image:url"]',
  ];
  for (const selector of selectors) {
    const el = document.querySelector(selector);
    const content = el?.getAttribute("content");
    if (content) return content;
  }
  return "";
}

function extract(): PageData {
  const text = extractWithDefuddle() ?? "Failed to extract page content";

  return {
    title: document.title,
    url: location.href,
    text: text.slice(0, MAX_CHARS),
    image: extractMetaImage(),
    ready: true,
  };
}

extract();
