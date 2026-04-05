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

/** Domain-specific readiness locators. For JS-rendered pages, Defuddle may
 *  extract too early and get page chrome instead of content. These selectors
 *  gate extraction — if the selector isn't present yet, we return ready=false
 *  so pollForContent retries until the real content has rendered. */
const READINESS_LOCATORS: [match: (hostname: string) => boolean, selector: string][] = [
  [(h) => h.includes("kagi.com"), ".search-result"],
];

function findReadinessSelector(): string | null {
  for (const [match, selector] of READINESS_LOCATORS) {
    if (match(location.hostname)) return selector;
  }
  return null;
}

function extractXml(): string | null {
  const contentType = document.contentType;
  if (contentType && (contentType.includes("xml") || contentType.includes("rss"))) {
    const serializer = new XMLSerializer();
    return serializer.serializeToString(document);
  }
  return null;
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
  // Kagi search results: grab first image from the inline image results
  const kagiImage = document.querySelector("._0_image_item");
  if (kagiImage) {
    const url = kagiImage.getAttribute("data-content_url");
    if (url) return url;
  }
  return "";
}

function extract(): PageData {
  const readinessSelector = findReadinessSelector();
  if (readinessSelector && !document.querySelector(readinessSelector)) {
    return { title: document.title, url: location.href, text: "", image: "", ready: false };
  }

  const text = extractXml() ?? extractWithDefuddle() ?? "Failed to extract page content";

  return {
    title: document.title,
    url: location.href,
    text: text.slice(0, MAX_CHARS),
    image: extractMetaImage(),
    ready: true,
  };
}

extract();
