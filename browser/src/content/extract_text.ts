/**
 * Content script — extracts main content from the current page using Defuddle.
 * Falls back to CSS heuristics if Defuddle returns insufficient content.
 * Bundled with esbuild (not compiled by tsc) since content scripts can't use imports.
 */

import { Defuddle } from "defuddle";

const MIN_CONTENT_LENGTH = 200;
const MAX_CHARS = 50_000;

interface PageData {
  title: string;
  url: string;
  text: string;
}

function extractWithDefuddle(): string | null {
  try {
    const clone = document.cloneNode(true) as Document;
    const result = new Defuddle(clone).parse();
    const text = result.content
      ? stripHtmlTags(result.content)
      : null;
    if (text && text.length >= MIN_CONTENT_LENGTH) {
      return text;
    }
  } catch {
    // Defuddle failed — fall through to heuristics
  }
  return null;
}

function extractWithHeuristics(): string | null {
  const selectors = [
    "article",
    "main",
    '[role="main"]',
    "#content",
    ".post-content",
    ".article-content",
    ".entry-content",
  ];

  for (const selector of selectors) {
    const el = document.querySelector(selector);
    if (el) {
      const text = (el as HTMLElement).innerText?.trim();
      if (text && text.length >= MIN_CONTENT_LENGTH) {
        return text;
      }
    }
  }
  return null;
}

function extractAllVisibleText(): string {
  const SKIP_TAGS = new Set([
    "SCRIPT", "STYLE", "NOSCRIPT", "SVG", "IFRAME",
    "NAV", "ASIDE", "FOOTER", "HEADER",
  ]);

  const chunks: string[] = [];
  const walker = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node: Text): number {
        const parent = node.parentElement;
        if (!parent) return NodeFilter.FILTER_REJECT;
        if (SKIP_TAGS.has(parent.tagName)) return NodeFilter.FILTER_REJECT;
        const style = window.getComputedStyle(parent);
        if (style.display === "none" || style.visibility === "hidden") {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    },
  );

  let node: Text | null;
  while ((node = walker.nextNode() as Text | null)) {
    const text = node.textContent?.trim();
    if (text) chunks.push(text);
  }
  return chunks.join("\n");
}

function stripHtmlTags(html: string): string {
  const div = document.createElement("div");
  div.innerHTML = html;
  return div.innerText || div.textContent || "";
}

function extract(): PageData {
  const text =
    extractWithDefuddle() ??
    extractWithHeuristics() ??
    extractAllVisibleText();

  return {
    title: document.title,
    url: location.href,
    text: text.slice(0, MAX_CHARS),
  };
}

extract();
