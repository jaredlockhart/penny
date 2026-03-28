/**
 * Content script — extracts visible text from the current page.
 * Injected programmatically by the background script via browser.tabs.executeScript.
 * Returns structured page data to the caller.
 */

const MAX_CHARS = 50_000;

const SKIP_TAGS = new Set([
  "SCRIPT", "STYLE", "NOSCRIPT", "SVG", "IFRAME", "OBJECT", "EMBED",
  "TEMPLATE", "HEAD", "META", "LINK",
]);

interface PageData {
  title: string;
  url: string;
  text: string;
}

function extractVisibleText(): string {
  const chunks: string[] = [];
  let charCount = 0;

  const walker = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node: Text): number {
        const parent = node.parentElement;
        if (!parent) return NodeFilter.FILTER_REJECT;
        if (SKIP_TAGS.has(parent.tagName)) return NodeFilter.FILTER_REJECT;
        if (isHidden(parent)) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      },
    },
  );

  let node: Text | null;
  while ((node = walker.nextNode() as Text | null)) {
    if (charCount >= MAX_CHARS) break;
    const text = node.textContent?.trim();
    if (text) {
      chunks.push(text);
      charCount += text.length;
    }
  }

  return chunks.join("\n").slice(0, MAX_CHARS);
}

function isHidden(el: HTMLElement): boolean {
  if (el.hasAttribute("aria-hidden") && el.getAttribute("aria-hidden") === "true") {
    return true;
  }
  const style = window.getComputedStyle(el);
  return (
    style.display === "none" ||
    style.visibility === "hidden" ||
    style.opacity === "0"
  );
}

function extract(): PageData {
  return {
    title: document.title,
    url: location.href,
    text: extractVisibleText(),
  };
}

// Return the result to executeScript caller
extract();
