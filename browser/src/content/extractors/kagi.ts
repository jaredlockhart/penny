/**
 * Kagi search results extractor.
 *
 * Extracts structured search results (title, URL, snippet) from Kagi's DOM.
 * Returns null when .search-result elements haven't rendered yet — the caller
 * (pollForContent) re-injects the content script until ready=true.
 */

export function extractKagiResults(): string | null {
  if (!location.hostname.includes("kagi.com")) return null;

  const results = document.querySelectorAll(".search-result");
  if (results.length === 0) return null;

  const lines: string[] = [];
  results.forEach((el) => {
    const linkEl = el.querySelector("a.__sri_title_link, a._0_URL");
    if (!linkEl) return;

    const title = (linkEl.getAttribute("title") || linkEl.textContent || "").trim();
    const href = linkEl.getAttribute("href") || "";
    if (!title || !href || href.startsWith("https://kagi.com/")) return;

    const descEl = el.querySelector(".__sri-desc, ._0_sri-description");
    const snippet = descEl
      ? (descEl as HTMLElement).innerText?.trim() || ""
      : "";

    let line = `${title}\n${href}`;
    if (snippet) line += `\n${snippet}`;
    lines.push(line);
  });

  return lines.length > 0 ? lines.join("\n\n---\n\n") : null;
}
