/**
 * Feed page — renders Penny's thoughts as a browsable card grid.
 * Two tabs: New (unnotified) and Archive (notified, paginated).
 */

import {
  type RuntimeMessage,
  RuntimeMessageType,
  type ThoughtCard,
} from "../protocol.js";

const grid = document.getElementById("grid")!;
const loading = document.getElementById("loading")!;
const loadMore = document.getElementById("load-more")!;
const loadMoreBtn = document.getElementById("load-more-btn")!;
const modal = document.getElementById("modal")!;
const modalImage = document.getElementById("modal-image") as HTMLImageElement;
const modalTitle = document.getElementById("modal-title")!;
const modalDate = document.getElementById("modal-date")!;
const modalText = document.getElementById("modal-text")!;

const ARCHIVE_PAGE_SIZE = 12;

let allThoughts: ThoughtCard[] = [];
let activeTab: "new" | "archive" = "new";
let archivePage = 0;
let modalThought: ThoughtCard | null = null;

// --- Init ---

function init(): void {
  browser.runtime.onMessage.addListener(handleMessage);
  browser.runtime.sendMessage({ type: RuntimeMessageType.ThoughtsRequest });

  for (const btn of Array.from(document.querySelectorAll(".tab"))) {
    btn.addEventListener("click", () => switchTab(btn.getAttribute("data-tab") as "new" | "archive"));
  }

  loadMoreBtn.addEventListener("click", loadNextPage);

  document.getElementById("modal-backdrop")!.addEventListener("click", closeModal);
  document.getElementById("modal-close")!.addEventListener("click", closeModal);
  document.getElementById("modal-thumbs-up")!.addEventListener("click", () => modalReact("\u{1f44d}"));
  document.getElementById("modal-thumbs-down")!.addEventListener("click", () => modalReact("\u{1f44e}"));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });
}

function handleMessage(message: RuntimeMessage): void {
  if (message.type === RuntimeMessageType.ThoughtsResponse) {
    allThoughts = message.thoughts;
    render();
  }
}

// --- Tabs ---

function switchTab(tab: "new" | "archive"): void {
  activeTab = tab;
  archivePage = 0;

  for (const btn of Array.from(document.querySelectorAll(".tab"))) {
    btn.classList.toggle("active", btn.getAttribute("data-tab") === tab);
  }

  render();
}

// --- Rendering ---

function render(): void {
  loading.classList.add("hidden");
  grid.innerHTML = "";

  const filtered = allThoughts.filter((t) =>
    activeTab === "new" ? !t.notified : t.notified
  );

  if (filtered.length === 0) {
    loading.textContent = activeTab === "new"
      ? "No new thoughts yet. Penny is still thinking..."
      : "No archived thoughts yet.";
    loading.classList.remove("hidden");
    loadMore.classList.add("hidden");
    return;
  }

  if (activeTab === "new") {
    renderCards(filtered);
    loadMore.classList.add("hidden");
  } else {
    const end = (archivePage + 1) * ARCHIVE_PAGE_SIZE;
    renderCards(filtered.slice(0, end));
    loadMore.classList.toggle("hidden", end >= filtered.length);
  }
}

function loadNextPage(): void {
  archivePage++;
  render();
}

function renderCards(thoughts: ThoughtCard[]): void {
  for (const thought of thoughts) {
    grid.appendChild(createCard(thought));
  }
}

function createCard(thought: ThoughtCard): HTMLElement {
  const card = document.createElement("div");
  card.className = "card";
  card.dataset.id = String(thought.id);
  card.style.cursor = "pointer";
  card.addEventListener("click", () => openModal(thought));

  if (thought.image) {
    const img = document.createElement("img");
    img.className = "card-image";
    img.src = thought.image;
    img.alt = thought.title;
    img.loading = "lazy";
    img.addEventListener("error", () => {
      img.replaceWith(createPlaceholder());
    });
    card.appendChild(img);
  } else {
    card.appendChild(createPlaceholder());
  }

  const body = document.createElement("div");
  body.className = "card-body";

  const title = document.createElement("div");
  title.className = "card-title";
  title.textContent = thought.title || "Untitled thought";
  body.appendChild(title);

  const byline = buildByline(thought);
  if (byline) {
    const date = document.createElement("div");
    date.className = "card-date";
    date.textContent = byline;
    body.appendChild(date);
  }

  const content = document.createElement("div");
  content.className = "card-content";
  content.innerHTML = thought.content;
  body.appendChild(content);

  card.appendChild(body);

  if (!thought.notified) {
    const actions = document.createElement("div");
    actions.className = "card-actions";

    const thumbsUp = document.createElement("button");
    thumbsUp.className = "reaction-btn thumbs-up";
    thumbsUp.innerHTML = '<i class="fa-solid fa-thumbs-up"></i>';
    thumbsUp.addEventListener("click", (e) => {
      e.stopPropagation();
      reactToThought(thought.id, "\u{1f44d}", card);
    });

    const thumbsDown = document.createElement("button");
    thumbsDown.className = "reaction-btn thumbs-down";
    thumbsDown.innerHTML = '<i class="fa-solid fa-thumbs-down"></i>';
    thumbsDown.addEventListener("click", (e) => {
      e.stopPropagation();
      reactToThought(thought.id, "\u{1f44e}", card);
    });

    actions.appendChild(thumbsUp);
    actions.appendChild(thumbsDown);
    card.appendChild(actions);
  }

  return card;
}

function reactToThought(thoughtId: number, emoji: string, card: HTMLElement | null): void {
  browser.runtime.sendMessage({
    type: RuntimeMessageType.ThoughtReaction,
    thought_id: thoughtId,
    emoji,
  });
  if (card) {
    card.style.transition = "opacity 0.3s, transform 0.3s";
    card.style.opacity = "0";
    card.style.transform = "scale(0.95)";
    setTimeout(() => card.remove(), 300);
  }
}

function createPlaceholder(): HTMLElement {
  const div = document.createElement("div");
  div.className = "card-image-placeholder";
  div.textContent = "💭";
  return div;
}

function buildByline(thought: ThoughtCard): string {
  const parts: string[] = [];
  if (thought.created_at) parts.push(formatDate(thought.created_at));
  parts.push(thought.seed_topic || "free thought");
  return parts.join(" · ");
}

function openModal(thought: ThoughtCard): void {
  modalThought = thought;
  modalImage.src = thought.image || "";
  modalTitle.textContent = thought.title || "Untitled thought";
  modalDate.textContent = buildByline(thought);
  modalText.innerHTML = thought.content;
  const actions = document.getElementById("modal-actions")!;
  actions.classList.toggle("hidden", thought.notified);
  modal.classList.remove("hidden");
}

function closeModal(): void {
  modal.classList.add("hidden");
  modalThought = null;
}

function modalReact(emoji: string): void {
  if (!modalThought) return;
  const card = document.querySelector(`.card[data-id="${modalThought.id}"]`) as HTMLElement | null;
  reactToThought(modalThought.id, emoji, card);
  closeModal();
}

function formatDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return "";
  }
}

// --- Boot ---

init();
