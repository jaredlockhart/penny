# Penny Browser Extension — Architecture & Design

## Vision

Penny originated as an idea 20 years ago: a personal agent that browses the web for you, finds things you'd find interesting, and builds a feed — rather than relying on external feeds. The browser extension brings this vision to life by giving Penny direct access to a browser, while keeping the existing server architecture intact.

## Architecture: Browser as a New Channel

The extension does NOT replace the current Penny architecture. It extends it.

```
                    ┌─────────────────┐
                    │   Penny Server  │
                    │   (Mac, 64GB)   │
                    │                 │
                    │  Agents         │
                    │  Scheduler      │
                    │  SQLite Store   │
                    │  Ollama (20B)   │
                    └──┬──────┬───────┘
                       │      │
              ┌────────┘      └────────┐
              │                        │
     ┌────────▼──────┐     ┌──────────▼──────────┐
     │ Signal Channel│     │ Browser Channel(s)  │
     │               │     │                      │
     │ Chat (phone,  │     │ Chat (sidebar)       │
     │  laptop, etc) │     │ Browser tools        │
     │               │     │ History reading       │
     └───────────────┘     │ Feed page            │
                           └──────────────────────┘
```

- **Penny server** stays where it is: model, store, thinking loop, scheduler, all on the Mac
- **Signal channel** stays as-is: talk to Penny from any device
- **Browser channel** is additive: a Firefox extension that connects via WebSocket, providing chat + browser capabilities
- **Multiple browsers** can connect: PC (always-on), MacBook (intermittent), etc. Each is a channel instance with a liveness state

## What the Extension Adds

The browser gives Penny capabilities she can't have through APIs alone:

- **Direct web browsing**: open pages, read full DOM, follow links, parse structured data
- **YouTube/video discovery**: see thumbnails, durations, view counts, comments, transcripts
- **Price tracking**: revisit URLs and detect changes (price drops, back-in-stock)
- **Rich media extraction**: preview images, Open Graph data, JSON-LD structured data
- **Browsing history as preferences**: passive preference extraction from what the user actually does, not just what they say
- **Contextual awareness**: see what the user is currently browsing, offer relevant information
- **Social/forum monitoring**: check Reddit threads, forums, Discord channels for interesting discussions
- **Feed page**: a browsable web page of Penny's discoveries, richer than Signal text notifications

## Extension Components

The extension itself is thin — no model, no database, no scheduler:

1. **Sidebar panel**: chat UI, WebSocket connection to Penny server
2. **Background script**: watches browsing history, manages browser tool API, handles liveness
3. **Content scripts**: read/interact with pages on demand (domain-gated)
4. **Feed page**: renders thoughts/discoveries from the server as a browsable feed

## Browser Tools

New tools exposed to Penny's agent loop when a browser channel is connected:

- `browse_url` — open a page, return sanitized text content
- `search_web` — use the browser's default search engine, return results
- `search_youtube` — search YouTube, return video metadata (title, duration, views, channel, thumbnail)
- `get_history` — read recent browsing history (filtered, domain + title only)
- `screenshot_page` — capture current page for vision model
- `get_current_tab` — what the user is looking at right now

The thinking agent can choose browser tools when a browser is connected, falling back to Perplexity when it's not. Purely additive.

## Skills

Skills are domain-specific playbooks — instructions for interacting with particular websites:

- **YouTube skill**: how to find videos, parse metadata, read transcripts, evaluate quality
- **Reddit skill**: how to find relevant threads, sort by quality, extract discussions
- **Amazon/Reverb skill**: how to find prices, reviews, availability, related products
- **Forum skill**: how to navigate threads, find expert opinions

Each skill is a document (not code) — instructions and DOM selectors that Penny reads when interacting with that domain. Skills are:

- Tied to the domain allowlist: granting access to youtube.com activates the YouTube skill
- User-extensible: write a skill document for any site
- Part of the trust boundary: defines what to extract and what to ignore

## Security Model

### Core Principles

Security is the foundational design constraint, not a feature added later.

1. **Zero permissions by default**: Penny has no access to any domain until the user explicitly grants it via an allowlist
2. **Collect, store, and expose nothing first**: only incrementally add data access as needed
3. **If Penny has no access to sensitive data, there's nothing to exfiltrate**

### Prompt Injection Defense

The primary threat: malicious web pages embedding instructions in content that Penny reads. Attack vectors include hidden text (CSS `display:none`, white-on-white), HTML comments, meta tags, YouTube descriptions/comments, forum posts, product descriptions.

**Three-layer defense:**

1. **Pre-sanitization**: content scripts extract visible text only, strip hidden content, HTML comments, invisible elements before anything reaches the model

2. **Sandboxed summarization**: raw sanitized web content goes to a **constrained model call** — system prompt is just "Summarize this page content." No tools, no user profile, no preferences, no history, no browsing context. Even if injected instructions survive sanitization, there's nothing to exfiltrate and no tools to act with.

3. **Summary is the only bridge**: Penny's real agent context (with tools, preferences, sensitive data) only sees the clean summary output from the sandboxed step. Untrusted content never shares a context window with sensitive data or tool access.

### HTML Sanitization

Use [ammonia](https://github.com/rust-ammonia/ammonia) (Rust) for sanitizing raw HTML before it reaches the model. Options:
- **In extension context**: compile ammonia to WASM via wasm-bindgen, use directly in the content/background script
- **In server context**: use [nh3](https://github.com/messense/nh3) (Python bindings to ammonia) if sanitization happens server-side

Ammonia strips dangerous elements/attributes while preserving safe content structure — purpose-built for this use case.

### Domain Allowlist

- No domains accessible by default
- User explicitly grants access per domain
- Allowlist maps directly to available skills
- Doubles as privacy control: "Penny can read guitar forums and YouTube, but not my email or banking"

### Data Policies

- Browsing history sent to server: domain + page title only, no full URLs with query params
- Filter out sensitive categories (health, finance, adult) before sending
- Communication over localhost or encrypted WebSocket only
- Model never sees raw untrusted web content — always extracted, sanitized, summarized first
- Penny never relays verbatim web page text through Signal — always summarized/rephrased
- Never pass raw cookies or session tokens to the server

### Tool Constraints

- Browser tools gated by domain allowlist
- Model output cannot drive navigation directly without gating
- Tool invocations rate-limited to prevent uncontrolled spidering
- Active tab scraping only on explicit user request — passive history reading is fine

## Multi-Browser Continuity

- Each browser instance connects as a channel with a unique identifier
- Firefox Sync provides identity continuity across machines (same Firefox Account)
- Persistent store lives on the Penny server — all browsers share it
- Liveness tracking: Penny knows which browsers are up and can choose which to use
- Always-on PC browser can do background research while laptop is closed
- Signal provides fallback communication when no browser is connected

## Feed Model vs Notification Model

The extension enables a shift from interruption-based notifications (Signal) to a browsable discovery feed:

- Signal notifications: Penny picks ONE thought, interrupts the user — high pressure on selection
- Feed page: Penny surfaces everything, user browses at their own pace — scoring becomes presentation order, not a gate
- Both coexist: feed for passive browsing, Signal for high-priority discoveries

## Technology

- **Extension**: TypeScript, WebExtensions API, Firefox-specific privileged APIs where needed
- **Communication**: WebSocket between extension and Penny server
- **Server**: existing Python/Docker architecture, unchanged
- **Model**: Ollama on server (20B), not constrained by browser ML engine limitations
- **Storage**: server SQLite, not browser storage (no IndexedDB limits)

## Incremental Build Path

1. Minimal Firefox extension: sidebar chat panel + WebSocket to Penny server (proves channel concept)
2. Add `browse_url` tool: extension can fetch and sanitize a page on demand
3. Add `search_web` tool: use browser's search engine
4. Browsing history reader: passive preference extraction
5. Feed page: render thoughts as a browsable page
6. YouTube skill: first domain-specific skill
7. Additional skills: Reddit, Amazon, forums, etc.
