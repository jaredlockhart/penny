# CLAUDE.md — Penny Chat Agent

## Architecture Overview

```mermaid
flowchart TD
    User((User)) -->|message| Channel[Signal / Discord]

    subgraph Foreground["Foreground (ChatAgent)"]
        Channel -->|extract| CA[ChatAgent]
        CA -->|"prompt + tools"| FG_Ollama["LLM<br>(OpenAI SDK)"]
        FG_Ollama -->|tool call| Browse[BrowseTool]
        Browse -->|"read page"| Browser[Browser Extension]
        Browse -->|"web search"| Browser
        Browser -.->|results| FG_Ollama
        FG_Ollama -->|response| CA
    end

    CA -->|reply| Channel -->|send| User
    CA -->|log| DB[(SQLite)]

    subgraph Scheduler["Background Scheduler (when idle)"]
        direction TB

        SE[ScheduleExecutor] -->|"cron tasks"| FG_Ollama2["LLM<br>(OpenAI SDK)"]

        Think[ThinkingAgent] -->|"inner monologue<br>+ tools"| FG_Ollama3["LLM<br>(OpenAI SDK)"]
        Think -.->|"stored thoughts"| DB

        History[HistoryAgent] -->|"summarize + extract<br>preferences"| FG_Ollama4["LLM<br>(OpenAI SDK)"]
        History -.->|"topic summaries<br>+ preferences"| DB

        Notify[NotifyAgent] -->|"thoughts,<br>check-ins"| FG_Ollama5["LLM<br>(OpenAI SDK)"]
    end

    Notify -->|proactive message| Channel
    User -.->|"resets idle<br>cancels background"| Scheduler
```

- **Channels**: Signal (WebSocket + REST) or Discord (discord.py bot)
- **Ollama**: Local LLM inference (default model: gpt-oss:20b)
- **Vision**: Optional vision model (e.g., qwen3-vl) for processing image attachments from Signal
- **Image Generation**: Optional image model (e.g., x/z-image-turbo) for generating images via `/draw` command
- **Embedding Model**: Optional dedicated embedding model (e.g., embeddinggemma) for preference deduplication and history embeddings
- **Browser Extension**: Web search and page reading — all web access goes through the connected browser
- **SQLite**: Logs all prompts and messages; stores preferences, thoughts, and conversation history

## Directory Structure

```
penny/
  penny.py            — Entry point. Penny class: creates agents, channel, scheduler
  config.py           — Config dataclass loaded from .env, channel auto-detection
  config_params.py    — ConfigParam + RuntimeParams: runtime-configurable settings with 3-tier lookup
  constants.py        — Enums (SearchTrigger, PreferenceValence), reaction emojis, browse constants
  prompts.py          — LLM prompt templates (thinking, history, preference extraction)
  responses.py        — All user-facing response strings (PennyResponse class)
  startup.py          — Startup announcement message generation (git commit info)
  datetime_utils.py   — Timezone derivation from location (geopy + timezonefinder)
  agents/
    base.py           — Agent base class: agentic loop, tool execution, Ollama integration
    models.py         — ChatMessage, ControllerResponse, MessageRole, ToolCallRecord, GeneratedQuery
    chat.py           — ChatAgent: conversation-mode agent (handles user messages with tools)
    recall.py         — build_recall_block: assembles ambient recall context from active memories
    notify.py         — NotifyAgent: notification outreach (thoughts, check-ins)
    thinking.py       — ThinkingAgent: continuous inner monologue loop
    history.py        — HistoryAgent: daily/weekly summarization + preference extraction
  scheduler/
    base.py           — BackgroundScheduler + Schedule ABC
    schedules.py      — PeriodicSchedule, AlwaysRunSchedule, DelayedSchedule implementations
    schedule_runner.py — ScheduleExecutor: runs user-created cron-based scheduled tasks
  commands/
    __init__.py       — create_command_registry() factory
    base.py           — Command ABC, CommandRegistry
    models.py         — CommandContext, CommandResult, CommandError
    github_issue.py   — GitHubIssueCommand base class for /bug and /feature
    preference_base.py — PreferenceBaseCommand, PreferenceAddCommand, PreferenceRemoveCommand
    config.py         — /config: view and modify runtime settings
    debug.py          — /debug: show agent status, git commit, system info
    index.py          — /commands: list available commands
    profile.py        — /profile: user info collection (name, location, DOB, timezone)
    schedule.py       — /schedule: create and list recurring background tasks
    unschedule.py     — /unschedule: delete a scheduled task
    mute.py           — /mute: silence Penny's notifications
    unmute.py         — /unmute: resume Penny's notifications
    like.py           — /like: show or add positive preferences
    unlike.py         — /unlike: remove positive preferences
    dislike.py        — /dislike: show or add negative preferences
    undislike.py      — /undislike: remove negative preferences
    test.py           — /test: isolated test mode for development
    draw.py           — /draw: generate images via Ollama image model (optional)
    bug.py            — /bug: file GitHub issues (optional, requires GitHub App)
    feature.py        — /feature: file GitHub feature requests (optional, requires GitHub App)
    email.py          — /email: search Fastmail email via JMAP (optional)
    zoho.py           — /zoho: search Zoho Mail via Zoho Mail API (optional)
  tools/
    base.py           — Tool ABC, ToolRegistry, ToolExecutor
    models.py         — ToolCall, ToolResult, ToolDefinition, SearchResult, and per-tool arg models
    browse.py         — BrowseTool: web search and page reading via browser extension
    content_cleaning.py — Post-processing for browse results (strips navigation, proxy images, boilerplate)
    search_emails.py  — SearchEmailsTool (JMAP + Zoho)
    read_emails.py    — ReadEmailTool (JMAP + Zoho)
    list_emails.py    — ListEmailsTool (folder listings)
    list_folders.py   — ListFoldersTool (available mailboxes)
    draft_email.py    — DraftEmailTool (compose + stage draft)
    memory_args.py    — Pydantic arg models for the memory tool surface
    memory_context.py — ContextVar + current_agent() for write attribution
    memory_tools.py   — 21 Tool subclasses over db.memories.* (collection + log + introspection) and build_memory_tools factory
  channels/
    __init__.py       — create_channel() factory, channel type constants
    base.py           — MessageChannel ABC, IncomingMessage, shared message handling
    signal/
      channel.py      — SignalChannel: httpx for REST, websockets for receive
      models.py       — Signal WebSocket envelope Pydantic models
    discord/
      channel.py      — DiscordChannel: discord.py bot integration
      models.py       — DiscordMessage, DiscordUser Pydantic models
  database/
    database.py       — Database facade: thin wrapper creating domain stores
    knowledge_store.py — KnowledgeStore: summarized web page content for factual recall
    message_store.py  — MessageStore: log_message, log_prompt, log_command, threads
    thought_store.py  — ThoughtStore: inner monologue persistence
    preference_store.py — PreferenceStore: add, query, dedup, embedding management
    user_store.py     — UserStore: get_info, save_info, mute/unmute
    memory_store.py   — MemoryStore: unified collections + logs with three-signal dedup
    cursor_store.py   — CursorStore: per-agent read cursors into log-shaped memories
    media_store.py    — MediaStore: binary blobs referenced by <media:ID> tokens
    models.py         — SQLModel tables (see Data Model section)
    migrate.py        — Migration runner: file discovery, tracking table, validation
    migrations/       — Numbered migration files (0001–0025)
  llm/
    client.py         — LlmClient: OpenAI SDK wrapper (chat + embed) for any OpenAI-compatible backend (Ollama, omlx, etc.)
    image_client.py   — OllamaImageClient: Ollama-specific HTTP client for image generation and model listing
    models.py         — LlmMessage, LlmResponse, LlmToolCall, LlmError hierarchy (SDK-decoupled Pydantic types)
    embeddings.py     — Re-exports serialize/deserialize/cosine from shared similarity/ package
    similarity.py     — Penny-specific: embed_text, sentiment scores, novelty, preference vectors
  email/
    protocol.py       — EmailClient Protocol — shared interface for JMAP + Zoho email backends
  jmap/
    client.py         — JmapClient: Fastmail JMAP API client (httpx)
    models.py         — JmapSession, EmailAddress, EmailSummary, EmailDetail
  zoho/
    client.py         — ZohoClient: Zoho Mail API client (httpx + OAuth refresh)
    models.py         — Zoho Mail API Pydantic models
  html_utils.py       — Shared HTML text extraction helpers
  tests/
    conftest.py       — Pytest fixtures for mocks and test config
    test_embeddings.py, test_similarity.py, test_periodic_schedule.py, test_scheduler.py
    mocks/
      signal_server.py  — Mock Signal WebSocket + REST server (aiohttp)
      llm_patches.py    — MockLlmClient: patches openai.AsyncOpenAI for chat + embed
    agents/           — Per-agent integration tests
      test_message.py, test_thinking.py, test_agentic_loop.py,
      test_context.py, test_history.py, test_notify.py
    channels/         — Channel integration tests
      test_signal_channel.py, test_signal_reactions.py, test_signal_vision.py,
      test_signal_formatting.py, test_startup_announcement.py
    commands/         — Per-command tests
      test_commands.py, test_config.py, test_debug.py, test_draw.py, test_email.py,
      test_feature.py, test_mute.py, test_preferences.py,
      test_schedule.py, test_bug.py, test_system.py, test_test_mode.py
    database/         — Migration validation tests
      test_migrations.py
    jmap/             — JMAP client tests
      test_client.py
    tools/            — Tool tests
      test_tool_timeout.py, test_tool_not_found.py, test_tool_reasoning.py
Dockerfile            — Python 3.14-slim
pyproject.toml        — Dependencies and project metadata
```

## Agent Architecture

### Agent Base Class (`agents/base.py`)
The base `Agent` class implements the core agentic loop:
- Calls the LLM (via `LlmClient`) with available tools
- Executes tool calls via `ToolExecutor` with parameter validation
- Handles duplicate tool call prevention
- Appends source URLs to responses when model omits them

**System prompt building (template method pattern):**
Each agent overrides `_build_system_prompt(user)` to compose its prompt from reusable building blocks on the base class: `_identity_section()`, `_profile_section()`, `_thought_section()`, `_dislike_section()`, `_related_knowledge_section()`, `_related_messages_section()`, `_instructions_section()`, `_context_block()`. No flags or conditionals — each agent explicitly declares what goes in its prompt. Tests assert on the exact full system prompt string to catch structural drift.

**Knowledge vs related-message retrieval scoring**: These use deliberately different algorithms:

- **Knowledge retrieval** scores each candidate as `max(weighted_decay, current_message_cosine)` and applies an absolute floor (`RELATED_KNOWLEDGE_SCORE_FLOOR`, default 0.34). The weighted-decay leg (decay=0.5 over the conversation window) preserves the "vague follow-up needs prior context" case — e.g. asking "is it a dud?" after a storm-glass thread still surfaces storm-glass entries. The current-message-cosine leg lets a strong direct match stand on its own merit even when the conversation has drifted to a different topic, so a topic-bearing question isn't dragged toward the prior thread. The absolute floor suppresses noise on greetings and topics the corpus doesn't cover, instead of forcing retrieval to pick its top-N least-bad guesses.
- **Related-message retrieval** scores by `cosine_to_current_message - α × centrality`, where centrality is the candidate's mean cosine to the rest of the corpus. Past messages should match what's being asked right now, not adjacent topics from earlier in the thread (which caused derailment when the model latched onto a stale prior turn). The centrality penalty (α=0.5) suppresses generic centroid-magnet boilerplate that would otherwise leak into every unrelated query. After scoring, an adaptive cutoff is applied: a cluster-strength gate (`head_mean / sample_mean ≥ 1.15`, in adjusted-score space, where head=top-5 and sample=top-20) suppresses flat noise plateaus entirely, then `cutoff = max(head_mean × 0.85, 0.25)` combines a relative band against the cluster center with an empirical absolute floor. Strong clusters return many messages, weak clusters return few, no cluster returns nothing. After cutoff selects hits, a single-pass neighbor expansion pulls user messages within ±5 minutes of each hit's timestamp, deduped by id and content and excluding the current conversation window — this captures conversational follow-ups that share no entity overlap with the current message ("yeah exactly i can't wait to try it") but live in the same conversation as a real hit. Neighbors are not themselves expanded. Centrality is cached per-sender in memory (lazy on first retrieval, drifts as new messages arrive — revisit with a DB column or background refresh if precision degrades or the corpus grows past a few thousand messages). Constants live in `penny/constants.py` under `PennyConstants.RELATED_MESSAGES_*`.

Conversation embeddings are computed once per request and shared between knowledge and message retrieval (the message path uses just the last element).

### Shared LLM Client Instances

All `LlmClient` instances are created centrally in `Penny.__init__()` and shared across agents and commands. `LlmClient` uses the OpenAI Python SDK and targets any OpenAI-compatible endpoint (Ollama's OpenAI-compat layer by default, or omlx/OpenAI cloud with a different `base_url`):

- `model_client`: Text model for all agents and commands
- `vision_model_client`: Optional vision model for image understanding
- `embedding_model_client`: Optional embedding model for preference deduplication
- `image_model_client`: `OllamaImageClient` for `/draw` (image generation uses Ollama's native REST API, not OpenAI-compatible)

### Specialized Agents

**ChatAgent** (`agents/chat.py`)
- Handles incoming user messages with tools (BrowseTool for search and page reading)
- Prompt: identity + profile + knowledge + related messages + page hint + conversation instructions
- Vision captioning: when images are present and vision model is configured, captions the image first, then forwards a combined prompt to the text LLM

**NotifyAgent** (`agents/notify.py`)
- Notification outreach — sends thoughts and check-ins when users are idle
- Runs on a PeriodicSchedule, separate from ChatAgent
- Two modes, each a `NotificationMode` subclass (`CheckinMode`, `ThoughtMode`):
  - Each mode declares its tools, system prompt composition, user prompt, and image extraction
  - `_execute_mode()` orchestrates the shared pipeline: prepare → tools → prompt → run → validate → send
  - Thought candidates: identity + profile + pending thought + instructions
  - Check-in: identity + profile + history + notified thought + instructions
- Candidate scoring: novelty (avoid repeating recent messages) + sentiment (preference alignment)
- Exponential backoff cooldown between autonomous messages

**ThinkingAgent** (`agents/thinking.py`)
- Autonomous research loop — Penny searches for information on topics she's interested in
- No identity or profile — thinking never communicates with the user
- Runs on a PeriodicSchedule
- Each cycle picks a random seed topic from positive user preferences to focus on
- Seeded cycles get scoped thought context; free cycles get NO thought context (prevents fixation — the model reads its own previous thoughts and re-searches them)
- Tools stay available on final step (`_keep_tools_on_final_step = True`) — no forced text synthesis
- At the end of each cycle, the raw search results (from `_tool_result_text`) are summarized into a detailed briefing and stored as a thought via ThoughtStore (single summarization step)
- Stored thought summaries bleed into chat context, giving Penny continuity of inner reasoning

**HistoryAgent** (`agents/history.py`)
- Background worker that extracts knowledge from browse results and user preferences from messages
- Runs on a PeriodicSchedule (highest priority among idle tasks — before notify and thinking)
- Each cycle: (0) extract knowledge from browse results (user-independent), then per user: (1) extract preferences from unprocessed messages
- Knowledge extraction: scans prompt logs incrementally (watermark in RuntimeConfig), extracts browse tool results, summarizes page content into prose paragraphs via LLM, embeds summaries, upserts by URL
- Preference extraction: two-pass LLM pipeline — (1) identify new topics from user messages + reactions, (2) classify valence (positive/negative). Topics are deduped against existing preferences via TCR + embedding similarity before storage

**ScheduleExecutor** (`scheduler/schedule_runner.py`)
- Background task: runs user-created cron-based scheduled tasks
- Checks every 60 seconds for due schedules (based on user timezone)
- Executes the schedule's prompt text via the agentic loop
- Sends results to the user via channel

## Scheduler System

The `scheduler/` module manages background tasks:

### BackgroundScheduler (`scheduler/base.py`)
- Runs tasks in priority order (schedule executor → history → notify → thinking)
- **Skips agents with no work**: when an agent returns False, continues to the next eligible schedule in the same tick. Only breaks when an agent does real work.
- Tracks global idle threshold (default: 60s)
- Notifies schedules when messages arrive (resets timers)
- Passes `is_idle` boolean to schedules (whether system is past global idle threshold)
- **Cancels active background task** when a foreground message arrives (`notify_foreground_start()` calls `task.cancel()`), freeing Ollama immediately for the user's message. Cancelled tasks are idempotent — unprocessed items stay in their queues and are re-picked up on the next cycle
- Commands do NOT interrupt background tasks — they run cooperatively

### Schedule Types (`scheduler/schedules.py`)

**AlwaysRunSchedule**
- Runs regardless of idle state at a configurable interval
- Used for ScheduleExecutor (60s interval)
- Not affected by idle threshold — scheduled tasks run even during active conversations

**PeriodicSchedule**
- Runs periodically while system is idle at a configurable interval
- Used for history (3600s), notify (300s), and thinking (1200s) agents
- Tracks last run time and fires again after interval elapses
- Resets when a message arrives

**DelayedSchedule**
- Runs after system becomes idle + random delay
- Available for future use (not currently used by any agent)

## Channel System

### MessageChannel ABC (`channels/base.py`)
- Defines interface: `listen()`, `send_message()`, `send_typing()`, `extract_message()`
- Implements shared logic: `handle_message()`, `send_response()`, `_typing_loop()`
- Holds references to chat agent, database, and scheduler
- **Progress tracker hook**: `_begin_progress(message)` is an optional override that returns a `ProgressTracker` (defined in `channels/base.py`). The tracker has two methods: `update(tools)` (called when a tool batch starts) and `clear()` (idempotent, called once on success and once again from the dispatch loop's `finally`). The default `_make_handle_kwargs` wires `progress.update` as `on_tool_start` for free, and the final response is always delivered via `send_response` so attachments and quote-replies work normally. Channels without a progress UI return `None`

### SignalChannel (`channels/signal/channel.py`)
- WebSocket connection for receiving messages
- REST API for sending messages, typing indicators, and reactions
- Handles quote-reply thread reconstruction
- **Startup connectivity validation**: `validate_connectivity()` retries DNS + a `GET /v1/about` probe up to `PennyConstants.SIGNAL_VALIDATE_MAX_ATTEMPTS` times with `SIGNAL_VALIDATE_RETRY_DELAY` between attempts (~60 s budget) so cold-boot startup can wait out signal-cli-rest-api's 30-60 s warmup. Each failed attempt is logged at WARNING; the final exhaustion is logged at ERROR and the `ConnectionError` is caught in `main()` and written to `penny.log` before exiting. `docker-compose.yml` also gates `penny` on a `curl /v1/about` healthcheck against `signal-api` via `depends_on: service_healthy`, so compose-managed startups never even hit the retry loop. Tests pass `max_attempts=1, retry_delay=0` to stay fast
- **In-flight progress as emoji reactions**: when a user message arrives, the channel reacts to it with 💭 (thinking) via `POST /v1/reactions`. As the agent's tool calls fire, `SignalProgressTracker.update()` swaps the reaction to a tool-specific emoji from `Tool.format_progress_emoji()` (BrowseTool returns 🔍 for searches, 📖 for URL reads). Signal limits each user to one reaction per message, so each new emoji cleanly replaces the previous — no clutter. When the agent finishes, `tracker.clear()` issues `DELETE /v1/reactions` to remove the reaction entirely, and the response is sent as a normal new message via `send_response` (with text + attachments + quote-reply, the same shape as before progress was added). The typing indicator runs alongside throughout. Why reactions instead of editing a "thinking..." text bubble: Signal mobile/desktop clients silently drop attachments added via message edit — even though the wire format technically allows them — so any final response with an image would lose its image. Reactions sidestep editing entirely

### DiscordChannel (`channels/discord/channel.py`)
- Uses discord.py for bot integration
- Listens to a single configured channel
- Handles 2000-character message limit by chunking
- Typing indicators auto-expire (no stop needed)

### Channel Factory (`channels/__init__.py`)
- `create_channel()` creates appropriate channel based on config
- Auto-detects channel type from credentials if not explicit

## Command System

Penny supports slash commands sent as messages (e.g., `/debug`, `/config`). Commands are handled before the message reaches the agent loop.

### Architecture (`commands/`)
- **Command ABC** (`base.py`): Each command implements `name`, `description`, `aliases`, and `async execute(context) → CommandResult`
- **CommandRegistry** (`base.py`): Maps command names/aliases to handlers, dispatches messages starting with `/`
- **Factory** (`__init__.py`): `create_command_registry()` registers all built-in commands

### Built-in Commands (always registered)
- **/commands** (`index.py`): Lists all available commands with descriptions
- **/debug** (`debug.py`): Shows agent status, git commit, system info, background task state
- **/config** (`config.py`): View and modify runtime settings (e.g., `/config idle_seconds 600`). Reads/writes RuntimeConfig table in SQLite; changes take effect immediately
- **/profile** (`profile.py`): View or update user profile (name, location, DOB). Derives IANA timezone from location. Required before Penny will chat
- **/schedule** (`schedule.py`): Create and list recurring cron-based background tasks (uses LLM to parse natural language timing)
- **/unschedule** (`unschedule.py`): Delete a scheduled task. `/unschedule` shows numbered list; `/unschedule <N>` deletes
- **/mute** (`mute.py`): Silence Penny's autonomous notifications
- **/unmute** (`unmute.py`): Resume Penny's notifications
- **/like** (`like.py`): Show positive preferences or add one (e.g., `/like dark roast coffee`)
- **/unlike** (`unlike.py`): Remove a positive preference by number
- **/dislike** (`dislike.py`): Show negative preferences or add one
- **/undislike** (`undislike.py`): Remove a negative preference by number

### Conditional Commands (registered based on config)
- **/test** (`test.py`): Enters isolated test mode — creates a separate DB and fresh agents for testing without affecting production data. Exit with `/test stop` (requires message agent factory)
- **/draw** (`draw.py`): Generate images via Ollama image model (requires `LLM_IMAGE_MODEL`)
- **/bug** (`bug.py`): File a bug report on GitHub (requires GitHub App config)
- **/feature** (`feature.py`): File a feature request on GitHub (requires GitHub App config)
- **/email** (`email.py`): Search Fastmail email via JMAP (requires `FASTMAIL_API_TOKEN`)
- **/zoho** (`zoho.py`): Search Zoho Mail via the Zoho Mail API (requires `ZOHO_API_ID`, `ZOHO_API_SECRET`, `ZOHO_REFRESH_TOKEN`)

### Runtime Configuration
- `/config` reads and writes to a `RuntimeConfig` table in SQLite
- `ConfigParam` definitions in `config_params.py` declare runtime-configurable settings with types and validation
- `RuntimeParams` class provides attribute access: `config.runtime.IDLE_SECONDS`
- Three-tier lookup chain: DB override → env override → ConfigParam.default
- Config values are read on each use (not cached), so changes take effect immediately
- Groups: Chat (max steps, search URL, context limits, retrieval thresholds, domain permission mode), Thinking (interval, max steps, dedup thresholds, free-thinking probability), History (interval, preference dedup, mention threshold, knowledge/embedding batch limits), Notify (idle threshold, check interval, cooldowns, candidates), Email (body max length, search/list limits, request timeout)

## Data Model

All tables defined in `database/models.py` as SQLModel classes:

- **PromptLog**: Every LLM call — `model`, `messages` (JSON), `response` (JSON), `thinking`, `duration_ms`, `agent_name`, `run_id`, `outcome`
- **MessageLog**: Every user/agent message — `direction`, `sender`, `content`, `parent_id` (thread chain), `external_id` (platform ID), `is_reaction`, `thought_id` FK (notification source)
- **UserInfo**: User profile — `name`, `location`, `timezone` (IANA), `date_of_birth`
- **CommandLog**: Command invocations — `command_name`, `command_args`, `response`, `error`
- **RuntimeConfig**: User-configurable settings — `key`, `value` (string, parsed on read)
- **Schedule**: User-created cron tasks — `cron_expression`, `prompt_text`, `user_timezone`
- **MuteState**: Per-user mute state — row exists = muted, delete = unmuted
- **Device**: Registered devices (Signal, Discord, browser addons) — used for multi-device routing and domain permission prompts
- **DomainPermission**: Per-domain allow/deny state for browser extension web access, synced across addons
- **Thought**: Inner monologue entries — `content` (full monologue), `title`, `image`, `valence`, `preference_id` FK (seed preference), `run_id`, `notified_at`
- **Preference**: User sentiment signals — `content`, `valence` (positive/negative), `source` (manual/extracted), `mention_count`, `embedding` (serialized float32 vector), `last_thought_at`. Extracted preferences must reach `PREFERENCE_MENTION_THRESHOLD` mentions before becoming thinking candidates; manual (`/like`) preferences bypass this gate
- **Knowledge**: Summarized web page content — `url` (unique), `title`, `summary` (prose paragraph), `embedding`, `source_prompt_id` FK (extraction watermark). One entry per URL, upserted on revisit
- **Memory**: Unified container for the task/memory framework — `name` (PK), `type` (`collection` or `log`), `description`, `recall` (`off` / `recent` / `relevant` / `all`), `archived`. Collections are keyed sets with dedup on write; logs are append-only keyless streams
- **MemoryEntry**: One entry in a memory — `memory_name` FK, `key` (nullable for logs), `content`, `author`, `key_embedding`, `content_embedding`. Entries are immutable once written — `update` replaces content for a given key
- **AgentCursor**: Per-agent read progress through a log-shaped memory — `(agent_name, memory_name)` PK, `last_read_at` high-water mark. Advanced two-phase by the orchestrator (pending during a run, committed on success)
- **Media**: Binary blobs (images, etc.) referenced by `<media:ID>` tokens in memory entry content — `mime_type`, `data`, `source_url`

## Message Flow

1. Channel receives message → `extract_message()` → `IncomingMessage`
2. Channel calls `handle_message()`:
   - Checks for slash commands first (dispatches via `CommandRegistry`)
   - Notifies scheduler (resets idle timers, suspends background tasks)
   - Starts typing indicator loop
   - Calls `ChatAgent.handle()` which:
     - Finds parent message if quote-reply (via `external_id` lookup)
     - Walks thread history for context
     - Runs agentic loop with tools
   - Logs incoming message to DB
   - Sends response via `send_response()` (logs + sends)
   - Stops typing indicator, resumes background tasks

## Thread/Context System

- Quote-replying continues a conversation thread
- `MessageLog.parent_id` creates a chain of messages
- `db.messages.get_thread_context()` walks the chain (up to 20 messages)

## Key Design Decisions

- **Browser-based search**: All web access (search, page reading) goes through the browser extension via BrowseTool. Text queries are converted to search URLs (configurable via `SEARCH_URL`). No third-party search APIs
- **URL fallback**: If the model's final response doesn't contain any URL, the agent appends the first source URL
- **Duplicate tool blocking**: Agent tracks called tools per message to prevent LLM tool-call loops
- **Tool parameter validation**: Tool parameters validated before execution; non-existent tools return clear error messages
- **Specialized agents**: Each task type (chat, thinking, history) has its own agent subclass
- **Priority scheduling**: Schedule executor → history → notify → thinking (agents with no work are skipped each tick)
- **Always-run schedules**: User-created schedules run regardless of idle state; thinking/history wait for idle
- **Global idle threshold**: Single configurable idle time (default: 60s) controls when idle-dependent tasks become eligible
- **Background cancellation**: Foreground message processing cancels active background tasks (`task.cancel()`) to free the LLM immediately; cancelled work is idempotent and retried next cycle
- **Commands don't interrupt background**: Slash commands run cooperatively without cancelling the active background task
- **Vision captioning**: When images are present and `LLM_VISION_MODEL` is configured, the vision model captions the image first with a vision-specific system prompt, then a combined prompt is forwarded to the text LLM. Search tools are disabled for image messages
- **Channel abstraction**: Signal and Discord share the same interface; easy to add more platforms
- **Async throughout**: asyncio, httpx.AsyncClient, openai.AsyncOpenAI, discord.py
- **Host networking**: Docker container uses --network host for simplicity (all services on localhost)
- **Pydantic everywhere**: All external data validated with Pydantic models
- **Table-to-bullets**: Markdown tables converted to bullet points in Python (saves model tokens vs. prompting "no tables")
- **Normal casing**: All user-facing strings (status messages, error messages, acknowledgments) use standard sentence casing — not all lowercase
- **Memory framework (Stages 1–3)**: A unified data primitive — *memory* — with two shapes (collection and log) and one access class `MemoryStore`. Collections dedup on write via a three-signal disjunction (key TCR, key cosine, content cosine — each with strict and relaxed thresholds in `PennyConstants`). Any strict hit, or any two relaxed hits, rejects the write. Logs append without dedup. Stage 2a added 21 model-facing memory tools (`memory_tools.py`). Stage 3 added `build_recall_block` (`recall.py`) — assembles ambient recall context for the chat agent's system prompt by dispatching each active memory by recall mode (`recent`/`relevant`/`all`); paired logs (`user-messages` + `penny-messages`) merge chronologically into a single Conversation section. `db.memories` replaces the per-domain stores that agents will be ported onto in subsequent stages. See `docs/task-framework-plan.md` (design) and `docs/memory-implementation-plan.md` (staged rollout)

## Dependencies

- `websockets`, `httpx`, `python-dotenv`, `pydantic`, `sqlmodel`, `openai`, `discord.py`, `psutil`, `dateparser`, `timezonefinder`, `geopy`, `pytz`, `croniter`, `PyJWT`
- Dev: `ruff` (lint/format), `ty` (type check), `pytest`, `pytest-asyncio`, `aiohttp` (mock Signal server)
- Python 3.14+

## Database Migrations

File-based migration system in `database/migrations/` (currently 0001–0025):
- Each migration is a numbered Python file (e.g., `0001_initial_schema.py`) with a `def up(conn)` function
- Two types: **schema** (DDL — ALTER TABLE, CREATE INDEX) and **data** (DML — UPDATE, backfills), both use `up()`
- Runner in `database/migrate.py` discovers files, tracks applied migrations in `_migrations` table
- Runs on startup before `create_tables()` in `penny.py`
- `make migrate-test`: copies production DB, applies migrations to copy, reports success/failure
- `make migrate-validate`: checks for duplicate migration number prefixes (also runs in `make check`)
- Rebase-only policy: if two PRs create the same migration number, the second must rebase and renumber
- Run standalone: `python -m penny.database.migrate [--test] [--validate] [db_path]`

Notable migrations:
- 0001: Initial schema (all core tables)
- 0002: `thought.notified_at` column
- 0003: Preference deduplication
- 0004: Drop `entity` and `fact` tables (old knowledge system removed)
- 0005: `preference.last_thought_at` column
- 0006: `messagelog.thought_id` FK (links messages to notification thoughts)
- 0007: `thought.preference_id` FK (links thoughts to seed preferences)
- 0008: `preference.source` + `preference.mention_count` (mention threshold gating)
- 0009: Drop `searchlog.extracted` column
- 0010: Reset reaction `processed` state
- 0011: Drop `preference.source_period_start/end` columns
- 0012: Fix `is_reaction` flag on historical reaction rows
- 0013: Reset conversation history watermarks
- 0014: Add embedding columns (preference, knowledge, etc.)
- 0015: `thought.title` column
- 0016: `device` table (multi-device routing)
- 0017: `thought.image_url` column
- 0018: `thought.valence` column
- 0019: `domain_permission` table (browser extension allowlist)
- 0020: Rename `thought.image_url` → `thought.image`
- 0021: `promptlog.agent_name` + `promptlog.run_id` columns
- 0022: `promptlog.outcome` + `thought.run_id` columns
- 0023: Add `knowledge` table, drop `conversationhistory` (replaced by knowledge + related messages)
- 0024: Drop legacy `searchlog` table (never written to since browser-based search)
- 0025: Add `memory`, `memory_entry`, `agent_cursor`, `media` tables (task/memory framework Stage 1)

## Extending

- **New tool**: Subclass `Tool` in tools/, implement `name`, `description`, `parameters`, `async execute()`, add to agent's tool list in penny.py
- **New channel**: Implement `MessageChannel` ABC, create models, add to `create_channel()` factory
- **New agent type**: Subclass `Agent`, implement `execute()` for background tasks or custom `handle()` for message processing
- **New command**: Subclass `Command` in commands/, implement `name`, `description`, `execute()`, register in `create_command_registry()`
- **New schedule type**: Subclass `Schedule`, implement `should_run()`, `reset()`, `mark_complete()`
- **New LLM backend**: Any OpenAI-compatible endpoint works via `LlmClient` — just set `base_url` / `api_key`. Non-OpenAI-compatible backends can implement the `LlmClient` interface directly (`async chat()`, `async embed()`)

## Test Infrastructure

Strongly prefer end-to-end integration tests over unit tests. Test through public entry points with mocks for external services. Prefer folding new assertions into existing tests over adding new test functions — only add a new test when no existing test covers the relevant code path.

**Mocks** (in `tests/mocks/`):
- `MockSignalServer`: WebSocket + REST server using aiohttp, captures outgoing messages and typing events
- `MockLlmClient` (`llm_patches.py`): Monkeypatches `openai.AsyncOpenAI` so `LlmClient` returns canned `LlmResponse` objects; configurable via `set_default_flow()` or `set_response_handler()`; tracks `requests` and `embed_requests` for assertions

**Fixtures** (in `tests/conftest.py`):
- `TEST_SENDER`: Standard test phone number constant
- `signal_server`: Starts mock Signal server on random port
- `mock_llm`: Patches the OpenAI SDK with configurable responses
- `make_config`: Factory for creating test configs with custom overrides
- `running_penny`: Async context manager for running Penny with cleanup (uses WebSocket detection, not sleep)
- `setup_llm_flow`: Factory to configure mock LLM for message + background task flow
- `wait_until(condition, timeout, interval)`: Polls a condition every 50ms until true or timeout (10s default)

**Test Timing** — never use `asyncio.sleep(N)` in tests:
- Use `wait_until(lambda: <condition>)` to poll for expected side effects (DB state, message count, etc.)
- `scheduler_tick_interval` is set to 0.05s in test config (vs 1.0s production) so scheduler-dependent tests complete quickly
- `running_penny` detects WebSocket connection via `signal_server._websockets` instead of sleeping
- For negative assertions (nothing should happen), verify immediately — don't sleep to "make sure"

**Test Flow**:
1. Start mock Signal server (random port)
2. Monkeypatch the OpenAI SDK (via `mock_llm`)
3. Create Penny with test config pointing to Signal mock
4. Push message through mock Signal WebSocket
5. `wait_until` the expected side effect (outgoing message, DB change, etc.)
6. Assert on captured messages, LLM requests, DB state

**Performance**: Test suite runs in ~30s (`scheduler_tick_interval` set to 0.05s in tests)
