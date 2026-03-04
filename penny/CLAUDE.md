# CLAUDE.md — Penny Chat Agent

## Architecture Overview

```mermaid
flowchart TD
    User((User)) -->|message| Channel[Signal / Discord]

    subgraph Foreground["Foreground (ChatAgent)"]
        Channel -->|extract| CA[ChatAgent]
        CA -->|"prompt + tools"| FG_Ollama["Ollama<br>foreground model"]
        FG_Ollama -->|tool call| Search[SearchTool]
        FG_Ollama -->|tool call| News[FetchNewsTool]
        Search -->|text| Perplexity[Perplexity API]
        Search -->|images| Serper[Serper API]
        News -->|articles| TheNewsAPI[TheNewsAPI.com]
        Search -.->|results| FG_Ollama
        News -.->|results| FG_Ollama
        FG_Ollama -->|response| CA
    end

    CA -->|reply + image| Channel -->|send| User
    CA -->|log| DB[(SQLite)]
    Search -->|log| DB

    subgraph Scheduler["Background Scheduler (when idle)"]
        direction TB

        SE[ScheduleExecutor] -->|"cron tasks"| FG_Ollama2["Ollama<br>foreground model"]

        subgraph Pipeline["ExtractionPipeline (priority order)"]
            direction TB
            P1["Phase 1<br>User Messages"] --> P2["Phase 2<br>Search Logs"]
            P2 --> P3["Phase 3<br>Embedding Backfill"]
        end

        Pipeline -->|"extract entities<br>& facts"| BG_Ollama["Ollama<br>background model"]
        BG_Ollama -.-> DB

        Think[ThinkingAgent] -->|"inner monologue<br>+ tools"| BG_Ollama
        Think -.->|"stored thoughts"| DB

        History[HistoryAgent] -->|"summarize<br>conversations"| BG_Ollama
        History -.->|"topic summaries"| DB
    end

    DB -.->|"unprocessed<br>messages & searches"| P1

    User -.->|"resets idle<br>cancels background"| Scheduler

    Embed["Ollama<br>embedding model"] -.-> Pipeline
```

- **Channels**: Signal (WebSocket + REST) or Discord (discord.py bot)
- **Ollama**: Local LLM inference (default model: gpt-oss:20b)
- **Vision**: Optional vision model (e.g., qwen3-vl) for processing image attachments from Signal
- **Image Generation**: Optional image model (e.g., x/z-image-turbo) for generating images via `/draw` command
- **Embedding Model**: Optional dedicated embedding model (e.g., embeddinggemma) for semantic validation, dedup, and entity context injection
- **Perplexity**: Web search — Penny always searches before answering, never uses model knowledge alone
- **Serper**: Image search (Google Images) — runs in parallel with Perplexity, attaches a relevant image to every response
- **SQLite**: Logs all prompts, searches, and messages; stores thread history via parent-child links

## Directory Structure

```
penny/
  penny.py            — Entry point. Penny class: creates agents, channel, scheduler
  config.py           — Config dataclass loaded from .env, channel auto-detection
  config_params.py    — ConfigParam definitions for runtime-configurable settings
  constants.py        — System prompt, research prompts, trigger enums
  prompts.py          — LLM prompt templates (extraction, thinking, history)
  responses.py        — All user-facing response strings (PennyResponse class)
  startup.py          — Startup announcement message generation (git commit info)
  datetime_utils.py   — Timezone derivation from location (geopy + timezonefinder)
  agents/
    base.py           — Agent base class: agentic loop, tool execution, Ollama integration
    models.py         — ChatMessage, ControllerResponse, MessageRole, ToolCallRecord, GeneratedQuery
    chat.py           — ChatAgent: conversation-mode agent (handles user messages with tools)
    thinking.py       — ThinkingAgent: continuous inner monologue loop
    penny_agent.py    — PennyAgent: penny agent composition
    extraction.py     — ExtractionPipeline: unified entity/fact extraction from search results and messages
    history.py        — HistoryAgent: daily conversation topic summarization
  scheduler/
    base.py           — BackgroundScheduler + Schedule ABC
    schedules.py      — PeriodicSchedule, AlwaysRunSchedule, DelayedSchedule implementations
    schedule_runner.py — ScheduleExecutor: runs user-created cron-based scheduled tasks
  commands/
    __init__.py       — create_command_registry() factory
    base.py           — Command ABC, CommandRegistry
    models.py         — CommandContext, CommandResult, CommandError
    github_issue.py   — GitHubIssueCommand base class for /bug and /feature
    config.py         — /config: view and modify runtime settings
    debug.py          — /debug: show agent status, git commit, system info
    index.py          — /commands: list available commands
    profile.py        — /profile: user info collection (name, location, DOB, timezone)
    memory.py         — /memory: view/manage knowledge base entities and facts
    forget.py         — /forget: remove entities or facts from knowledge base
    schedule.py       — /schedule: create and list recurring background tasks
    unschedule.py     — /unschedule: delete a scheduled task
    mute.py           — /mute: silence Penny's proactive messages
    unmute.py         — /unmute: resume Penny's proactive messages
    test.py           — /test: isolated test mode for development
    draw.py           — /draw: generate images via Ollama image model (optional)
    bug.py            — /bug: file GitHub issues (optional, requires GitHub App)
    feature.py        — /feature: file GitHub feature requests (optional, requires GitHub App)
    email.py          — /email: search Fastmail email via JMAP (optional)
  tools/
    base.py           — Tool ABC, ToolRegistry, ToolExecutor
    models.py         — ToolCall, ToolResult, ToolDefinition, SearchResult
    search.py         — SearchTool: Perplexity text + Serper images (run in parallel)
    news.py           — NewsTool: TheNewsAPI.com client (optional, requires NEWS_API_KEY)
    fetch_news.py     — FetchNewsTool: tool wrapper for NewsTool (used by chat + thinking)
    search_emails.py  — SearchEmailsTool (Fastmail JMAP)
    read_emails.py    — ReadEmailTool (Fastmail JMAP)
  jmap/
    client.py         — JmapClient: Fastmail JMAP API client (httpx)
    models.py         — JmapSession, EmailAddress, EmailSummary, EmailDetail
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
    entity_store.py   — EntityStore: get, get_or_create, get_for_user, embeddings
    fact_store.py     — FactStore: add, get_for_entity, embeddings
    message_store.py  — MessageStore: log_message, log_prompt, log_command, threads
    search_store.py   — SearchStore: log, get, mark_extracted
    thought_store.py  — ThoughtStore: inner monologue persistence
    history_store.py  — HistoryStore: conversation topic summaries
    user_store.py     — UserStore: get_info, save_info, mute/unmute
    models.py         — SQLModel tables (see Data Model section)
    migrate.py        — Migration runner: file discovery, tracking table, validation
    migrations/       — Numbered migration files (0001–0037)
  ollama/
    client.py         — OllamaClient: wraps official ollama SDK async client
    models.py         — ChatResponse, ChatResponseMessage
    embeddings.py     — Embedding utilities (serialize, deserialize, find_similar, build_entity_embed_text, token_containment_ratio)
  tests/
    conftest.py       — Pytest fixtures for mocks and test config
    test_embeddings.py, test_similarity.py, test_periodic_schedule.py, test_scheduler.py
    mocks/
      signal_server.py  — Mock Signal WebSocket + REST server (aiohttp)
      ollama_patches.py — Ollama SDK monkeypatch (MockOllamaAsyncClient)
      search_patches.py — Perplexity + Serper image search monkeypatches
    agents/           — Per-agent integration tests
      test_message.py, test_extraction.py, test_thinking.py, test_agentic_loop.py
    channels/         — Channel integration tests
      test_signal_channel.py, test_signal_reactions.py, test_signal_vision.py,
      test_signal_formatting.py, test_startup_announcement.py
    commands/         — Per-command tests
      test_commands.py, test_config.py, test_debug.py, test_draw.py, test_email.py,
      test_feature.py, test_memory.py, test_mute.py, test_forget.py,
      test_schedule.py, test_bug.py, test_system.py, test_test_mode.py
    database/         — Migration validation tests
      test_migrations.py
    jmap/             — JMAP client tests
      test_client.py
    tools/            — Tool tests
      test_search_redaction.py, test_tool_timeout.py, test_tool_not_found.py,
      test_missing_tool_params.py, test_tool_reasoning.py
Dockerfile            — Python 3.12-slim
pyproject.toml        — Dependencies and project metadata
```

## Agent Architecture

### Agent Base Class (`agents/base.py`)
The base `Agent` class implements the core agentic loop:
- Builds message context with system prompt and history
- Calls Ollama with available tools
- Executes tool calls via `ToolExecutor` with parameter validation
- Handles duplicate tool call prevention
- Appends source URLs to responses when model omits them

### Shared Ollama Client Instances

All OllamaClient instances are created centrally in `Penny.__init__()` and shared across agents and commands:

- `model_client`: Text model for all agents and commands
- `vision_model_client`: Optional vision model for image understanding
- `embedding_model_client`: Optional embedding model for semantic validation, dedup, and entity context injection
- `image_model_client`: Optional image generation model for `/draw` command

### Specialized Agents

**ChatAgent** (`agents/chat.py`)
- Handles incoming user messages with tools (search, news)
- Prepares thread context from quoted messages
- Returns response with parent_id for thread linking
- Vision captioning: when images are present and vision model is configured, captions the image first, then forwards a combined prompt to the foreground model

**ExtractionPipeline** (`agents/extraction.py`)
- Unified knowledge pipeline that runs as a single scheduled background task
- Processes in strict priority order (three phases per execution):
  1. **User messages** (highest priority): freshest signals, processed first
  2. **Search logs** (drain backlog): entity/fact extraction from search results
  3. **Embedding backfill**: backfills missing embeddings for facts and entities
- **Search log extraction**: Two-pass entity/fact extraction (identify entities → extract facts per entity) from search results. Checks `trigger` field to determine mode — full extraction (user-triggered, creates entities with validation) vs known-only (penny-triggered, facts only)
- **Post-fact semantic pruning**: After fact extraction, LLM checks if extracted facts are semantically relevant to the entity; removes irrelevant facts
- **Entity validation**: New entity candidates pass structural filter (word count, LLM artifacts, URLs, numbers, dates, locations) then semantic filter (embedding similarity to query, threshold ~0.50)
- **Insertion-time dedup**: Before creating a new entity, checks all existing entities using dual-threshold detection — token containment ratio (TCR >= 0.60) as fast lexical pre-filter, then embedding cosine similarity (>= 0.85) for confirmation. Both must pass. Routes to existing entity instead of creating a duplicate
- **Message extraction**: Extracts entities/facts from user messages
- Pre-filters messages before LLM calls: skips short messages (< 20 chars) and slash commands
- Fact dedup: normalized string match (fast) then embedding similarity (paraphrase detection, threshold=0.85)
- Facts track provenance via `source_search_log_id` or `source_message_id`

**ThinkingAgent** (`agents/thinking.py`)
- Continuous inner monologue loop — Penny's autonomous conscious mind
- Runs on a PeriodicSchedule (after ExtractionPipeline in priority order)
- Each cycle picks a random seed topic from conversation history to focus on
- Thinks out loud to itself using tools (search, news), accumulates reasoning
- At the end of each cycle, summarizes the monologue and stores it as a thought via ThoughtStore
- Stored thought summaries bleed into chat context, giving Penny continuity of inner reasoning

**HistoryAgent** (`agents/history.py`)
- Background worker that compacts daily conversations into topic summaries
- Runs on a PeriodicSchedule (lowest priority — after thinking)
- For each user: summarizes today's messages (midnight to now) via upsert (rolling update)
- Backfills completed past days that lack history entries
- Stored summaries used as seed topics for ThinkingAgent and as context for ChatAgent

**ScheduleExecutor** (`scheduler/schedule_runner.py`)
- Background task: runs user-created cron-based scheduled tasks
- Checks every 60 seconds for due schedules (based on user timezone)
- Executes the schedule's prompt text via the agentic loop
- Sends results to the user via channel

## Scheduler System

The `scheduler/` module manages background tasks:

### BackgroundScheduler (`scheduler/base.py`)
- Runs tasks in priority order (schedule executor → extraction → thinking → history)
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
- Used for the knowledge pipeline, thinking agent, and history agent (default: 10s, near-continuous while idle)
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

### SignalChannel (`channels/signal/channel.py`)
- WebSocket connection for receiving messages
- REST API for sending messages and typing indicators
- Handles quote-reply thread reconstruction

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
- **/memory** (`memory.py`): Browse and manage Penny's knowledge base. `/memory` lists all entities (with fact count); `/memory <number>` shows entity details and facts
- **/forget** (`forget.py`): Remove entities or facts from knowledge base
- **/schedule** (`schedule.py`): Create and list recurring cron-based background tasks (uses LLM to parse natural language timing)
- **/unschedule** (`unschedule.py`): Delete a scheduled task. `/unschedule` shows numbered list; `/unschedule <N>` deletes
- **/test** (`test.py`): Enters isolated test mode — creates a separate DB and fresh agents for testing without affecting production data. Exit with `/test stop`

### Conditional Commands (registered based on config)
- **/draw** (`draw.py`): Generate images via Ollama image model (requires `OLLAMA_IMAGE_MODEL`)
- **/bug** (`bug.py`): File a bug report on GitHub (requires GitHub App config)
- **/feature** (`feature.py`): File a feature request on GitHub (requires GitHub App config)
- **/email** (`email.py`): Search Fastmail email via JMAP (requires `FASTMAIL_API_TOKEN`)

### Runtime Configuration
- `/config` reads and writes to a `RuntimeConfig` table in SQLite (migration `0002_add_runtime_config_table.py`)
- `ConfigParam` definitions in `config_params.py` declare runtime-configurable settings with types and validation
- Three-tier lookup chain: DB override → env override → ConfigParam.default
- Config values are read on each use (not cached), so changes take effect immediately
- Categories: extraction thresholds, entity dedup settings, scheduling intervals, and more

## Knowledge System

Penny learns about things the user cares about by extracting entities and facts from conversations and searches. The system is built on two core principles:

1. **Entity creation is user-gated** — only user-triggered messages and searches create new entities
2. **Fact extraction is universal** — any search result can produce facts about known entities

### Data Model

- **Entity** (`database/models.py`): Named things Penny knows about (products, people, places). Has optional embedding for similarity search
- **Fact**: Individual facts with full provenance — tracks `source_search_log_id` or `source_message_id`, plus `learned_at` timestamp
- **Thought** (`database/models.py`): Inner monologue entries from the ThinkingAgent. Has `summary` (condensed reasoning), `content` (full monologue), `seed_topic`, and `sender` (user context)
- **ConversationHistory** (`database/models.py`): Daily conversation topic summaries. Has `sender`, `date`, `topics` (comma-separated), `summary`

### Search Trigger Tracking

Every SearchLog has a `trigger` field determining extraction behavior:

| Trigger | New entities? | New facts? |
|---|---|---|
| `user_message` | Yes | Yes |

### Two-Mode Extraction

The ExtractionPipeline checks each SearchLog's trigger to determine mode:
- **Full mode** (user-triggered): Identifies new AND known entities, validates candidates before creation
- **Known-only mode** (penny-triggered): Only matches against known entities, never creates new ones

### Entity Validation

New entity candidates pass through three filters before creation:
1. **Structural filter** (deterministic): Rejects names > 8 words, LLM output artifacts, URLs, markdown, bare numbers, dates, locations
2. **Semantic filter** (embedding-based): Rejects candidates with low cosine similarity to the triggering query (threshold ~0.50)
3. **Post-fact pruning** (LLM-based): After fact extraction, checks if facts are semantically relevant to the entity

### Entity Dedup (Insertion-Time)

Before creating a new entity, checks all existing entities using dual-threshold detection:
- **Token containment ratio (TCR >= 0.60)**: Fast lexical pre-filter
- **Embedding cosine similarity (>= 0.85)**: Confirmation via paraphrase detection
- Both must pass; routes to existing entity instead of creating a duplicate

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

- **Always search**: System prompt forces search on every message — no hallucinated answers
- **One search per message**: System prompt tells model it only gets one search, so it combines everything into a single comprehensive query
- **Parallel search + images**: Single `SearchTool` runs Perplexity (text) and Serper (images) concurrently via `asyncio.gather`, image failures degrade gracefully to text-only. `skip_images` flag disables image search
- **URL extraction**: URLs extracted from Perplexity results, appended as Sources list so the model can pick the most relevant one
- **URL fallback**: If the model's final response doesn't contain any URL, the agent appends the first source URL
- **Duplicate tool blocking**: Agent tracks called tools per message to prevent LLM tool-call loops
- **Tool parameter validation**: Tool parameters validated before execution; non-existent tools return clear error messages
- **Specialized agents**: Each task type (chat, extraction, thinking, history) has its own agent subclass
- **Priority scheduling**: Schedule executor → extraction → thinking → history (agents with no work are skipped each tick)
- **Always-run schedules**: User-created schedules run regardless of idle state; knowledge pipeline waits for idle
- **Global idle threshold**: Single configurable idle time (default: 60s) controls when idle-dependent tasks become eligible
- **Background cancellation**: Foreground message processing cancels active background tasks (`task.cancel()`) to free Ollama immediately; cancelled work is idempotent and retried next cycle
- **Commands don't interrupt background**: Slash commands run cooperatively without cancelling the active background task
- **Vision captioning**: When images are present and `OLLAMA_VISION_MODEL` is configured, the vision model captions the image first with a vision-specific system prompt, then a combined prompt is forwarded to the foreground model. Search tools are disabled for image messages
- **Channel abstraction**: Signal and Discord share the same interface; easy to add more platforms
- **Async throughout**: asyncio, httpx.AsyncClient, ollama.AsyncClient, discord.py
- **Host networking**: Docker container uses --network host for simplicity (all services on localhost)
- **Pydantic everywhere**: All external data validated with Pydantic models
- **Table-to-bullets**: Markdown tables converted to bullet points in Python (saves model tokens vs. prompting "no tables")
- **Normal casing**: All user-facing strings (status messages, error messages, acknowledgments) use standard sentence casing — not all lowercase

## Dependencies

- `websockets`, `httpx`, `python-dotenv`, `pydantic`, `sqlmodel`, `ollama`, `perplexityai`, `discord.py`, `psutil`, `dateparser`, `timezonefinder`, `geopy`, `pytz`, `croniter`, `PyJWT`
- Dev: `ruff` (lint/format), `ty` (type check), `pytest`, `pytest-asyncio`, `aiohttp` (mock Signal server)
- Python 3.12+

## Database Migrations

File-based migration system in `database/migrations/` (currently 0001–0037):
- Each migration is a numbered Python file (e.g., `0001_add_reaction_fields.py`) with a `def up(conn)` function
- Two types: **schema** (DDL — ALTER TABLE, CREATE INDEX) and **data** (DML — UPDATE, backfills), both use `up()`
- Runner in `database/migrate.py` discovers files, tracks applied migrations in `_migrations` table
- Runs on startup before `create_tables()` in `penny.py`
- `make migrate-test`: copies production DB, applies migrations to copy, reports success/failure
- `make migrate-validate`: checks for duplicate migration number prefixes (also runs in `make check`)
- Rebase-only policy: if two PRs create the same migration number, the second must rebase and renumber
- Run standalone: `python -m penny.database.migrate [--test] [--validate] [db_path]`

Notable migrations:
- 0007: `Schedule` table for user-created recurring tasks
- 0008: Drop `parent_summary` (removed SummarizeAgent)
- 0010–0011: `ResearchTask` and `ResearchIteration` tables (deprecated, dropped by 0018)
- 0012: `Entity` and `entity_extraction_cursor` tables for entity knowledge base
- 0013: `entity_search_log` join table (replaces cursor; tracks entity-to-search provenance)
- 0014–0016: Facts restructure, embedding columns, engagement table (knowledge system phases 1–3)
- 0017: `source_message_id` on `fact` table (message-sourced fact provenance)
- 0018: Drop `research_tasks` and `research_iterations` tables (deprecated research system)
- 0019: `LearnPrompt` table + `trigger`/`learn_prompt_id` columns on `SearchLog` (knowledge system v2)
- 0020: `notified_at` on `fact` table (notification decoupling)
- 0021: Drop `fact_last_verified` column (fact verification deprecated)
- 0022: Drop `preference` table (organic engagement replaces explicit preferences)
- 0023: `announced_at` on `LearnPrompt` table (learn completion notifications)
- 0028: `Event` and `EventEntity` tables (knowledge system v4 — time-aware events)
- 0029: `FollowPrompt` table (ongoing monitoring subscriptions for event system)
- 0032: `heat` and `heat_cooldown` columns on `entity` table (thermodynamic interest scoring)
- 0033: `heat_decayed_at` and `heat_cooldown_until` columns on `entity` table, drops deprecated `heat_cooldown` (time-based decay)
- 0034: `Thought` table for ThinkingAgent inner monologue persistence
- 0035: `recipient` column on `MessageLog` table
- 0036: `ConversationHistory` table for daily topic summaries
- 0037: Drop `learnprompt`, `followprompt`, `event`, `evententity` tables; remove `learn_prompt_id` from `searchlog`

## Extending

- **New tool**: Subclass `Tool` in tools/, implement `name`, `description`, `parameters`, `async execute()`, add to agent's tool list in penny.py
- **New channel**: Implement `MessageChannel` ABC, create models, add to `create_channel()` factory
- **New agent type**: Subclass `Agent`, implement `execute()` for background tasks or custom `handle()` for message processing
- **New command**: Subclass `Command` in commands/, implement `name`, `description`, `execute()`, register in `create_command_registry()`
- **New schedule type**: Subclass `Schedule`, implement `should_run()`, `reset()`, `mark_complete()`
- **New LLM**: Match `OllamaClient` interface (`async chat()`, `async generate()`)

## Test Infrastructure

Strongly prefer end-to-end integration tests over unit tests. Test through public entry points with mocks for external services. Prefer folding new assertions into existing tests over adding new test functions — only add a new test when no existing test covers the relevant code path.

**Mocks** (in `tests/mocks/`):
- `MockSignalServer`: WebSocket + REST server using aiohttp, captures outgoing messages and typing events
- `MockOllamaAsyncClient`: Monkeypatches `ollama.AsyncClient`, configurable responses via `set_default_flow()` or `set_response_handler()`
- `MockPerplexity`: Monkeypatches Perplexity SDK; `search_image` mocked via AsyncMock

**Fixtures** (in `tests/conftest.py`):
- `TEST_SENDER`: Standard test phone number constant
- `signal_server`: Starts mock Signal server on random port
- `mock_ollama`: Patches Ollama SDK with configurable responses
- `make_config`: Factory for creating test configs with custom overrides
- `running_penny`: Async context manager for running Penny with cleanup (uses WebSocket detection, not sleep)
- `setup_ollama_flow`: Factory to configure mock Ollama for message + background task flow
- `wait_until(condition, timeout, interval)`: Polls a condition every 50ms until true or timeout (10s default)

**Test Timing** — never use `asyncio.sleep(N)` in tests:
- Use `wait_until(lambda: <condition>)` to poll for expected side effects (DB state, message count, etc.)
- `scheduler_tick_interval` is set to 0.05s in test config (vs 1.0s production) so scheduler-dependent tests complete quickly
- `running_penny` detects WebSocket connection via `signal_server._websockets` instead of sleeping
- For negative assertions (nothing should happen), verify immediately — don't sleep to "make sure"

**Test Flow**:
1. Start mock Signal server (random port)
2. Monkeypatch Ollama and search SDKs
3. Create Penny with test config pointing to Signal mock
4. Push message through mock Signal WebSocket
5. `wait_until` the expected side effect (outgoing message, DB change, etc.)
6. Assert on captured messages, Ollama requests, DB state

**Performance**: Test suite runs in ~30s (`scheduler_tick_interval` set to 0.05s in tests)
