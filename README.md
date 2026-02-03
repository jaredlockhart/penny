# Penny

A local-first AI agent that communicates via Signal and runs entirely on your machine.

**Author:** Jared Lockhart

## Overview

Penny is a personal AI agent built with simplicity and privacy in mind. It runs locally, uses open-source models via Ollama, and communicates through Signal for a secure, familiar interface.

**How it works:**

You send a message on Signal. Penny searches the web via Perplexity and finds a relevant image via DuckDuckGo — both in parallel — then responds in a casual, lowercase style with the image attached. If you reply to one of Penny's messages, it rebuilds the conversation thread for context.

**Key Features:**
- **Perplexity Search**: Every response is grounded in a web search — Penny never answers from model knowledge alone
- **Image Attachments**: Every response includes a relevant image from DuckDuckGo, sent as a Signal attachment (degrades gracefully if unavailable)
- **Source URLs**: URLs extracted from Perplexity search results and annotations, presented as a Sources list so the model picks the most relevant one
- **Thread-Based Context**: Quote-reply to continue a conversation; Penny walks the message chain to rebuild history
- **Spontaneous Continuation**: Penny randomly continues idle conversations by searching for something new about the topic (configurable idle timeout and interval)
- **Thread Summarization**: Background task summarizes idle threads via Ollama and caches the summary for future context
- **Full Logging**: Every Ollama prompt, Perplexity search, and user/agent message is logged to SQLite
- **Agent Loop**: Multi-step reasoning with tool calling (up to 5 steps), with duplicate tool call protection
- **Search Result Cleaning**: Regex-based stripping of markdown and citations from Perplexity results before they reach the LLM
- **Retry on Failure**: Ollama client retries up to 3 times on transient errors (e.g. malformed tool call JSON)

## Architecture

### System Components

```
┌─────────────────────────────────────┐
│          HOST ENVIRONMENT            │
│                                      │
│  signal-cli-rest-api (json-rpc)     │
│  ├─ REST: localhost:8080            │
│  │  └─ POST /v2/send                │
│  └─ WebSocket: ws://localhost:8080  │
│     └─ /v1/receive/<number>         │
│                                      │
│  ollama                              │
│  └─ API: localhost:11434            │
│     └─ POST /api/chat                │
│                                      │
│  perplexity API (external)          │
│                                      │
│  ./data/agent.db (SQLite)           │
│                                      │
└──────────────┬───────────────────────┘
               │ --network host
        ┌──────▼────────┐
        │  CONTAINER    │
        │               │
        │  Penny Agent  │
        │  (Python)     │
        │               │
        │  - Message    │
        │    Listener   │
        │  - Agent      │
        │    Loop       │
        │  - Search     │
        │    (Perplexity│
        │    + DDG img) │
        │  - Spontaneous│
        │    Continuation│
        │  - Thread     │
        │    Summarizer │
        │  - SQLite     │
        │    Logging    │
        └───────────────┘
```

### Message Flow

1. User sends Signal message (or quote-replies to a previous response)
2. If quote-reply: look up the quoted message, walk the parent chain to build thread history
3. Log incoming message (linked to parent if replying)
4. Run agent loop: Ollama calls search tool, which runs Perplexity (text) and DuckDuckGo (images) in parallel. Search results include extracted source URLs for the model to reference.
5. Log outgoing message (linked to incoming)
6. Send response back via Signal with image attachment (if available)
7. Background: when idle, summarize threads and optionally continue dangling conversations

### Design Decisions

- **Host Services**: signal-cli-rest-api and Ollama run directly on host (easier debugging, no nested containers)
- **Containerized Agent**: Only the Python agent runs in Docker (simple, portable, reproducible)
- **Networking**: `--network host` for simplicity (all local, no security concerns)
- **Persistence**: SQLite on host filesystem via volume mount (survives container restarts)
- **Communication**: WebSocket for receiving (real-time), REST for sending (simple)
- **Thread History**: Parent-child message linking via quote-reply, not sliding window

## Data Model

Penny stores three types of log data in SQLite:

**PromptLog**: Every call to Ollama
- Model name, full message list (JSON), tool definitions (JSON), response (JSON)
- Thinking/reasoning trace (if model supports it)
- Call duration in milliseconds

**SearchLog**: Every Perplexity search (image searches are not logged separately)
- Query text, response text, call duration

**MessageLog**: Every user message and agent response
- Direction (incoming/outgoing), sender, content
- Parent ID (foreign key to self) for threading quote-replies
- Parent summary (cached thread summary for context reconstruction)

## Setup & Running

### Prerequisites

1. **signal-cli-rest-api** running on host (port 8080)
2. **Ollama** running on host (port 11434)
3. **Perplexity API key** (for web search)
4. Docker & Docker Compose installed

### Quick Start

```bash
# 1. Create .env file with your configuration
cp .env.example .env
# Edit .env with your settings

# 2. Start the agent
make up
```

### Make Commands

```bash
make up          # Build and start all services (foreground)
make kill        # Tear down containers and remove local images
make check       # Format check (read-only), lint, and typecheck
make fmt         # Format with ruff
make lint        # Lint with ruff
make fix         # Format + autofix lint issues
make typecheck   # Type check with ty
```

All dev tool commands run in temporary Docker containers via `docker-compose run --rm`, with source volume-mounted so changes write back to the host filesystem.

## Configuration

Configuration is managed via a `.env` file in the project root:

```bash
# .env
SIGNAL_NUMBER="+1234567890"
SIGNAL_API_URL="http://localhost:8080"
OLLAMA_API_URL="http://localhost:11434"
OLLAMA_MODEL="llama3.2"
LOG_LEVEL="INFO"
DB_PATH="./data/agent.db"
LOG_FILE="./data/penny.log"
PERPLEXITY_API_KEY="your-api-key"

# Agent behavior (optional, defaults shown)
MESSAGE_MAX_STEPS=5
SUMMARIZE_IDLE_SECONDS=30
CONTINUE_IDLE_SECONDS=1800
CONTINUE_MIN_SECONDS=1800
CONTINUE_MAX_SECONDS=10800
```

**Required:**
- `SIGNAL_NUMBER`: Your registered Signal number

**Optional (with defaults):**
- `SIGNAL_API_URL`: signal-cli REST API endpoint (default: http://localhost:8080)
- `OLLAMA_API_URL`: Ollama API endpoint (default: http://localhost:11434)
- `OLLAMA_MODEL`: Model name to use (default: llama3.2)
- `LOG_LEVEL`: Logging verbosity — DEBUG, INFO, WARNING, ERROR (default: INFO)
- `DB_PATH`: SQLite database location (default: /app/data/penny.db)
- `LOG_FILE`: Log file path (default: none)
- `PERPLEXITY_API_KEY`: API key for web search (without this, the agent has no tools)
- `MESSAGE_MAX_STEPS`: Max agent loop steps per message (default: 5)
- `SUMMARIZE_IDLE_SECONDS`: Idle time before summarizing threads (default: 30)
- `CONTINUE_IDLE_SECONDS`: Idle time before spontaneous continuation is eligible (default: 1800)
- `CONTINUE_MIN_SECONDS`: Minimum random delay between continuation attempts (default: 1800)
- `CONTINUE_MAX_SECONDS`: Maximum random delay between continuation attempts (default: 10800)

## Inspiration

Based on learnings from openclaw — built to be simpler, cleaner, and more maintainable.

## License

MIT
