# Penny

A local-first AI agent that communicates via Signal and runs entirely on your machine with tool-based task management and autonomous background processing.

**Author:** Jared Lockhart

## Overview

Penny is a personal AI agent built with simplicity and privacy in mind. It runs locally, uses open-source models via Ollama, and communicates through Signal for a secure, familiar interface.

**How it works:**

At its core, Penny is a communication loop between you (via Signal) and an LLM (via Ollama). There are two main loops:

1. **Chat Loop**: Immediate back-and-forth conversation - you send a message, Penny responds
2. **Task Loop**: Background processing - Penny works on deferred tasks during idle time

The agent builds conversation context (history + memories) and uses an agentic loop with tool calling to let the model take actions (store memories, search the web, manage tasks, etc.). But fundamentally, it's just facilitating conversation between you and the model.

**Key Features:**
- **Tool-Based Architecture**: Separate tool registries for message handling (store_memory, create_task) and task processing (with search, time, task management)
- **Background Task Processing**: Autonomously works on deferred tasks during idle periods
- **Automatic History Compactification**: Summarizes long conversation histories to maintain context efficiently
- **Long-Term Memory**: Stores important facts and preferences across conversations
- **Agentic Loop**: Multi-step reasoning with tool calling (up to 5 steps for messages, 10 for tasks)

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
        │  - Task       │
        │    Processor  │
        │  - Agentic    │
        │    Loop       │
        │  - Tools      │
        │  - Memory     │
        └───────────────┘
```

### Agentic Architecture

Penny uses a **dual-context system** with separate tool registries:

**Message Handler (Immediate Responses)**
- Tools: `store_memory`, `create_task`
- Max steps: 5
- Purpose: Quick responses, memory storage, task creation

**Task Processor (Background Work)**
- Tools: `store_memory`, `get_current_time`, `list_tasks`, `complete_task`, `perplexity_search`
- Max steps: 10
- Purpose: Autonomous work on deferred tasks during idle time

**Background Task Processing:**
1. User sends message → classified as immediate question or deferred task
2. If task: acknowledge immediately, add to queue
3. After 5 seconds of idle time: pick up first pending task
4. Run agentic loop with full tool access
5. Complete task and send result to requester

**History Compactification:**
- After 30 seconds of idle time with no tasks (configurable)
- Retrieves last 500 messages (configurable)
- Uses Ollama to generate a summary (3-5 paragraphs)
- Stores summary as a message for future context injection
- Prevents repeated summarization with cooldown tracking

### Design Decisions

- **Host Services**: signal-cli-rest-api and Ollama run directly on host (easier debugging, no nested containers)
- **Containerized Agent**: Only the Python agent runs in Docker (simple, portable, reproducible)
- **Networking**: `--network host` for simplicity (all local, no security concerns)
- **Persistence**: SQLite on host filesystem via volume mount (survives container restarts)
- **Communication**: WebSocket for receiving (real-time), REST for sending (simple)
- **Parallel Execution**: Message listener and task processor run concurrently via asyncio.gather()

## Key Features

### Automatic History Compactification

Penny automatically manages conversation context to maintain long-term coherence without overwhelming the LLM's context window:

**How it works:**
1. After being idle for 30 seconds (configurable) with no pending tasks
2. Retrieves the last 500 messages (configurable) from the conversation
3. Sends them to Ollama with a summarization prompt
4. Stores the summary (3-5 paragraphs) as a special message: `[CONVERSATION_SUMMARY YYYY-MM-DD HH:MM]`
5. Summary gets included in future conversation context alongside recent messages

**Benefits:**
- Maintains context from hundreds of messages in just a few paragraphs
- Preserves key facts, preferences, and conversation flow
- Automatic cooldown prevents repeated summarization
- No manual intervention required

**Configuration:**
- `HISTORY_COMPACTION_LIMIT`: Messages to summarize (default: 500)
- `HISTORY_COMPACTION_IDLE_SECONDS`: Idle time before compacting (default: 30.0)
- `HISTORY_COMPACTION_MIN_MESSAGES`: Minimum messages needed (default: 10)

### Background Task Processing

Penny can defer complex work and handle it autonomously during idle periods:

**Task Flow:**
1. User sends a request (e.g., "look up the weather in Tokyo")
2. Penny acknowledges immediately ("I'll look up the weather in Tokyo for you")
3. Task is added to the queue with status `pending`
4. After 5 seconds of idle time, task processor picks it up
5. Runs agentic loop with full tool access (search, time, etc.)
6. Sends result back to user when complete

**Tool Separation:**
- **Message Handler**: Can only `store_memory` and `create_task` (quick, focused)
- **Task Processor**: Full tool access including `perplexity_search`, `get_current_time`, etc.

### Long-Term Memory

The `store_memory` tool allows Penny to remember facts across conversations:

**Examples:**
- "My name is Jared" → Stored and recalled in future conversations
- "I prefer metric units" → Applied to all future responses
- "I'm working on project X" → Context for task-related questions

**Memory Structure:**
- Content: The fact or preference
- Context: Additional details
- Importance: 1-10 scale (future: decay and prioritization)
- Timestamps: Created and last accessed

## Data Model

Penny stores three types of data in SQLite:

**Messages**: Complete conversation log
- Stores all incoming messages from Signal and outgoing responses from Penny
- Includes direction, sender, recipient, timestamp
- Multi-line responses are tracked with chunk indices
- Optional thinking traces from the model

**Memories**: Long-term knowledge
- Facts, preferences, and behavioral rules that persist across conversations
- Includes importance scoring (1-10 scale) for future prioritization
- Tracks when memories were created and last accessed

**Tasks**: Deferred work queue
- Background tasks waiting to be processed during idle time
- Tracks status (pending → in_progress → completed)
- Stores the requester, task description, and final result

## Setup & Running

### Prerequisites

1. **signal-cli-rest-api** running on host (port 8080)
2. **Ollama** running on host (port 11434)
3. Docker & Docker Compose installed

### Quick Start

```bash
# 1. Create .env file with your configuration
cp .env.example .env
# Edit .env with your settings

# 2. Start the agent
docker-compose up --build
```

### docker-compose.yml

```yaml
services:
  penny:
    build: .
    network_mode: host
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env
    restart: unless-stopped
```

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

# Optional: Perplexity API for web search
PERPLEXITY_API_KEY="your-api-key"

# Agent behavior (optional, defaults shown)
MESSAGE_MAX_STEPS=5
TASK_MAX_STEPS=10
IDLE_TIMEOUT_SECONDS=5.0
TASK_CHECK_INTERVAL=1.0
CONVERSATION_HISTORY_LIMIT=100
HISTORY_COMPACTION_LIMIT=500
HISTORY_COMPACTION_IDLE_SECONDS=30.0
HISTORY_COMPACTION_MIN_MESSAGES=10
```

**Configuration Options:**

**Required:**
- `SIGNAL_NUMBER`: Your registered Signal number
- `SIGNAL_API_URL`: signal-cli REST API endpoint (default: http://localhost:8080)
- `OLLAMA_API_URL`: Ollama API endpoint (default: http://localhost:11434)
- `OLLAMA_MODEL`: Model name to use (default: llama3.2)

**Optional:**
- `LOG_LEVEL`: Logging verbosity - DEBUG, INFO, WARNING, ERROR (default: INFO)
- `DB_PATH`: SQLite database location (default: ./data/agent.db)
- `LOG_FILE`: Log file location (default: ./data/penny.log)
- `PERPLEXITY_API_KEY`: API key for web search tool (optional)

**Agent Behavior:**
- `MESSAGE_MAX_STEPS`: Max agentic loop steps for immediate messages (default: 5)
- `TASK_MAX_STEPS`: Max agentic loop steps for background tasks (default: 10)
- `IDLE_TIMEOUT_SECONDS`: Idle time before processing tasks (default: 5.0)
- `TASK_CHECK_INTERVAL`: How often to check for tasks (default: 1.0)
- `CONVERSATION_HISTORY_LIMIT`: Recent messages for context (default: 100)
- `HISTORY_COMPACTION_LIMIT`: Messages to summarize during compaction (default: 500)
- `HISTORY_COMPACTION_IDLE_SECONDS`: Idle time before compacting history (default: 30.0)
- `HISTORY_COMPACTION_MIN_MESSAGES`: Minimum messages required to compact (default: 10)

## Development Roadmap

### Completed Features (v0.1)
- [x] Core agent loop with WebSocket message handling
- [x] SQLite message logging (incoming/outgoing with chunks)
- [x] Ollama integration with tool calling support
- [x] Conversation context (100 recent messages)
- [x] Docker containerization
- [x] Error handling and reconnection logic
- [x] Tool-based architecture with ToolRegistry
- [x] Agentic controller with multi-step reasoning
- [x] Background task processing system
- [x] Long-term memory storage (store_memory tool)
- [x] Perplexity search integration
- [x] Automatic history compactification
- [x] Dual-context system (message handler vs task processor)
- [x] Typing indicators for better UX

## Inspiration

Based on learnings from openclaw - built to be simpler, cleaner, and more maintainable.

## License

MIT
