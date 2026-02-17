# Knowledge System: Master Plan

## Goal

Penny learns what the user likes and is interested in, finds information about those things, proactively grows that knowledge over time, and uses it to have better conversations and surface relevant new information.

---

## Core Concept

Two parallel datasets grow together:

**World Knowledge** — entities and facts that exist objectively. "The KEF LS50 Meta is a bookshelf speaker. It costs $1,600. It uses a MAT driver." Sourced from searches, user teaching, and proactive research. Every fact tracks where it came from and when.

**User Interest Graph** — what the user cares about and how much. "User likes audio equipment (high confidence). User is interested in KEF LS50 Meta (very high — searched 3 times, used /learn, reacted positively to a message about it). User dislikes sports (medium — said so once)." Built from accumulated signals across all interactions.

These two datasets combine to drive everything: what Penny says in conversation, what it researches next, and what it proactively shares. High interest + thin knowledge = research priority. High interest + rich knowledge + time elapsed = check for news.

---

## User Entry Points

### 1. Message (primary interaction)

User sends a message. Penny responds — same as today, except:

- Before responding, retrieve relevant entities via embedding similarity against the message
- Inject known facts into the response prompt so Penny can reference what it already knows
- **Knowledge sufficiency check**: if existing facts are enough to answer the question, respond from memory without searching (faster, feels like Penny remembers). If not, search as usual but with better context.
- After responding, extract entities/facts/signals from both the search results and the user's message
- The message itself is a signal: entities mentioned get a low-strength positive signal

### 2. /like, /unlike, /dislike, /undislike (preference management)

Explicit preference manipulation:
- `/like espresso machines` — adds a like preference, high-strength positive signal
- `/dislike sports` — adds a dislike preference, high-strength negative signal
- `/unlike espresso machines` — removes the like
- `/undislike sports` — removes the dislike

These complement implicit preference extraction from messages and reactions with explicit user control.

### 3. /learn [topic] (active research)

Signals the system that this is an active area to explore:
- Creates or boosts the entity to high interest
- Research loop picks it up as top priority (high interest × low knowledge)
- As the research loop discovers information, it messages the user with findings
- Over subsequent cycles, continues filling gaps until knowledge is rich
- Then naturally transitions to monitoring mode (checks for news rather than building basics)

Multiple `/learn` targets queue naturally via priority scoring. No separate research mode — just the system's normal loop running at elevated priority.

### 4. /memory (knowledge browsing)

View and manage the knowledge store (existing, stays as-is):
- `/memory` — list all entities with fact counts
- `/memory [number]` — show entity details and facts
- `/memory [number] delete` — remove an entity (inverse of /learn)

### 5. /interests (interest graph visibility)

View what Penny thinks you care about:
- `/interests` — show ranked entities by computed interest score
- Displays: entity name, interest score, fact count, last activity
- Lets the user verify Penny's model matches reality and correct misunderstandings

### 6. Passive signals (no user action required)

- **Emoji reactions** to Penny's messages — positive reaction = positive signal for entities in that message, negative = negative signal. **Negative reaction on a proactive message = strong "stop" signal** that sharply drops that entity's research priority.
- **Conversation patterns** — repeated questions about the same entity, follow-up questions, sustained engagement accumulate as interest signals
- **Search patterns** — what the user asks Penny to search for reveals what they care about

---

## Data Model

### Facts as first-class objects

Today facts are a text blob on the entity. They become their own table:

- `id`, `entity_id`, `content` (the fact text)
- `source_url` — where the information came from
- `source_search_log_id` — which search produced it
- `learned_at` — when Penny first learned this
- `last_verified` — when this was last confirmed current
- `embedding` — vector embedding for semantic search

This enables per-fact provenance, per-fact staleness detection, per-fact semantic search, and per-fact deduplication via embeddings.

### EntitySearchLog replaced by simpler tracking

The current EntitySearchLog join table is replaced by:
- `extracted` boolean on SearchLog (marks whether extraction has processed it)
- Fact-level `source_search_log_id` (tracks provenance more granularly per fact, not per entity)

### Entities get embeddings

- `embedding` — vector embedding of the entity name + key facts
- Used for fuzzy matching ("those speakers" → "KEF LS50 Meta") and for computing preference-entity relationships

### Signal model

**Signal table** (new):
- `id`, `user`, `entity_id` (nullable), `preference_id` (nullable)
- `signal_type` — explicit_statement, emoji_reaction, search_initiated, follow_up_question, learn_command, message_mention, like_command, dislike_command
- `valence` — positive, negative, neutral
- `strength` — numeric weight
- `source_message_id` — which message produced this signal
- `created_at`

**Interest score** — computed from accumulated signals per entity per user: `sum(strength × recency_decay)`. Drives research priority and context injection ranking. Negative scores possible for disliked entities (suppresses them from research).

### Preference-entity relationships (computed, not stored)

No explicit join table. Preferences and entities both have embeddings — relationships are computed on the fly via cosine similarity. At Penny's scale (hundreds of entities, dozens of preferences), this is fast. When needing "which entities match this preference?", compute `find_similar(preference_embedding, entity_embeddings, top_k)`.

### Embedding generation

All embeddings generated by Ollama's embedding model (local, no API cost). Stored as binary blobs in SQLite. Similarity search via cosine distance computed in Python.

---

## Background Loops

Two background loops replace the current four systems (EntityExtractor, PreferenceAgent, Followup, Discovery). Research replaces the current /research system.

### Loop 1: Extraction

**Trigger**: New data enters the system (message, search result, /learn, /like).

**What it does**:
1. Extract entities and facts from content (search results AND user messages)
2. Track fact sources (URL, search log, timestamp)
3. Extract user signals — what does this interaction reveal about preferences?
4. Link signals to entities found in the same context
5. Match new entities against existing preferences via embedding similarity
6. Generate/update embeddings for new entities, facts, and preferences
7. Deduplicate facts using embedding similarity

**Message filtering**: Not every message warrants LLM extraction. Skip messages below a minimum length, pure commands, and common low-content patterns (single emoji, acknowledgments). This avoids burning Ollama cycles on noise.

**Replaces**: EntityExtractor + PreferenceAgent, unified into one pipeline.

### Loop 2: Research (adaptive enrichment + briefing)

**Trigger**: Periodic during idle. Picks the highest-priority target.

**Priority scoring**: `interest_score × (1 / fact_count) × staleness_factor`

**Adaptive behavior based on entity knowledge depth:**

| Entity State | Mode | Search Strategy | Messaging |
|---|---|---|---|
| Few facts (0-5) | Enrichment | Broad queries to fill gaps | Message on any substantial findings |
| Moderate facts (5-15) | Enrichment | Targeted queries for what's NOT known | Message on meaningful new info |
| Many facts (15+), stale | Briefing | "What's new since [date]" queries | Message only if genuinely novel |
| Many facts, recent | Skip | — | — |
| Negative interest | Skip | — | — |

This is one loop that naturally transitions per entity as knowledge accumulates. No explicit state management needed.

**What it does**:
1. Score all entities, pick the top candidate
2. Check fact count to determine mode
3. Build search query appropriate to mode
4. Run search via Perplexity
5. Extract candidate facts, compare against existing via embedding similarity
6. Store genuinely new facts with sources, update `last_verified` on confirmed facts
7. If substantial/novel findings → compose and send a message to the user
8. Update embeddings

**Replaces**: Followup (random, repetitive) + Discovery (unfocused, irrelevant) + /research (closed one-shot reports). All collapsed into one priority-driven adaptive loop.

---

## Signals Reference

| Signal | Source | Strength | Example |
|---|---|---|---|
| /learn command | User types /learn X | Very high (1.0) | `/learn kef ls50 meta` |
| /like command | User types /like X | High (0.8) | `/like audio equipment` |
| /dislike command | User types /dislike X | High negative (0.8) | `/dislike sports` |
| Explicit statement | "I love X" in message | High (0.7) | "I'm really into espresso" |
| Search initiated | User searches for X | Medium-high (0.6) | "search for LS50 Meta reviews" |
| Follow-up question | User asks more about X | Medium (0.5) | "what about the LS50's bass response?" |
| Negative reaction on proactive msg | 👎 on unsolicited message | High negative (0.8) | "stop telling me about this" |
| Positive reaction on proactive msg | 👍 on unsolicited message | Medium (0.5) | "more like this" |
| Positive reaction on normal msg | 👍 on user-initiated message | Medium-low (0.3) | casual approval |
| Mentioned in passing | X appears but isn't focus | Low (0.2) | "I was listening on my LS50s when..." |

---

## Deprecation Path

| Current System | Replaced By | Rationale |
|---|---|---|
| Followup agent | Research loop (enrichment mode) | Followup picks random old conversations and repeats info. Research loop targets the most valuable knowledge gaps. |
| Discovery agent | Research loop (briefing mode) | Discovery picks random broad likes and produces irrelevant results. Research loop monitors specific entities for genuine novelty. |
| Research system (/research) | /learn + Research loop | Research is a closed one-shot report. /learn signals ongoing interest; research loop incrementally builds knowledge and reports findings. |
| PreferenceAgent | Extraction loop | Preference extraction becomes part of unified extraction. |
| EntityExtractor | Extraction loop | Entity extraction expands to process messages (not just search results) and generate embeddings. |
| EntityCleaner | Stays (unchanged) | Duplicate merging still needed. Eventually updated to use embeddings for better detection. |

---

## Implementation Phases

### Phase 1 — Data Foundation

- [ ] #282 — Restructure facts into dedicated table with source tracking + replace EntitySearchLog
- [ ] #283 — Embedding infrastructure: Ollama integration, storage, and similarity search
- [ ] #284 — Add embeddings to entities, facts, and preferences (blocked by #282, #283)

### Phase 2 — Entity Context Injection

- [ ] #285 — Entity context injection + knowledge sufficiency check (blocked by #284)

### Phase 3 — Signal Model & Preferences

- [ ] #286 — Signal model: track user interest signals and compute interest scores (blocked by #282)
- [ ] #287 — Preference commands (/like etc.) + /interests command (blocked by #286)
- [ ] #288 — Emoji reaction signal extraction + proactive message stop mechanism (blocked by #286, #284)

### Phase 4 — Unified Extraction

- [ ] #289 — Unified extraction loop with message filtering (blocked by #282, #284, #286)

### Phase 5 — Learn & Research

- [ ] #290 — /learn command and adaptive research loop (blocked by #286, #289)

### Phase 6 — Deprecate Old Systems

- [ ] #292 — Deprecate followup, discovery, research, standalone extractors (blocked by #289, #290)

---

## Dependency Graph

```
Phase 1 — Data Foundation
  #282  Facts table + replace EntitySearchLog ─────────┐
  #283  Embedding infrastructure ─────────────────────┐│
                                                      ↓↓
  #284  Embeddings on entities/facts/preferences ─────┤
                                                      │
Phase 2                                               │
  #285  Entity context injection + sufficiency check ←┘
                                                      
Phase 3                                               
  #286  Signal model & interest scoring ←── #282      
  #287  Preference commands + /interests ←── #286     
  #288  Emoji reaction signals + stop mechanism ←── #286, #284

Phase 4
  #289  Unified extraction loop + message filtering ←── #282, #284, #286

Phase 5
  #290  /learn + adaptive research loop ←── #286, #289

Phase 6
  #292  Deprecate old systems ←── #289, #290
```

### Parallelism

- **#282 and #283** can start immediately and in parallel (no dependencies)
- **Phase 2 and Phase 3** can proceed in parallel once Phase 1 is done (#285 needs #284; #286 needs #282)
- **#287 and #288** can proceed in parallel once #286 is done
- **Phases 4-6** are sequential

### Design Decisions

1. **One research loop, not two** — Enrichment and briefing are the same process with different parameters. The entity's knowledge depth determines the behavior (broad research vs novelty checking). One loop adapts naturally.
2. **No entity-preference join table** — Relationships computed via embedding similarity on the fly. Avoids storage/maintenance overhead at Penny's scale.
3. **Processed flag replaces EntitySearchLog** — Simpler extraction tracking. Fact-level source tracking handles provenance.
4. **Knowledge sufficiency check before search** — Penny responds from memory when it can, searches when it can't. Makes knowledge tangible.
5. **Message filtering in extraction** — Not every message needs LLM processing. Lightweight pre-filter saves compute.
6. **Thumbs-down as stop mechanism** — Negative reaction on proactive messages is the user's way to say "enough about this." High-strength negative signal sharply drops priority.
7. **Interest graph visibility** — /interests lets the user see and verify what Penny thinks they care about.
