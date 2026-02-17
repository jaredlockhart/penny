# Knowledge System: Master Plan

## Goal

Penny learns what the user likes and is interested in, finds information about those things, proactively grows that knowledge over time, and uses it to have better conversations and surface relevant new information.

---

## Core Concept

Two parallel datasets grow together:

**World Knowledge** — entities and facts that exist objectively. "The KEF LS50 Meta is a bookshelf speaker. It costs $1,600. It uses a MAT driver." Sourced from searches, user teaching, and proactive research. Every fact tracks where it came from and when.

**User Interest Graph** — what the user cares about and how much. "User likes audio equipment (high confidence). User is interested in KEF LS50 Meta (very high — searched 3 times, used /learn, reacted positively to a message about it). User dislikes sports (medium — said so once)." Built from accumulated signals across all interactions.

These two datasets combine to drive everything: what Penny says in conversation, what it researches next, and what it proactively shares. High interest + thin knowledge = research priority. High interest + rich knowledge + time elapsed = briefing candidate.

---

## User Entry Points

### 1. Message (primary interaction)

User sends a message. Penny searches and responds — same as today, except:

- Before searching, retrieve relevant entities via embedding similarity against the message
- Inject known facts into the response prompt so Penny can reference what it already knows
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
- Enrichment loop picks it up as top priority (high interest × low knowledge)
- As enrichment discovers information, it messages the user with findings
- Over subsequent cycles, continues filling gaps until knowledge is rich
- Then naturally transitions to monitoring mode (briefing loop watches for updates)

Multiple `/learn` targets queue naturally via priority scoring. No separate research mode — just the system's normal loops running at elevated priority.

### 4. /memory (knowledge browsing)

View and manage the knowledge store (existing, stays as-is):
- `/memory` — list all entities with fact counts
- `/memory [number]` — show entity details and facts
- `/memory [number] delete` — remove an entity (inverse of /learn)

### 5. Passive signals (no user action required)

- **Emoji reactions** to Penny's messages — positive reaction = positive signal for entities in that message, negative = negative signal
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

### Entities get embeddings

- `embedding` — vector embedding of the entity name + key facts
- Used for fuzzy matching ("those speakers" → "KEF LS50 Meta") and for linking preferences to entities

### Richer preference/signal model

**Preference table** (revised):
- `id`, `user`, `topic`, `type` (like/interest/dislike/disinterest), `embedding`
- Links to entities via a join table

**Signal table** (new):
- `id`, `user`, `entity_id` (nullable), `preference_id` (nullable)
- `signal_type` — explicit_statement, emoji_reaction, search_initiated, follow_up_question, learn_command, message_mention, like_command, dislike_command
- `valence` — positive, negative, neutral
- `strength` — numeric weight
- `source_message_id` — which message produced this signal
- `created_at`

**Interest score** — computed from accumulated signals per entity per user: `sum(strength × recency_decay)`. Drives enrichment priority, briefing frequency, and context injection ranking.

### Entity-preference relationships

Join table linking preferences to entities:
- User "likes audio equipment" → linked to KEF LS50 Meta, Sennheiser HD650
- Created during extraction when a preference and entity appear in the same context

### Embedding generation

All embeddings generated by Ollama's embedding model (local, no API cost). Stored as binary blobs in SQLite. Similarity search via cosine distance computed in Python.

---

## Background Loops

Three background loops replace the current systems:

### Loop 1: Extraction

**Trigger**: New data enters the system (message, search result, reaction, /learn, /like).

**What it does**:
1. Extract entities and facts from content (search results AND user messages)
2. Track fact sources (URL, search log, timestamp)
3. Extract user signals — what does this interaction reveal about preferences?
4. Link signals to entities found in the same context
5. Link preferences to entities when they co-occur
6. Generate/update embeddings for new entities, facts, and preferences
7. Deduplicate facts using embedding similarity

**Replaces**: EntityExtractor + PreferenceAgent, unified into one pipeline.

### Loop 2: Enrichment

**Trigger**: Periodic during idle. Picks the highest-priority target.

**Priority scoring**: `interest_score × (1 / fact_count) × staleness_factor`

**What it does**:
1. Score all entities, pick the top candidate
2. Search for current information, focusing on what's NOT already in known facts
3. Extract new facts with sources
4. Update `last_verified` on confirmed existing facts
5. **Message the user with findings** when substantial new information is discovered
6. Update embeddings

**Replaces**: Followup (random, repetitive) + Discovery (unfocused, irrelevant) + /research (closed one-shot reports). All collapsed into one priority-driven loop.

### Loop 3: Briefing

**Trigger**: Periodic during idle, longer intervals. Only fires when genuinely new information exists.

**Targets**: High-interest entities with rich existing knowledge.

**What it does**:
1. Search for recent developments about candidate entities
2. Compare findings against known facts via embedding similarity — is this actually new?
3. If genuinely new and noteworthy → message the user
4. If nothing new → silently update `last_verified`, do nothing visible
5. New facts feed back into the entity store

**Replaces**: Discovery, with novelty filtering.

---

## Signals Reference

| Signal | Source | Strength | Example |
|---|---|---|---|
| /learn command | User types /learn X | Very high | `/learn kef ls50 meta` |
| /like command | User types /like X | High | `/like audio equipment` |
| /dislike command | User types /dislike X | High (negative) | `/dislike sports` |
| Explicit statement | "I love X" in message | High | "I'm really into espresso" |
| Search initiated | User searches for X | Medium-high | "search for LS50 Meta reviews" |
| Follow-up question | User asks more about X | Medium | "what about the LS50's bass response?" |
| Positive emoji reaction | Thumbs up on message about X | Medium-low | 👍 on a message mentioning the LS50 |
| Negative emoji reaction | Thumbs down on message about X | Medium-low (negative) | 👎 on a message about sports |
| Mentioned in passing | X appears but isn't the focus | Low | "I was listening on my LS50s when..." |

---

## Deprecation Path

| Current System | Replaced By | Rationale |
|---|---|---|
| Followup agent | Enrichment loop | Followup picks random old conversations and repeats info. Enrichment targets the most valuable knowledge gaps. |
| Discovery agent | Briefing loop | Discovery picks random broad likes and produces irrelevant results. Briefing monitors specific entities for genuine novelty. |
| Research system (/research) | /learn + Enrichment | Research is a closed one-shot report. /learn signals ongoing interest; enrichment incrementally builds knowledge and reports findings. |
| PreferenceAgent | Extraction loop | Preference extraction becomes part of unified extraction. |
| EntityExtractor | Extraction loop | Entity extraction expands to process messages (not just search results) and generate embeddings. |
| EntityCleaner | Stays (unchanged) | Duplicate merging still needed. Eventually updated to use embeddings for better detection. |

---

## Implementation Phases

### Phase 1 — Data Foundation

- [ ] #282 — Restructure facts into dedicated table with source tracking
- [ ] #283 — Embedding infrastructure: Ollama integration, storage, and similarity search
- [ ] #284 — Add embeddings to entities, facts, and preferences (blocked by #282, #283)

### Phase 2 — Entity Context Injection

- [ ] #285 — Entity context injection: use knowledge in responses (blocked by #284)

### Phase 3 — Signal Model & Preferences

- [ ] #286 — Signal model: track user interest signals and compute interest scores (blocked by #282)
- [ ] #287 — Preference commands: /like, /unlike, /dislike, /undislike (blocked by #286)
- [ ] #288 — Emoji reaction signal extraction (blocked by #286, #284)

### Phase 4 — Unified Extraction

- [ ] #289 — Unified extraction loop: merge entity and preference extraction (blocked by #282, #284, #286)

### Phase 5 — Learn & Enrichment

- [ ] #290 — /learn command and enrichment loop (blocked by #286, #289)

### Phase 6 — Briefing

- [ ] #291 — Briefing loop: novelty-filtered entity monitoring (blocked by #289, #290)

### Phase 7 — Deprecate Old Systems

- [ ] #292 — Deprecate followup, discovery, and research systems (blocked by #289, #290, #291)

---

## Dependency Graph

```
Phase 1 — Data Foundation
  #282  Facts table with source tracking ─────────────┐
  #283  Embedding infrastructure ─────────────────────┐│
                                                      ↓↓
  #284  Embeddings on entities/facts/preferences ─────┤
                                                      │
Phase 2                                               │
  #285  Entity context injection ←────────────────────┘
                                                      
Phase 3                                               
  #286  Signal model & interest scoring ←── #282      
  #287  Preference commands ←── #286                  
  #288  Emoji reaction signals ←── #286, #284         
                                                      
Phase 4                                               
  #289  Unified extraction loop ←── #282, #284, #286  
                                                      
Phase 5                                               
  #290  /learn + enrichment loop ←── #286, #289       
                                                      
Phase 6                                               
  #291  Briefing loop ←── #289, #290                  
                                                      
Phase 7                                               
  #292  Deprecate old systems ←── #289, #290, #291    
```

### Parallelism

- **#282 and #283** can start immediately and in parallel (no dependencies)
- **Phase 2 and Phase 3** can proceed in parallel once Phase 1 is done (#285 needs #284; #286 needs #282)
- **#287 and #288** can proceed in parallel once #286 is done
- **Phases 5-7** are sequential
