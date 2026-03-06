# Context Injection Plan

Principle: only inject what the model needs for its exact next step. Keep it tight.

## Chat Entry Points

### 1. User message
- [x] Profile (name)
- [x] Entities (top 5 by similarity to message, max 5 facts each)
- [ ] History (last 7 days with dates — "Mar 5: topic1, topic2")
- [ ] Thoughts (1 most recently notified — not 10)
- [ ] ~~Dislikes~~ REMOVED — only used by thinking to steer
- [ ] Conversation history (messages since last rollup — not midnight)

### 2. Proactive: check-in
- [x] Profile (name)
- [ ] History (last 7 days with dates)
- [ ] ~~Dislikes~~ REMOVED
- [ ] Conversation history (since last rollup)
- No entities, no thoughts — just "hey what's up"

### 3. Proactive: news
- [x] Profile (name)
- [ ] ~~Dislikes~~ REMOVED
- [ ] Conversation history (since last rollup)
- No entities, no thoughts, no history

### 4. Proactive: thought response
- [x] Profile (name)
- [x] The single thought being shared (research report)
- [x] Entities (top 5 by similarity to thought content)
- [ ] ~~Dislikes~~ REMOVED
- [ ] Conversation history (since last rollup)
- No history bullets — the thought IS the content

## Thinking Entry Points

### 5. Seeded initial step
- [x] Profile (name)
- [ ] Thoughts (recent 10 — to avoid repetition)
- [ ] Dislikes (all — to steer away from)
- No entities (no anchor yet), no history (seed came from history)

### 6. Free thinking
- Nothing — explore freely (already correct)

### 7. News browsing
- Same as seeded (5) — profile, thoughts, dislikes

### 8. Continuation steps (system rebuilt mid-loop)
- [x] Profile (name)
- [x] Entities (top 5 by similarity to accumulated monologue)
- [ ] Thoughts (recent 10 — to avoid repetition)
- [ ] Dislikes (all — to steer away from)

## Changes Required

### History with dates
- Update `_build_history_context` to format as "Mar 5: topic1, topic2"
- Currently flat bullets with no temporal info

### Thought limit reduction
- Chat: cap at 1 most recently notified (currently 10)
- Thinking: cap at 10 (currently 50 via THOUGHT_CONTEXT_LIMIT override)

### Remove dislikes from chat
- Chat `get_context()` stops calling `_build_dislike_context()`
- Thinking keeps it (steers research away from disliked topics)

### Conversation history: since last rollup
- Currently: messages since midnight today
- Change to: messages since last history rollup's period_end
- Fallback: midnight if no rollup exists

### Proactive context differentiation
- Check-in: profile + history + conversation only
- News: profile + conversation only
- Thought: profile + thought + entities + conversation
