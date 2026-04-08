# Penny PR Review Guide

A comprehensive checklist for reviewing pull requests against the project's established patterns, conventions, and hard-won lessons. Every rule here comes from CLAUDE.md files or feedback from production incidents.

---

## 1. Code Style

### Pydantic for All Structured Data
- [ ] All structured data (API payloads, config, internal messages) uses Pydantic models — no raw dicts
- [ ] Every `Tool.execute(**kwargs)` validates through a Pydantic args model (e.g., `SearchArgs(**kwargs)`) as its first line
- [ ] Tool results use structured Pydantic models where applicable

### Constants and Naming
- [ ] All string literals in logic are defined as constants or enums — no magic strings
- [ ] Variable names are fully spelled out — no abbreviations (`message` not `msg`, `config` not `cfg`, `format_args` not `fmt`). Standard short names (`i`, `n`, `db`) in tight loops or established domain terms are fine
- [ ] f-strings used everywhere — no string concatenation with `+`

### Method Size and Structure
- [ ] Every method is roughly 10-20 lines (hard max ~25). Long methods are decomposed into named steps via extraction — no new abstractions, just decomposition
- [ ] Every class has a summary method (after `__init__`) that composes calls to other methods, reading like a table of contents

### Imports
- [ ] All imports are at the top of the file — no inline/inner imports inside functions or methods
- [ ] If a circular import exists, the fix is to move the shared type to a common location (e.g., `base.py`), not to defer the import
- [ ] `TYPE_CHECKING` guards are only used for type-only imports that would cause real circular dependency at runtime

### Dead Code
- [ ] No unused constants, variables, methods, or imports left behind after changes
- [ ] Follow the chain — if removing a method, also remove constants it was the only consumer of

### Optional Values
- [ ] Optional fields use `None` (`str | None = None`), never empty string defaults (`""`)
- [ ] `""` serializes as empty string over the wire, which is NOT nullish in JS/TS — `"" ?? fallback` evaluates to `""`, breaking null-coalescing

---

## 2. Database

### Schema and Foreign Keys
- [ ] All relationships use proper FK references (`preference_id INTEGER REFERENCES preference(id)`) — never denormalize by storing copies of data from another table
- [ ] New tables follow the SQLModel pattern in `database/models.py`

### Ordering
- [ ] Datetime columns (`created_at`, `timestamp`, `learned_at`, etc.) used for recency ordering in all queries
- [ ] Auto-increment `id` columns are NEVER used to infer chronological order — IDs are for joins and lookups only

### Store Pattern
- [ ] Database access goes through domain-specific store classes (`db.messages`, `db.preferences`, `db.thoughts`, etc.)
- [ ] The `Database` class is a thin facade — no business logic, just creates and exposes stores
- [ ] Access pattern: `self.db.messages.log_message(...)`, NOT `self.db.log_message(...)`

### Migrations
- [ ] New migrations are numbered sequentially in `database/migrations/`
- [ ] Migration files have a `def up(conn)` function
- [ ] No duplicate migration number prefixes (enforced by `make migrate-validate`)
- [ ] Schema migrations use DDL (ALTER TABLE, CREATE INDEX); data migrations use DML (UPDATE, backfills)

### Production Data
- [ ] **NEVER** modify production DB (runtime_config, preferences, thoughts, etc.) without explicit user approval
- [ ] Don't question or investigate whether runtime_config values are "taking effect" — if a value exists in the DB, the user put it there intentionally

---

## 3. Architecture and Design

### Python-Space Over Model-Space
- [ ] Deterministic actions (posting comments, creating labels, validating output) are handled in Python code, not delegated to the LLM
- [ ] Model-space is reserved for tasks that genuinely need reasoning (writing specs, analyzing code, generating responses)

### Pass Parameters, Don't Swap State
- [ ] No temporary swapping of instance state (e.g., `self.db`) to change behavior
- [ ] Dependencies are passed as parameters through the call chain

### Template Method Over Conditionals
- [ ] Multiple modes/variants use building blocks composed by each variant — no flags or if/else chains
- [ ] Examples: agent system prompts (building blocks like `_identity_section()`, `_profile_section()`), notification modes (`NotificationMode` subclasses)

### Compositional Pattern (Two-Layer)
- [ ] When multiple variants share a pipeline but differ in configuration: (1) prompt composition — each mode picks building blocks, (2) pipeline composition — each mode declares properties the orchestrator reads
- [ ] New modes = new class, no touching the pipeline
- [ ] Preferred over if/else chains, flag-based toggling, or deep class hierarchies

### No Client-Server Duplication
- [ ] Transformations exist in exactly one place — if the server does markdown-to-HTML, don't reimplement in the client
- [ ] Before writing a transformation, check if it already exists elsewhere

### No getattr Duck Typing
- [ ] Never use `getattr(obj, "method_name", None)` to check method availability
- [ ] Define methods on the base class as no-ops (with `return` to satisfy B027), override in subclasses that implement them

### No Shared State Races
- [ ] Async tasks receive dependencies as parameters — never reach into shared dicts/registries from async tasks expecting data to still be there
- [ ] Pass references directly when spawning tasks, don't have tasks look them up later

### Queues Over Locks
- [ ] For serializing async operations, prefer `asyncio.Queue` with a worker task over `asyncio.Lock`
- [ ] Pattern: callers enqueue `(data, result_future)` tuple and await the future; worker pops, processes, resolves

### Initialize at Startup
- [ ] Heavyweight setup (copying databases, creating resources) belongs at startup, not lazily inside handlers
- [ ] Static data captured at Docker build time via build args, not parsed at runtime

---

## 4. Error Handling

### Narrow Exceptions
- [ ] Catch the exact exception type expected (`asyncio.CancelledError`, `TimeoutError`, etc.)
- [ ] Never use `contextlib.suppress(Exception)` or `suppress(BaseException)` — they hide real bugs
- [ ] If multiple exceptions are possible, list them explicitly

### No Silent Fallbacks
- [ ] Never add `dict.get(key) or ""` or `or 0` or `or []` just to avoid dealing with a missing value — these mask bugs
- [ ] If a value might be absent, handle it with correct logic (e.g., `datetime.min` as a sortable sentinel) or let it raise
- [ ] Ask: does this default value make sense in downstream logic? If not, it's a bug mask

### No Silent Error Swallowing
- [ ] Never write catch blocks that silently swallow errors and fall through to a "fallback" implementation
- [ ] If a primary path fails, it must fail loudly (log the error, return an error state)
- [ ] Multiple strategies must be independently tested and selection must be explicit, not error-driven

### Verify Primary Path First
- [ ] Never write fallback/alternative code paths before verifying the primary path works with real output
- [ ] Pattern: (1) write primary path, (2) build and test against real input, (3) confirm correct output, (4) only then consider fallbacks

### Verify Imports
- [ ] When adding a new library dependency, verify the import resolves correctly (default vs named exports)
- [ ] Check the actual export shape — don't assume

---

## 5. Testing

### Test Invocation
- [ ] Tests run ONLY via `make fix check 2>&1 | tee /tmp/check-output.txt; echo "EXIT_CODE=$pipestatus[1]" >> /tmp/check-output.txt`
- [ ] Never use `make pytest`, `make check` alone, `docker compose run`, or any other variation
- [ ] Check EXIT_CODE first, then grep for FAILED or `error[` as needed

### All Changes Require Tests
- [ ] Every code change has corresponding test coverage — tests are part of the implementation, not a follow-up
- [ ] If all tests pass after behavior changes, that indicates a coverage gap — add tests that would fail if the change were reverted

### Integration Tests Preferred
- [ ] Test through public entry points (`agent.run()`, `has_work()`, full message flow) — not internal functions in isolation
- [ ] Unit tests only for pure utility functions with many edge cases (CODEOWNERS parsing, config loading)
- [ ] Mock at system boundaries (Ollama, Signal, GitHub CLI, Claude CLI) but let internal code execute end-to-end

### Test Organization
- [ ] Tests organized in this order: (1) comprehensive happy-path integration tests, (2) special success cases, (3) error/edge cases, (4) unit tests at the bottom
- [ ] Each primary variant/mode has a comprehensive test that exercises the entire code path

### Fold Into Existing Tests
- [ ] Prefer adding assertions to an existing test that covers the relevant code path over creating a new test function
- [ ] Only add a new test when no existing test covers the relevant code path

### Deterministic Tests
- [ ] All `random.random`, `random.choice`, and other random calls are monkeypatched in tests that assert on specific codepaths
- [ ] Even if a test "usually" takes the right path, a 1-in-3 chance of hitting the wrong branch causes flaky CI

### No Real Timers
- [ ] Use `wait_until(condition)` instead of `asyncio.sleep(N)` — poll for the expected side effect with a generous timeout
- [ ] For negative assertions (nothing should happen), verify immediately — don't sleep "to make sure"

### Cover All Codepaths
- [ ] Features work for ALL variants/modes — don't silently bail on a subset (e.g., seeded vs free thoughts, manual vs extracted preferences)
- [ ] If a variant genuinely needs different handling, call it out rather than silently skipping

### Exit Code Is Truth
- [ ] `make check` exit 0 = pass, exit 1 = fail. If it fails, something needs fixing
- [ ] Never attribute failures to "pre-existing issues" — main is always green (branch protection enforced)
- [ ] Never stash changes and test main to check if failures are pre-existing

---

## 6. Prompt Engineering

### System Prompt Structure
- [ ] Consistent `##` / `###` header hierarchy to delineate sections
- [ ] Standard structure: `## Identity` → `## Context` (with `###` subsections) → `## Instructions`
- [ ] Every agent overrides `_build_system_prompt(user)` composing from building blocks
- [ ] Tests assert on the exact full system prompt string to catch structural drift

### No Conflicting Instructions
- [ ] Read each instruction and verify it doesn't contradict another
- [ ] Thinking models are especially sensitive — contradictory signals cause extensive deliberation and empty output

### No Context Fixation
- [ ] Don't inject an agent's own previous outputs into its context unless there's a specific reason (like scoped dedup)
- [ ] Free-form previous outputs prime the model to revisit the same topics repeatedly

### Dry-Run Prompt Changes
- [ ] When modifying any LLM prompt, dry-run against real production prompt logs before deploying
- [ ] Test on 3+ diverse examples, not just the one that triggered the change
- [ ] Compare old output vs new output side-by-side

### Check Model Thinking
- [ ] When investigating prompt issues, check the `thinking` field in the `promptlog` table
- [ ] Search for keywords: "conflict", "but the instructions say", "contradicts", "wait,", "not sure if"

---

## 7. Forbidden Patterns

These patterns have each caused production bugs or wasted days of debugging. Flag them immediately.

### CRITICAL: Never Invent Arbitrary Truncations or Limits

This is the single most recurring source of production bugs in this project. It has caused **days** of cumulative debugging across multiple incidents. The pattern is always the same: a "reasonable" limit or default is added that nobody asked for, it silently discards or corrupts data, and the resulting bug is far harder to trace than the original problem would have been.

**The rule is absolute: never invent a value the user didn't ask for.**

- [ ] No new `max_tokens` values — adding `max_tokens: 600` to a call caused a model to stop mid-thought before writing the actual response, making it look like a model bug when it was self-inflicted truncation
- [ ] No new character limits — no `content[:500]`, no `text[:1000]`, no `summary[:200]`
- [ ] No new `.slice()` caps in TypeScript — no `results.slice(0, 5)`, no `items.slice(0, 10)`
- [ ] No new array/list length caps — no `items[:N]` unless the user specified N
- [ ] No new `max_results`, `max_items`, `max_length`, `max_chars` parameters with invented defaults
- [ ] No lossy data transformations that discard information "to be safe"
- [ ] No "reasonable defaults" for limits — what seems reasonable is wrong; it silently breaks things downstream and the resulting bug is always harder to diagnose than the original problem
- [ ] Existing limits that are already in the codebase (like `MAX_CHARS` in `extract_text.ts`) are there for a tested reason — don't remove those. The rule is about not **inventing new ones**
- [ ] Build the simplest thing first with NO truncation, ship it, and only address breakage when it actually happens — not preemptively
- [ ] If the user explicitly asks for a limit with a specific value, implement exactly that value — don't round it or "improve" it

**How to spot this in review:** Search the diff for any newly introduced numeric literal that caps, truncates, slices, or limits data. If the PR description doesn't say "user requested this limit of N," reject it. Common disguises:
- `[:N]` slicing on strings, lists, or query results
- `max_tokens`, `max_length`, `max_results` parameters
- `TRUNCATION_LIMIT`, `MAX_ITEMS`, `RESULT_CAP` constants
- `.slice(0, N)` in TypeScript
- `content[:N] + "..."` ellipsis truncation
- `if len(x) > N: x = x[:N]` guard clauses
- `LIMIT N` in SQL queries that didn't have one before
- Default parameter values that cap output (e.g., `def get_items(limit: int = 10)`)

### Never Use Arbitrary Thresholds
- [ ] Don't hardcode numeric thresholds ("after 3 tool calls, do X") when the behavior should be based on structural conditions ("are tools still available?")
- [ ] If a threshold is truly needed, derive it from the relevant config parameter

### Never Loop-and-Bail for Independent Items
- [ ] When processing a list of independent items, NEVER use a pattern where one item's failure blocks all others
- [ ] Each independent item must be processed on its own merits — no "pick the best one, try it, bail if it fails"

### No Monkeypatching Library Internals
- [ ] Only use publicly exported library APIs
- [ ] If a capability isn't publicly exported, implement it independently — don't hook into internals

### No Cloud Assets
- [ ] All assets (CSS, fonts, icons, JS libraries) bundled locally — never loaded from CDNs or external URLs
- [ ] Install via npm and reference from node_modules or copy into the project

### No Abbreviated Variable Names
- [ ] `message` not `msg`, `config` not `cfg`, `format_args` not `fmt`, `context` not `ctx`

### No Default Empty Values
- [ ] `None` for optional values, never `""`

### No getattr Duck Typing
- [ ] Define methods on base class, don't use `getattr(obj, "method", None)`

### No Silent Fallbacks
- [ ] `or ""`, `or 0`, `or []` as fallbacks are bug masks

---

## 8. Async Patterns

- [ ] `asyncio.Queue` with worker for serialization, not `asyncio.Lock`
- [ ] Pass dependencies directly to async tasks — don't fish from shared dicts
- [ ] Catch `asyncio.CancelledError` specifically (it's `BaseException` in Python 3.9+, not `Exception`)
- [ ] Background tasks must be idempotent — cancelled work stays in queues and retried next cycle

---

## 9. Browser Extension (TypeScript)

- [ ] Related content rendered in the same DOM container — not as separate siblings in scrollable lists
- [ ] No client-side reimplementation of server-side transformations
- [ ] All assets bundled locally, no CDN references
- [ ] Strict extractors for known page types — no generic fallback when the fallback produces garbage
- [ ] Known page types use a `ready` flag and are excluded from the generic fallback chain
- [ ] Verify library imports resolve (default vs named exports) before building on them

---

## 10. Git and Workflow

- [ ] All changes go through PRs — never push directly to `main`
- [ ] Feature branches created from latest `main` (`git checkout main && git pull origin main` first)
- [ ] Rebase on latest `main` before building, testing, or committing
- [ ] Check PR is still open before every `git push` (`gh pr list --head <branch> --state open`)
- [ ] Push branch before `gh pr create` (requires branch to exist on remote)
- [ ] Use `make token` for all GitHub operations: `GH_TOKEN=$(make token) gh pr create ...`
- [ ] CLAUDE.md and README.md updated when making significant changes (new features, architecture, config, API, directory structure)

---

## 11. Similarity and Embedding Code

- [ ] All similarity logic lives in the `similarity/` package — agents don't implement their own
- [ ] `similarity/embeddings.py`: Pure math (cosine similarity, TCR, serialize/deserialize)
- [ ] `penny/ollama/similarity.py`: Composed operations using embeddings + OllamaClient

---

## 12. Response Style

- [ ] Sentence case for all Penny response strings ("Okay, I'll learn more about {topic}")
- [ ] Markdown tables converted to bullet points in Python (saves model tokens)

---

## Quick Reference: Red Flags

If you see any of these in a PR, flag immediately:

| Pattern | Why It's Wrong |
|---|---|
| `or ""` / `or 0` / `or []` as fallback | Masks bugs, breaks null-coalescing in JS/TS |
| `except Exception:` / `suppress(Exception)` | Too broad, hides real bugs |
| `getattr(obj, "method", None)` | Duck typing bypasses type system |
| `max_tokens: 600` or any invented limit | Self-inflicted truncation, days of debugging |
| `content[:500]` or any `[:N]` slice on data | Silent data loss, nobody asked for this |
| `.slice(0, N)` in TypeScript | Same as above, JS edition |
| `LIMIT 10` added to a query that had none | Silently drops results |
| `def foo(limit: int = 10)` new default cap | Invented constraint, will bite later |
| `if len(x) > N: x = x[:N]` guard clause | Preemptive truncation that masks real issues |
| `asyncio.sleep(N)` in tests | Fragile timing, use `wait_until()` |
| `from foo import bar` inside a function | Hidden dependency, reorganize modules instead |
| Raw dict passed through system | Must use Pydantic model |
| `ORDER BY id DESC` for recency | Must use datetime column |
| `self.db.do_thing(...)` bypassing store | Must go through `self.db.store.do_thing(...)` |
| Inline `"magic string"` in logic | Must be a constant or enum |
| `msg`, `cfg`, `fmt`, `ctx` variable names | Spell it out fully |
| `try: ... except: <fallback code>` | Primary must fail loudly, not silently fall through |
| `field: str = ""` on Pydantic model | Use `str \| None = None` |
| CDN link for CSS/JS/fonts | Bundle locally |
