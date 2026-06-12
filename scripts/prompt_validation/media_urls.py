"""Media-URL suite — does the chat reply carry a page's Image: URL?

Browse writes ``Image: https://media.penny.local/<id>`` into page sections;
the channel turns that URL into an attached image at egress — but only if
the model includes it in its reply.  Production observation (June 12 2026,
Bravely Default run): three Image: URLs shown, none carried, while source
URLs are carried ~73% of the time (promptlog audit) because the prompts
explicitly demand them.  This suite measures whether prompt support closes
that gap.

CALIBRATION: ``source-url`` carry is the control metric.  Production
carries source URLs ~73%; if this fixture scores far below that, the
fixture is unrealistic and must be fixed before any media-URL conclusion
is drawn.  Getting it calibrated took three fixes, recorded here so they
aren't relearned: (1) the topic must sit inside the model's world
knowledge — it dismissed a fictional game as untrustworthy and answered
from priors; (2) search-result links must use REAL domains — the model
ignores ``.example`` links and browses domains it knows from priors;
(3) ``converse()`` used to invoke ``tool_result`` twice per call, so
stateful serve functions fed the model their second-invocation value
(a repeat-rejection) instead of the page — the model literally never
saw a fixture page, which invalidated every chat-carry number measured
before June 12 2026.

Results (gpt-oss:20b, 8 samples, calibrated fixture):
- baseline prompt:  source-url 6/8, media-url 0/8  (matches production:
  73% source carry, 0/3 media carry in the live promptlog)
- with the Image:-line paragraph in CONVERSATION_PROMPT's URL rules:
  source-url 7/8, media-url 7/8
"""
from __future__ import annotations

from scripts.prompt_validation._harness import (
    CaseResult,
    Harness,
    browse_tool,
    conversation_prompt,
    converse,
    load_seed_skills,
    load_tool,
    penny_identity,
    render_skills_recall,
    run_samples,
)

NAME = "media_urls"

_SEED, _ = load_seed_skills()

SYSTEM = f"{penny_identity()}\n\n{render_skills_recall(_SEED)}\n\n{conversation_prompt()}"

TOOLS = [browse_tool(), load_tool("ReadLatestTool")]

_PAGE_URL = "https://www.wholelattelove.com/blogs/articles/gaggia-classic-pro-review"
_MEDIA_URL = "https://media.penny.local/7"

_USER_MSG = "can you find me information about the gaggia classic pro espresso machine?"

# Step 1: a Kagi-style search result section (what a search query returns).
# Real domains — the model ignores .example links and browses from priors.
_SEARCH_SECTION = (
    "## browse search: gaggia classic pro\n"
    "## Search results\n\n"
    "### Gaggia Classic Pro Review - Whole Latte Love\n"
    f"[wholelattelove.com/blogs/articles/gaggia-classic-pro-review]({_PAGE_URL})\n"
    "An in-depth review of the Gaggia Classic Pro: build, boiler, "
    "steam performance, and who it suits.\n\n"
    "### Gaggia Classic - Wikipedia\n"
    "[en.wikipedia.org/wiki/Gaggia_Classic](https://en.wikipedia.org/wiki/Gaggia_Classic)\n"
    "The Gaggia Classic is a semi-automatic espresso machine first "
    "produced in 1991.\n\n"
    "### Best espresso machines 2026 - Serious Eats\n"
    "[seriouseats.com/best-espresso-machines](https://www.seriouseats.com/best-espresso-machines)\n"
    "Our top picks at every price, including the Gaggia Classic Pro.\n\n"
    "### Gaggia Classic Pro - Reddit r/espresso wiki\n"
    "[reddit.com/r/espresso/wiki/machines](https://www.reddit.com/r/espresso/wiki/machines)\n"
    "Community notes on mods, temperature surfing, and OPV adjustment.\n"
)

# Step 2: the page section exactly as BrowseTool._build_section emits it —
# label line, Image: line, then the extension's Title/URL header and body.
_PAGE_SECTION = (
    f"## browse: {_PAGE_URL}\n"
    f"Image: {_MEDIA_URL}\n"
    "Title: Gaggia Classic Pro Review - Whole Latte Love\n"
    f"URL: {_PAGE_URL}\n\n"
    "The Gaggia Classic Pro remains the default recommendation for a "
    "first real espresso machine, and after three months of daily use "
    "it is easy to see why. The commercial-style 58mm portafilter, "
    "solid-steel housing, and simple single-boiler design have barely "
    "changed since 1991, and that is the point: there is very little "
    "to break and a huge community of guides when something does.\n\n"
    "In testing, shots pulled within a few degrees of target after a "
    "ten-minute warmup, and the 2026 revision's improved steam wand "
    "produced microfoam good enough for latte art — a real step up "
    "from the older Panarello wand. Temperature surfing is still part "
    "of the routine, though a PID kit (about $85) removes the "
    "guesswork and remains the most popular first mod.\n\n"
    "At $449 it sits comfortably below heat-exchanger machines while "
    "producing shots that embarrass machines twice its price once "
    "dialed in. Pair it with a capable grinder — that matters more "
    "than the machine — and budget for a bottomless portafilter to "
    "diagnose channeling. For a first machine you intend to keep for "
    "a decade, this is still the one to beat."
)

_REPEAT_MSG = "You already made this exact tool call. Try a different query or tool."


def _serve_factory():
    seen: set[str] = set()

    def serve(name: str, args: dict):
        if name != "browse":
            return "(no entries)"
        queries = tuple(args.get("queries") or [])
        key = repr(sorted(queries))
        if key in seen:
            return _REPEAT_MSG
        seen.add(key)
        sections = []
        for q in queries:
            if q.startswith("http"):
                # Any page read returns the fixture page (one page exists).
                sections.append(_PAGE_SECTION)
            else:
                sections.append(_SEARCH_SECTION)
        return "\n\n---\n\n".join(sections)

    return serve


def _run_case(h: Harness) -> CaseResult:
    conv = converse(
        h, SYSTEM, _USER_MSG, TOOLS, _serve_factory(), force_text_final=True
    )
    calls = [
        ",".join(c["args"].get("queries", ["?"]))[:40] if c["name"] == "browse" else c["name"]
        for c in conv.all_calls()
    ] or ["(none)"]
    text = conv.final_text or ""
    source_carried = _PAGE_URL in text
    media_carried = _MEDIA_URL in text
    print(
        f"      calls={calls}  source-url={'✓' if source_carried else '✗'}  "
        f"media-url={'✓' if media_carried else '✗'}  reply[:80]={text[:80]!r}"
    )
    fails: list[str] = []
    if not text.strip():
        fails.append("no final text reply")
    if not source_carried:
        fails.append("CALIBRATION: source URL not carried (fixture may be unrealistic)")
    if not media_carried:
        fails.append("media URL not carried")
    return CaseResult("", not fails, fails)


CASES = [("image-url-carry", _run_case)]


def run(h: Harness, samples: int, only: str | None = None) -> list[CaseResult]:
    results: list[CaseResult] = []
    for cid, fn in CASES:
        if only and only != cid:
            continue

        def one(fn=fn, cid=cid):
            r = fn(h)
            return CaseResult(cid, r.passed, r.fails)

        results.extend(run_samples(f"{NAME}:{cid}", samples, one))
    return results
