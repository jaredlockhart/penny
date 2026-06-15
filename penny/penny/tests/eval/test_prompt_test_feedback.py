"""``prompt_test`` (dry-run) feedback contracts — recreates a prod failure.

Run ``e5a7c9e3`` (the quality collector): the agent drafted a corrected
extraction_prompt whose steps called a tool that no longer exists
(``read_latest`` — split into ``collection_read_latest`` + ``log_read``) and
pointed a collection-only read at a LOG (``user-messages``).  It dry-ran the
draft with ``prompt_test`` — but the dry-run returned only *counts* of
writes/sends and swallowed the read calls and their refusals entirely, so the
model had no signal its draft was broken and looped without converging.

These cases pin the contract the fix must satisfy: when a candidate prompt's
cycle makes an invalid call (unknown tool, or wrong shape for the memory), the
text ``prompt_test`` returns must surface that error so the model can correct.

Report-only (``min_pass_rate=None``): the model is stochastic, so we watch the
X/Y rate move from ~0 (the bug) toward N/N (fixed), not a hard pass/fail.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.eval

# Step 1 names a tool that doesn't exist anymore — the dry-run must report the
# 'no such tool' error rather than silently doing nothing.
_INVENTED_TOOL_PROMPT = (
    "Record what the user mentioned recently.\n"
    "1. Call the tool named read_latest with memory='user-messages' to read the "
    "recent messages — use exactly that tool name.\n"
    "2. done(success=true, summary='read the recent messages')."
)
# Step 1 points a collection-only read at a LOG — the dry-run must report the
# wrong-shape refusal rather than silently doing nothing.
_WRONG_SHAPE_PROMPT = (
    "Record what the user mentioned recently.\n"
    "1. Call collection_read_latest with memory='user-messages' to read the "
    "recent messages — use exactly that tool.\n"
    "2. done(success=true, summary='read the recent messages')."
)

_INTENT = "Track what the user mentions in recent conversation."


def _surfaces_error(tool: str):
    """Scorer: the dry-run text must name the bad tool AND flag it as an error."""

    def _score(output: str) -> list[str]:
        low = output.lower()
        errored = any(
            marker in low for marker in ("error", "no such tool", "not found", "refus", "is a log")
        )
        if tool in output and errored:
            return []
        return [f"dry-run did not surface the bad {tool!r} call — got: {output[:300]!r}"]

    return _score


async def test_dry_run_surfaces_invented_tool(dry_run_eval) -> None:
    await dry_run_eval(
        case_id="prompt-test-invented-tool",
        suspect="recent-chatter",
        intent=_INTENT,
        candidate_prompt=_INVENTED_TOOL_PROMPT,
        score=_surfaces_error("read_latest"),
        min_pass_rate=None,
    )


async def test_dry_run_surfaces_wrong_shape(dry_run_eval) -> None:
    await dry_run_eval(
        case_id="prompt-test-wrong-shape",
        suspect="recent-chatter",
        intent=_INTENT,
        candidate_prompt=_WRONG_SHAPE_PROMPT,
        score=_surfaces_error("collection_read_latest"),
        min_pass_rate=None,
    )
