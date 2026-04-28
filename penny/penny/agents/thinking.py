"""ThinkingAgent — Penny's autonomous inner monologue.

Runs on the scheduler when the system is idle.  Each cycle is a
fully model-driven agent loop: the prompt steers the model through
reading a seed topic from ``likes``, scanning ``dislikes``,
browsing the web, drafting a thought, deduping against existing
thoughts via ``exists``, and writing the result to
``unnotified-thoughts``.

The agent class is just identity + system prompt; everything else
is the shared agent shell.
"""

from __future__ import annotations

from penny.agents.base import BackgroundAgent
from penny.prompts import Prompt


class ThinkingAgent(BackgroundAgent):
    """Background worker that produces inner-monologue thoughts."""

    name = "thinking"
    system_prompt = Prompt.THINKING_SYSTEM_PROMPT
