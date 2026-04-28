"""NotifyAgent — Penny's proactive outreach.

Just a schedule + a prompt + a terminator: the model reads its
unnotified thoughts, picks one to share, moves it to
``notified-thoughts``, and delivers via ``send_message``.  The
``send_message`` tool itself enforces the mute and cooldown gates,
so this class has nothing to add — if there are no unnotified
thoughts, the prompt instructs the model to call ``done`` without
sending.
"""

from __future__ import annotations

from penny.agents.base import Agent
from penny.constants import NotifyPromptType
from penny.tools.send_message import SendMessageTool


class NotifyAgent(Agent):
    """Background outreach agent — sends thoughts when the user is idle."""

    name = "notify"
    prompt_type = NotifyPromptType.CYCLE
    terminator_tool = SendMessageTool.name
