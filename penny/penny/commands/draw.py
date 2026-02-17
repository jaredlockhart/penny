"""Image generation command using Ollama."""

import logging

from penny.commands.base import Command
from penny.commands.models import CommandContext, CommandResult
from penny.responses import PennyResponse

logger = logging.getLogger(__name__)


class DrawCommand(Command):
    """Generate an image from a text prompt."""

    name = "draw"
    description = "Generate an image from a text description"
    help_text = (
        "Usage: /draw <prompt>\n\n"
        "Generates an image using the configured Ollama image model. "
        "Describe what you want to see and the image will be sent back."
    )

    def __init__(self, image_model: str):
        self._image_model = image_model

    async def execute(self, args: str, context: CommandContext) -> CommandResult:
        """Execute the draw command."""
        prompt = args.strip()

        if not prompt:
            return CommandResult(text=PennyResponse.DRAW_USAGE)

        try:
            image_b64 = await context.ollama_client.generate_image(
                prompt=prompt, model=self._image_model
            )
            return CommandResult(text="", attachments=[image_b64])

        except Exception as e:
            logger.exception("Failed to generate image")
            return CommandResult(text=PennyResponse.DRAW_ERROR.format(error=e))
