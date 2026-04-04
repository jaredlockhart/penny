"""Legacy shim — imports redirected to penny.llm."""

from penny.llm.client import LlmClient as OllamaClient

__all__ = ["OllamaClient"]
