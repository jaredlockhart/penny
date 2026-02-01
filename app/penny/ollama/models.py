"""Pydantic models for Ollama API structures."""

from typing import Any

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request to generate a completion."""

    model: str
    prompt: str
    stream: bool = False
    options: dict[str, Any] | None = None


class GenerateResponse(BaseModel):
    """Response from a generation request."""

    model: str
    created_at: str = Field(alias="created_at")
    response: str
    done: bool
    context: list[int] | None = None
    total_duration: int | None = Field(default=None, alias="total_duration")
    load_duration: int | None = Field(default=None, alias="load_duration")
    prompt_eval_count: int | None = Field(default=None, alias="prompt_eval_count")
    prompt_eval_duration: int | None = Field(default=None, alias="prompt_eval_duration")
    eval_count: int | None = Field(default=None, alias="eval_count")
    eval_duration: int | None = Field(default=None, alias="eval_duration")

    class Config:
        populate_by_name = True
