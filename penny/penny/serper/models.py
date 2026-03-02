"""Serper API response models."""

from pydantic import BaseModel


class SerperImageResult(BaseModel):
    """A single image result from the Serper API."""

    imageUrl: str = ""
    imageWidth: int = 0
    imageHeight: int = 0
    thumbnailUrl: str = ""
    title: str = ""
    source: str = ""
    domain: str = ""
    link: str = ""
    position: int = 0


class SerperImageResponse(BaseModel):
    """Response from the Serper image search API."""

    images: list[SerperImageResult] = []
