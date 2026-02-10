"""ProfilePromptHandler for collecting basic user information on first interaction."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import dateparser
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder

if TYPE_CHECKING:
    from penny.database import Database
    from penny.ollama import OllamaClient

logger = logging.getLogger(__name__)

PROFILE_COLLECTION_PROMPT = (
    "Hey! Before we dive in, I need to collect a bit of info about you. "
    "Can you tell me your name, location (city/country), and date of birth?\n\n"
    "For example: 'Jared, Toronto, Canada, March 15 1990'"
)

PROFILE_PARSE_PROMPT = (
    "Parse the following user response into structured fields. "
    "Extract: name, location (natural language), and date of birth.\n\n"
    "Respond with ONLY a JSON object in this exact format:\n"
    '{"name": "...", "location": "...", "dob": "..."}\n\n'
    "User response:\n"
)

PROFILE_ERROR_MESSAGE = (
    "I couldn't parse that. Can you try again with your name, location, and date of birth? "
    "For example: 'Jared, Toronto, Canada, March 15 1990'"
)


class ProfilePromptHandler:
    """Handles collecting and parsing basic user information."""

    def __init__(self, db: Database, ollama_client: OllamaClient):
        """
        Initialize handler.

        Args:
            db: Database instance
            ollama_client: Ollama client for parsing responses
        """
        self.db = db
        self.ollama = ollama_client
        self._tf = TimezoneFinder()
        self._geolocator = Nominatim(user_agent="penny_profile_handler")

    def needs_profile(self, sender: str) -> bool:
        """
        Check if a user needs to provide their basic profile.

        Args:
            sender: User identifier

        Returns:
            True if user has no UserInfo record
        """
        return self.db.get_user_info(sender) is None

    async def collect_profile(self, sender: str, user_message: str) -> tuple[str, bool]:
        """
        Attempt to collect profile from user's message.

        Args:
            sender: User identifier
            user_message: User's response to profile prompt

        Returns:
            Tuple of (response_message, success)
            If success=False, response_message is an error asking for retry
            If success=True, response_message is a confirmation
        """
        # Parse the user's response
        parsed = await self._parse_response(user_message)
        if not parsed:
            return PROFILE_ERROR_MESSAGE, False

        name, location, dob_str = parsed

        # Parse date of birth
        dob_date = dateparser.parse(dob_str, settings={"PREFER_DATES_FROM": "past"})
        if not dob_date:
            return PROFILE_ERROR_MESSAGE, False

        dob_formatted = dob_date.strftime("%Y-%m-%d")

        # Derive timezone from location
        timezone = await self._get_timezone(location)
        if not timezone:
            return (
                f"I couldn't determine a timezone for '{location}'. "
                "Can you be more specific with the location?",
                False,
            )

        # Save to database
        self.db.save_user_info(
            sender=sender,
            name=name,
            location=location,
            timezone=timezone,
            date_of_birth=dob_formatted,
        )

        logger.info("Collected profile for %s: %s, %s, %s", sender, name, location, dob_formatted)
        return "Got it! I've saved your info. Now, what were you asking about?", True

    async def _parse_response(self, text: str) -> tuple[str, str, str] | None:
        """
        Parse user response to extract name, location, and DOB.

        Args:
            text: User's natural language response

        Returns:
            Tuple of (name, location, dob_string) or None if parsing failed
        """
        # Build prompt for Ollama
        prompt = f"{PROFILE_PARSE_PROMPT}{text}"

        try:
            # Use generate (not chat) for simple extraction
            response = await self.ollama.generate(prompt=prompt)
            answer = response.response.strip()

            # Extract JSON from the response (handle markdown code blocks)
            json_match = re.search(r"\{[^}]+\}", answer)
            if not json_match:
                logger.warning("Failed to extract JSON from Ollama response: %s", answer)
                return None

            import json

            data = json.loads(json_match.group(0))

            name = data.get("name", "").strip()
            location = data.get("location", "").strip()
            dob = data.get("dob", "").strip()

            if not name or not location or not dob:
                logger.warning("Parsed JSON missing required fields: %s", data)
                return None

            return name, location, dob

        except Exception as e:
            logger.warning("Failed to parse profile response: %s", e)
            return None

    async def _get_timezone(self, location: str) -> str | None:
        """
        Derive IANA timezone from natural language location.

        Args:
            location: Natural language location (e.g., "Toronto, Canada")

        Returns:
            IANA timezone string (e.g., "America/Toronto") or None if lookup failed
        """
        try:
            # Geocode location to lat/lon
            geo_result = self._geolocator.geocode(location)
            if not geo_result:
                logger.warning("Geocoding failed for location: %s", location)
                return None

            # Get timezone from lat/lon
            timezone = self._tf.timezone_at(lat=geo_result.latitude, lng=geo_result.longitude)
            if not timezone:
                logger.warning(
                    "Timezone lookup failed for location: %s (%f, %f)",
                    location,
                    geo_result.latitude,
                    geo_result.longitude,
                )
                return None

            logger.debug("Resolved timezone for %s: %s", location, timezone)
            return timezone

        except Exception as e:
            logger.warning("Timezone derivation failed for %s: %s", location, e)
            return None

    def get_profile_summary(self, sender: str) -> str | None:
        """
        Get a summary of user's profile for injection into prompts.

        Args:
            sender: User identifier

        Returns:
            Formatted string with user context, or None if no profile
        """
        user_info = self.db.get_user_info(sender)
        if not user_info:
            return None

        return (
            f"User context: {user_info.name}, {user_info.location} "
            f"({user_info.timezone}), born {user_info.date_of_birth}"
        )
