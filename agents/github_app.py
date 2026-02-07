"""GitHub App authentication for agent operations.

Generates installation access tokens so agent PRs and commits
come from the bot identity instead of a personal account.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path

GITHUB_API = "https://api.github.com"

logger = logging.getLogger(__name__)


class GitHubApp:
    def __init__(self, app_id: int, private_key_path: Path, installation_id: int):
        self.app_id = app_id
        self.private_key_path = private_key_path
        self.installation_id = installation_id
        self._token: str | None = None
        self._token_expires: float = 0
        self._slug: str | None = None

    def _make_jwt(self) -> str:
        import jwt  # PyJWT[crypto]

        now = int(time.time())
        payload = {
            "iat": now - 60,
            "exp": now + 600,
            "iss": str(self.app_id),
        }
        private_key = self.private_key_path.read_text()
        return jwt.encode(payload, private_key, algorithm="RS256")

    def _api_request(self, method: str, path: str, token: str) -> dict:
        url = f"{GITHUB_API}{path}"
        req = urllib.request.Request(url, method=method)
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", "application/vnd.github+json")
        if method == "POST":
            req.data = b"{}"
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def _fetch_slug(self) -> str:
        if self._slug is None:
            jwt_token = self._make_jwt()
            data = self._api_request("GET", "/app", jwt_token)
            self._slug = data["slug"]
            logger.info(f"GitHub App: {self._slug} (id={self.app_id})")
        return self._slug

    def get_token(self) -> str:
        """Get a valid installation access token, refreshing if expired."""
        if self._token and time.time() < self._token_expires:
            return self._token

        jwt_token = self._make_jwt()
        data = self._api_request(
            "POST",
            f"/app/installations/{self.installation_id}/access_tokens",
            jwt_token,
        )
        self._token = data["token"]
        # Installation tokens last 1 hour; expire 5 min early
        self._token_expires = time.time() + 3300
        logger.info("GitHub App: refreshed installation token")
        return self._token

    @property
    def bot_name(self) -> str:
        return f"{self._fetch_slug()}[bot]"

    @property
    def bot_email(self) -> str:
        return f"{self.app_id}+{self._fetch_slug()}[bot]@users.noreply.github.com"

    def get_env(self) -> dict[str, str]:
        """Environment variables for subprocess to use bot identity."""
        return {
            "GH_TOKEN": self.get_token(),
            "GIT_AUTHOR_NAME": self.bot_name,
            "GIT_AUTHOR_EMAIL": self.bot_email,
            "GIT_COMMITTER_NAME": self.bot_name,
            "GIT_COMMITTER_EMAIL": self.bot_email,
        }


if __name__ == "__main__":
    import os

    app = GitHubApp(
        app_id=int(os.environ["GITHUB_APP_ID"]),
        private_key_path=Path(os.environ["GITHUB_APP_PRIVATE_KEY_PATH"]),
        installation_id=int(os.environ["GITHUB_APP_INSTALLATION_ID"]),
    )
    for key, value in app.get_env().items():
        print(f"export {key}='{value}'")
