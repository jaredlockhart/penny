"""GitHub App authentication for agent operations.

Generates installation access tokens so agent PRs and commits
come from the bot identity instead of a personal account.
"""

# /// script
# requires-python = ">=3.12"
# dependencies = ["PyJWT[crypto]", "python-dotenv"]
# ///

from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path

GITHUB_API = "https://api.github.com"
JWT_ALGORITHM = "RS256"
BOT_SUFFIX = "[bot]"
NOREPLY_DOMAIN = "users.noreply.github.com"

# API paths
API_APP = "/app"
API_ACCESS_TOKENS = "/app/installations/{install_id}/access_tokens"

# Environment variable keys for bot identity
ENV_GH_TOKEN = "GH_TOKEN"
ENV_GIT_AUTHOR_NAME = "GIT_AUTHOR_NAME"
ENV_GIT_AUTHOR_EMAIL = "GIT_AUTHOR_EMAIL"
ENV_GIT_COMMITTER_NAME = "GIT_COMMITTER_NAME"
ENV_GIT_COMMITTER_EMAIL = "GIT_COMMITTER_EMAIL"

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
        return jwt.encode(payload, private_key, algorithm=JWT_ALGORITHM)

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
            data = self._api_request("GET", API_APP, jwt_token)
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
            API_ACCESS_TOKENS.format(install_id=self.installation_id),
            jwt_token,
        )
        self._token = data["token"]
        # Installation tokens last 1 hour; expire 5 min early
        self._token_expires = time.time() + 3300
        logger.info("GitHub App: refreshed installation token")
        return self._token

    @property
    def bot_name(self) -> str:
        return f"{self._fetch_slug()}{BOT_SUFFIX}"

    @property
    def bot_email(self) -> str:
        return f"{self.app_id}+{self._fetch_slug()}{BOT_SUFFIX}@{NOREPLY_DOMAIN}"

    def get_env(self) -> dict[str, str]:
        """Environment variables for subprocess to use bot identity."""
        return {
            ENV_GH_TOKEN: self.get_token(),
            ENV_GIT_AUTHOR_NAME: self.bot_name,
            ENV_GIT_AUTHOR_EMAIL: self.bot_email,
            ENV_GIT_COMMITTER_NAME: self.bot_name,
            ENV_GIT_COMMITTER_EMAIL: self.bot_email,
        }


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    project_root = Path(__file__).parent.parent.parent.parent
    load_dotenv(project_root / ".env")

    key_path = Path(os.environ["GITHUB_APP_PRIVATE_KEY_PATH"])
    if not key_path.is_absolute():
        key_path = project_root / key_path

    app = GitHubApp(
        app_id=int(os.environ["GITHUB_APP_ID"]),
        private_key_path=key_path,
        installation_id=int(os.environ["GITHUB_APP_INSTALLATION_ID"]),
    )
    print(app.get_token())
