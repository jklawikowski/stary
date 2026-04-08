"""Centralized configuration helpers for Stary."""

import fnmatch
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)


def get_dagster_base_url() -> Optional[str]:
    """Return the configured DAGSTER_BASE_URL, normalized (no trailing slash).

    Returns ``None`` if the environment variable is not set or empty.
    Raises ``ValueError`` if the value is not a valid HTTP(S) URL.
    """
    raw = os.environ.get("DAGSTER_BASE_URL", "").strip()
    if not raw:
        return None
    return validate_and_normalize_url(raw)


def validate_and_normalize_url(url: str) -> str:
    """Validate that *url* looks like a proper HTTP(S) URL and strip trailing
    slashes.

    Raises ``ValueError`` for clearly invalid values.
    """
    url = url.strip().rstrip("/")
    if not re.match(r"^https?://[^\s/$.?#].[^\s]*$", url, re.IGNORECASE):
        raise ValueError(
            f"DAGSTER_BASE_URL is not a valid HTTP(S) URL: {url!r}"
        )
    return url


def build_dagster_run_url(
    dagster_base_url: Optional[str],
    run_id: Optional[str],
) -> Optional[str]:
    """Construct a Dagster UI run URL from the base URL and a run ID.

    Returns ``None`` when either piece is missing so callers can gracefully
    degrade.
    """
    if not dagster_base_url or not run_id:
        return None
    base = dagster_base_url.rstrip("/")
    return f"{base}/runs/{run_id}"


# ---------------------------------------------------------------------------
# Copilot SDK configuration
# ---------------------------------------------------------------------------


def get_copilot_github_token() -> Optional[str]:
    """Return the GitHub token for Copilot SDK authentication.

    Checks COPILOT_GITHUB_TOKEN first, then falls back to GH_TOKEN.
    Returns None if neither is set.
    """
    token = os.environ.get("COPILOT_GITHUB_TOKEN", "").strip()
    if token:
        return token
    return os.environ.get("GH_TOKEN", "").strip() or None


def get_copilot_model() -> str:
    """Return the model to use for Copilot SDK inference.

    Defaults to 'gpt-4o' if COPILOT_MODEL is not set.
    """
    return os.environ.get("COPILOT_MODEL", "gpt-4o").strip()


# ---------------------------------------------------------------------------
# Repository allowlist
# ---------------------------------------------------------------------------

_ALLOWED_REPOS_ENV = "ALLOWED_REPOS"


class RepoAllowlist:
    """Gate write operations to an explicit set of GitHub repositories.

    Patterns are ``owner/repo`` exact matches or ``owner/*`` org-wide
    wildcards.  Matching is case-insensitive.

    If the pattern list is empty the allowlist **denies everything**
    (fail-closed).
    """

    def __init__(self, patterns: list[str]):
        self._patterns = [p.strip().lower() for p in patterns if p.strip()]

    def is_allowed(self, owner: str, repo: str) -> bool:
        """Return *True* if ``owner/repo`` matches at least one pattern."""
        if not self._patterns:
            return False
        candidate = f"{owner}/{repo}".lower()
        return any(fnmatch.fnmatch(candidate, p) for p in self._patterns)

    def assert_allowed(self, owner: str, repo: str) -> None:
        """Raise ``ValueError`` when the repo is not on the allowlist."""
        if not self.is_allowed(owner, repo):
            raise ValueError(
                f"Repository {owner}/{repo} is not in ALLOWED_REPOS. "
                f"Allowed patterns: {self._patterns or '(none — all repos denied)'}"
            )

    @property
    def patterns(self) -> list[str]:
        return list(self._patterns)

    def __repr__(self) -> str:
        return f"RepoAllowlist({self._patterns!r})"


def get_repo_allowlist() -> RepoAllowlist:
    """Build a :class:`RepoAllowlist` from the ``ALLOWED_REPOS`` env var.

    The variable is a comma-separated list of ``owner/repo`` or
    ``owner/*`` patterns.  An empty / unset variable produces a
    fail-closed allowlist that denies every repository.
    """
    raw = os.environ.get(_ALLOWED_REPOS_ENV, "")
    patterns = [p.strip() for p in raw.split(",") if p.strip()]
    allowlist = RepoAllowlist(patterns)
    logger.info("Repo allowlist: %s", allowlist)
    return allowlist
