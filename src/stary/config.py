"""Centralized configuration helpers for Stary."""

import os
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Default bot / service account
# ---------------------------------------------------------------------------
DEFAULT_BOT_ACCOUNT = "sys_qaplatformbot"


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
