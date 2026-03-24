"""Low-level Jira REST API adapter.

Provides a clean, reusable interface for Jira operations with:
- Centralized authentication
- Retry logic for transient failures
- Consistent error handling

This module has NO business logic — it's a pure data-access layer.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from stary.telemetry import record_jira_request

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL", "https://jira.devtools.intel.com")
JIRA_TOKEN = os.environ.get("JIRA_TOKEN", "")

DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 0.5


@dataclass
class JiraIssue:
    """Represents a Jira issue with core fields."""

    key: str
    summary: str = ""
    description: str = ""
    fields: dict | None = None


@dataclass
class JiraComment:
    """Represents a Jira comment."""

    id: str
    body: str
    author: str = ""
    created: str = ""


class JiraAdapter:
    """Low-level Jira REST API operations.

    Handles authentication, retries, and HTTP details. Business logic
    belongs in higher-level modules (TriggerDetector, TicketStatusMarker).
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    ):
        self.base_url = (base_url or JIRA_BASE_URL).rstrip("/")
        self.token = token or JIRA_TOKEN
        self.timeout = timeout
        self._session = self._create_session(max_retries, backoff_factor)

    def _create_session(
        self,
        max_retries: int,
        backoff_factor: float,
    ) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Build request headers with authentication."""
        if not self.token:
            raise RuntimeError("JIRA_TOKEN environment variable is not set")
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Core HTTP methods
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        json_body: dict | None = None,
    ) -> requests.Response:
        """Execute an HTTP request with error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/rest/api/2/issue/PROJ-123")
            params: Query parameters
            json_body: JSON request body

        Returns:
            Response object

        Raises:
            requests.HTTPError: On non-2xx responses
            RuntimeError: On JSON parse errors
        """
        url = f"{self.base_url}{endpoint}"
        t0 = time.monotonic()
        resp = self._session.request(
            method=method,
            url=url,
            headers=self._headers(),
            params=params,
            json=json_body,
            timeout=self.timeout,
        )
        record_jira_request(method, endpoint, resp.status_code, time.monotonic() - t0)
        if not resp.ok:
            logger.warning(
                "Request failed: %s %s — HTTP %d: %.500s",
                method, endpoint, resp.status_code, resp.text,
            )
        resp.raise_for_status()
        return resp

    def _get(
        self,
        endpoint: str,
        params: dict | None = None,
    ) -> requests.Response:
        """Execute a GET request."""
        return self._request("GET", endpoint, params=params)

    def _post(
        self,
        endpoint: str,
        json_body: dict | None = None,
    ) -> requests.Response:
        """Execute a POST request."""
        return self._request("POST", endpoint, json_body=json_body)

    # ------------------------------------------------------------------
    # Issue operations
    # ------------------------------------------------------------------

    def search_issues(
        self,
        jql: str,
        fields: list[str] | None = None,
        max_results: int = 50,
    ) -> list[JiraIssue]:
        """Search for issues using JQL.

        Args:
            jql: JQL query string
            fields: List of fields to retrieve (default: ["key"])
            max_results: Maximum number of results

        Returns:
            List of JiraIssue objects
        """
        params = {
            "jql": jql,
            "fields": ",".join(fields or ["key"]),
            "maxResults": max_results,
        }
        resp = self._get("/rest/api/2/search", params=params)
        try:
            data = resp.json()
        except requests.exceptions.JSONDecodeError:
            raise RuntimeError(
                f"Jira returned non-JSON response (HTTP {resp.status_code}). "
                f"First 500 chars: {resp.text[:500]}"
            )

        issues: list[JiraIssue] = []
        for item in data.get("issues", []):
            fields_data = item.get("fields") or {}
            issues.append(
                JiraIssue(
                    key=item["key"],
                    summary=fields_data.get("summary", "") or "",
                    description=fields_data.get("description", "") or "",
                    fields=fields_data if fields_data else None,
                )
            )
        return issues

    def get_issue(
        self,
        issue_key: str,
        fields: list[str] | None = None,
    ) -> JiraIssue:
        """Fetch a single issue by key.

        Args:
            issue_key: Issue key (e.g., "PROJ-123")
            fields: List of fields to retrieve

        Returns:
            JiraIssue object
        """
        params = {}
        if fields:
            params["fields"] = ",".join(fields)

        resp = self._get(f"/rest/api/2/issue/{issue_key}", params=params)
        data = resp.json()
        fields_data = data.get("fields") or {}
        return JiraIssue(
            key=data.get("key", issue_key),
            summary=fields_data.get("summary", "") or "",
            description=fields_data.get("description", "") or "",
            fields=fields_data if fields_data else None,
        )

    # ------------------------------------------------------------------
    # Comment operations
    # ------------------------------------------------------------------

    def get_comments(self, issue_key: str) -> list[JiraComment]:
        """Get all comments on an issue.

        Args:
            issue_key: Issue key (e.g., "PROJ-123")

        Returns:
            List of JiraComment objects
        """
        resp = self._get(f"/rest/api/2/issue/{issue_key}/comment")
        data = resp.json()
        comments: list[JiraComment] = []
        for c in data.get("comments", []) or []:
            author_data = c.get("author") or {}
            comments.append(
                JiraComment(
                    id=c.get("id", "") or "",
                    body=c.get("body", "") or "",
                    author=author_data.get("name", "") or author_data.get("displayName", "") or "",
                    created=c.get("created", "") or "",
                )
            )
        return comments

    def add_comment(self, issue_key: str, body: str) -> JiraComment:
        """Add a comment to an issue.

        Args:
            issue_key: Issue key (e.g., "PROJ-123")
            body: Comment body (supports Jira wiki markup)

        Returns:
            The created JiraComment
        """
        resp = self._post(
            f"/rest/api/2/issue/{issue_key}/comment",
            json_body={"body": body},
        )
        data = resp.json()
        return JiraComment(
            id=data.get("id", ""),
            body=data.get("body", body),
        )

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    def build_browse_url(self, issue_key: str) -> str:
        """Build a browsable URL for an issue."""
        return f"{self.base_url}/browse/{issue_key}"

    @staticmethod
    def extract_issue_key(ticket_url: str) -> str:
        """Extract issue key from a Jira browse URL.

        Args:
            ticket_url: e.g., "https://jira.devtools.intel.com/browse/PROJ-123"

        Returns:
            Issue key (e.g., "PROJ-123")
        """
        return ticket_url.rstrip("/").rsplit("/", 1)[-1]
