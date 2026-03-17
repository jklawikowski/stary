"""Trigger detection for Jira tickets.

Detects tickets that have been triggered for processing based on
specific comment markers. Uses JiraAdapter for API operations.

This module focuses on trigger detection logic. Status marking is
handled by TicketStatusMarker in ticket_status.py.

The trigger user is the faceless/service account ``sys_qaplatformbot``,
used for production automation.
"""

from dataclasses import dataclass
from typing import Protocol

from stary.jira_adapter import JiraComment
from stary.ticket_status import (
    DEFAULT_DONE_MARKER,
    DEFAULT_FAILED_MARKER,
    DEFAULT_RETRY_MARKER,
    DEFAULT_WIP_MARKER,
    MAX_RETRY_COUNT,
)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
TRIGGER_COMMENT = "[~sys_qaplatformbot] do it"
TRIGGER_PR_ONLY = "[~sys_qaplatformbot] pull request"
TRIGGER_RETRY = "[~sys_qaplatformbot] retry"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TriggerConfig:
    """Configuration for trigger detection."""

    trigger_comment: str = TRIGGER_COMMENT
    trigger_pr_only: str = TRIGGER_PR_ONLY
    trigger_retry: str = TRIGGER_RETRY
    wip_marker: str = DEFAULT_WIP_MARKER
    processed_marker: str = DEFAULT_DONE_MARKER
    failed_marker: str = DEFAULT_FAILED_MARKER
    retry_marker: str = DEFAULT_RETRY_MARKER
    max_retry_count: int = MAX_RETRY_COUNT


@dataclass
class TriggeredTicket:
    """Represents a ticket that has been triggered for processing."""

    key: str
    url: str
    auto_merge: bool
    retry_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dict for backward compatibility."""
        return {
            "ticket_key": self.key,
            "ticket_url": self.url,
            "auto_merge": self.auto_merge,
            "retry_count": self.retry_count,
        }


# ---------------------------------------------------------------------------
# Protocol for Jira operations (allows easy testing)
# ---------------------------------------------------------------------------


class JiraClient(Protocol):
    """Protocol defining required Jira operations for trigger detection."""

    base_url: str

    def search_issues(
        self,
        jql: str,
        fields: list[str] | None = None,
        max_results: int = 50,
    ) -> list:
        """Search for issues using JQL."""
        ...

    def get_comments(self, issue_key: str) -> list:
        """Get all comments on an issue."""
        ...

    def build_browse_url(self, issue_key: str) -> str:
        """Build a browsable URL for an issue."""
        ...


# ---------------------------------------------------------------------------
# Trigger Detector
# ---------------------------------------------------------------------------


class TriggerDetector:
    """Detects Jira tickets that have been triggered for processing.

    Uses JiraAdapter for API operations. Does NOT modify tickets —
    that's handled by TicketStatusMarker.
    """

    def __init__(
        self,
        jira: JiraClient,
        config: TriggerConfig | None = None,
    ):
        """Initialize the trigger detector.

        Args:
            jira: Jira client (JiraAdapter or compatible object)
            config: Trigger detection configuration
        """
        self.jira = jira
        self.config = config or TriggerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def poll(self) -> list[TriggeredTicket]:
        """Query Jira and return triggered tickets.

        Uses separate JQL queries per trigger type so that comment
        fetches are only needed for retry candidates.
        """
        print("[TriggerDetector] Querying Jira for triggered tickets …")

        triggered: list[TriggeredTicket] = []
        seen_keys: set[str] = set()

        # "do it" triggers — resolved entirely from JQL
        for issue in self.jira.search_issues(
            self._build_do_it_jql(), fields=["key"], max_results=50,
        ):
            seen_keys.add(issue.key)
            url = self.jira.build_browse_url(issue.key)
            print(
                f"[TriggerDetector] Triggered: {issue.key} → {url} "
                f"(type=do_it, auto_merge=True)"
            )
            triggered.append(
                TriggeredTicket(key=issue.key, url=url, auto_merge=True)
            )

        # "pull request" triggers — resolved from JQL, deduplicated
        for issue in self.jira.search_issues(
            self._build_pr_only_jql(), fields=["key"], max_results=50,
        ):
            if issue.key in seen_keys:
                continue
            seen_keys.add(issue.key)
            url = self.jira.build_browse_url(issue.key)
            print(
                f"[TriggerDetector] Triggered: {issue.key} → {url} "
                f"(type=pr_only, auto_merge=False)"
            )
            triggered.append(
                TriggeredTicket(key=issue.key, url=url, auto_merge=False)
            )

        # Retry triggers — only these need a comment fetch for validation
        for issue in self.jira.search_issues(
            self._build_retry_jql(), fields=["key"], max_results=50,
        ):
            if issue.key in seen_keys:
                continue
            comments = self.jira.get_comments(issue.key)
            trigger_type, retry_count = self.parse_trigger_type(comments)
            if trigger_type is None:
                print(f"[TriggerDetector] {issue.key}: retry not valid, skipping.")
                continue
            seen_keys.add(issue.key)
            url = self.jira.build_browse_url(issue.key)
            auto_merge = trigger_type in ("do_it", "retry")
            print(
                f"[TriggerDetector] Triggered: {issue.key} → {url} "
                f"(type={trigger_type}, retry_count={retry_count}, auto_merge={auto_merge})"
            )
            triggered.append(
                TriggeredTicket(
                    key=issue.key, url=url,
                    auto_merge=auto_merge, retry_count=retry_count,
                )
            )

        print(f"[TriggerDetector] {len(triggered)} ticket(s) confirmed.")
        return triggered

    def _base_jql(self) -> str:
        """Common JQL predicates shared by all trigger queries."""
        return (
            'status != Closed AND status != Done AND status != Resolved '
            'AND updated >= -7d '
            f'AND NOT comment ~ "\\"{self.config.wip_marker}\\"" '
            f'AND NOT comment ~ "\\"{self.config.processed_marker}\\"" '
        )

    def _build_do_it_jql(self) -> str:
        """JQL for "do it" triggers (excludes failed tickets)."""
        return (
            self._base_jql()
            + f'AND comment ~ "\\"{self.config.trigger_comment}\\"" '
            f'AND NOT comment ~ "\\"{self.config.failed_marker}\\"" '
        )

    def _build_pr_only_jql(self) -> str:
        """JQL for "pull request" triggers (excludes failed tickets)."""
        return (
            self._base_jql()
            + f'AND comment ~ "\\"{self.config.trigger_pr_only}\\"" '
            f'AND NOT comment ~ "\\"{self.config.failed_marker}\\"" '
        )

    def _build_retry_jql(self) -> str:
        """JQL for retry triggers (requires failed marker)."""
        return (
            self._base_jql()
            + f'AND comment ~ "\\"{self.config.trigger_retry}\\"" '
            f'AND comment ~ "\\"{self.config.failed_marker}\\"" '
        )

    def parse_trigger_type(
        self,
        comments: list[JiraComment],
    ) -> tuple[str | None, int]:
        """Determine which trigger comment is present in a list of comments.

        Args:
            comments: List of JiraComment objects

        Returns:
            Tuple of (trigger_type, retry_count) where:
            - trigger_type is "do_it", "pr_only", "retry", or None
            - retry_count is the number of retry comments (0 for non-retry triggers)

        Priority: retry > do_it > pr_only (if retry is valid)
        For retry triggers, validates that retry count is within limits.
        """
        has_failed = any(
            self.config.failed_marker in c.body for c in comments
        )
        retry_count = self._count_retry_comments(comments)

        # Check if retry is requested and valid
        if retry_count > 0 and has_failed:
            if retry_count <= self.config.max_retry_count:
                if self._is_retry_newer_than_failed(comments):
                    return "retry", retry_count
                # Retry comment exists but is older than failed marker
                return None, 0
            # Max retries exceeded
            return None, 0

        # If ticket has failed marker but no valid retry, reject
        if has_failed:
            return None, 0

        # Normal trigger detection
        has_do_it = any(
            self.config.trigger_comment in c.body for c in comments
        )
        has_pr_only = any(
            self.config.trigger_pr_only in c.body for c in comments
        )

        if has_do_it:
            return "do_it", 0
        if has_pr_only:
            return "pr_only", 0
        return None, 0

    def _count_retry_comments(self, comments: list[JiraComment]) -> int:
        """Count the number of retry trigger comments.

        Args:
            comments: List of JiraComment objects

        Returns:
            Number of comments containing the retry marker
        """
        return sum(
            1 for c in comments if self.config.trigger_retry in c.body
        )

    def _is_retry_newer_than_failed(
        self,
        comments: list[JiraComment],
    ) -> bool:
        """Check if the most recent retry comment is newer than the most recent failed marker.

        Args:
            comments: List of JiraComment objects (assumed ordered by creation time)

        Returns:
            True if retry is newer than the last failure, False otherwise
        """
        last_retry_idx = -1
        last_failed_idx = -1

        for idx, c in enumerate(comments):
            if self.config.trigger_retry in c.body:
                last_retry_idx = idx
            if self.config.failed_marker in c.body:
                last_failed_idx = idx

        # Retry must be newer (higher index) than the last failure
        return last_retry_idx > last_failed_idx


