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
from stary.ticket_status import DEFAULT_DONE_MARKER, DEFAULT_WIP_MARKER

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
TRIGGER_COMMENT = "[~sys_qaplatformbot] do it"
TRIGGER_PR_ONLY = "[~sys_qaplatformbot] pull request"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TriggerConfig:
    """Configuration for trigger detection."""

    trigger_comment: str = TRIGGER_COMMENT
    trigger_pr_only: str = TRIGGER_PR_ONLY
    wip_marker: str = DEFAULT_WIP_MARKER
    processed_marker: str = DEFAULT_DONE_MARKER


@dataclass
class TriggeredTicket:
    """Represents a ticket that has been triggered for processing."""

    key: str
    url: str
    auto_merge: bool

    def to_dict(self) -> dict:
        """Convert to dict for backward compatibility."""
        return {
            "ticket_key": self.key,
            "ticket_url": self.url,
            "auto_merge": self.auto_merge,
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

        Returns:
            List of TriggeredTicket objects
        """
        print("[TriggerDetector] Querying Jira for tickets with trigger comment …")
        issues = self._search_triggered_tickets()
        print(f"[TriggerDetector] JQL returned {len(issues)} candidate(s).")

        triggered: list[TriggeredTicket] = []
        for issue in issues:
            key = issue.key
            trigger_type = self._get_trigger_type(key)
            if trigger_type is None:
                print(f"[TriggerDetector] {key}: trigger comment not confirmed, skipping.")
                continue
            ticket_url = self.jira.build_browse_url(key)
            auto_merge = trigger_type == "do_it"
            print(f"[TriggerDetector] Triggered: {key} → {ticket_url} (auto_merge={auto_merge})")
            triggered.append(
                TriggeredTicket(key=key, url=ticket_url, auto_merge=auto_merge)
            )

        print(f"[TriggerDetector] {len(triggered)} ticket(s) confirmed.")
        return triggered

    def build_jql(self) -> str:
        """Build JQL query for triggered tickets.

        Public for testing/debugging purposes.
        """
        return (
            'status != Closed AND status != Done AND status != Resolved '
            f'AND (comment ~ "\\"{self.config.trigger_comment}\\"" '
            f'OR comment ~ "\\"{self.config.trigger_pr_only}\\"") '
            f'AND NOT comment ~ "\\"{self.config.wip_marker}\\"" '
            f'AND NOT comment ~ "\\"{self.config.processed_marker}\\"" '
        )

    def parse_trigger_type(self, comments: list[JiraComment]) -> str | None:
        """Determine which trigger comment is present in a list of comments.

        Args:
            comments: List of JiraComment objects

        Returns:
            "do_it" if the full-flow trigger is found,
            "pr_only" if the PR-only trigger is found,
            None if no trigger is found.

        If both triggers are present, "do_it" takes precedence.
        """
        has_do_it = any(
            self.config.trigger_comment in c.body for c in comments
        )
        has_pr_only = any(
            self.config.trigger_pr_only in c.body for c in comments
        )

        if has_do_it:
            return "do_it"
        if has_pr_only:
            return "pr_only"
        return None

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _search_triggered_tickets(self) -> list:
        """Search for candidate tickets using JQL."""
        jql = self.build_jql()
        return self.jira.search_issues(jql, fields=["key"], max_results=50)

    def _get_trigger_type(self, issue_key: str) -> str | None:
        """Determine which trigger comment is present for an issue."""
        comments = self.jira.get_comments(issue_key)
        return self.parse_trigger_type(comments)
