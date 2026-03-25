"""Trigger detection and ticket state validation for Jira tickets.

Detects tickets that have been triggered for processing based on
specific comment markers. Uses JiraAdapter for API operations.

This module focuses on:
- Trigger detection logic (TriggerDetector)
- Ticket state validation from comment history (TicketStateValidator)

Status marking is handled by TicketStatusMarker in ticket_status.py.

The trigger user is the faceless/service account ``sys_qaplatformbot``,
used for production automation.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from stary.jira_adapter import JiraComment

logger = logging.getLogger(__name__)
from stary.ticket_status import (
    DEFAULT_DONE_MARKER,
    DEFAULT_FAILED_MARKER,
    DEFAULT_RETRY_MARKER,
    DEFAULT_WIP_MARKER,
    MAX_RETRY_COUNT,
    TicketState,
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
    query_span_days: int = 1
    jira_labels: list[str] | None = None


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
        """Query Jira and return triggered tickets (all trigger types).

        Uses separate JQL queries per trigger type so that comment
        fetches are only needed for retry candidates.
        """
        logger.info("Querying Jira for triggered tickets")

        triggered: list[TriggeredTicket] = []
        seen_keys: set[str] = set()

        triggered.extend(self.poll_do_it(seen_keys))
        triggered.extend(self.poll_pr_only(seen_keys))
        triggered.extend(self.poll_retry(seen_keys))

        logger.info("%d ticket(s) confirmed", len(triggered))
        return triggered

    def poll_do_it(
        self, seen_keys: set[str] | None = None,
    ) -> list[TriggeredTicket]:
        """Query Jira for "do it" triggered tickets only."""
        if seen_keys is None:
            seen_keys = set()
        triggered: list[TriggeredTicket] = []

        for issue in self.jira.search_issues(
            self._build_do_it_jql(), fields=["key"], max_results=50,
        ):
            if issue.key in seen_keys:
                continue
            seen_keys.add(issue.key)
            url = self.jira.build_browse_url(issue.key)
            logger.info(
                "Triggered: %s → %s (type=do_it, auto_merge=True)",
                issue.key, url,
            )
            triggered.append(
                TriggeredTicket(key=issue.key, url=url, auto_merge=True)
            )

        return triggered

    def poll_pr_only(
        self, seen_keys: set[str] | None = None,
    ) -> list[TriggeredTicket]:
        """Query Jira for "pull request" triggered tickets only."""
        if seen_keys is None:
            seen_keys = set()
        triggered: list[TriggeredTicket] = []

        for issue in self.jira.search_issues(
            self._build_pr_only_jql(), fields=["key"], max_results=50,
        ):
            if issue.key in seen_keys:
                continue
            seen_keys.add(issue.key)
            url = self.jira.build_browse_url(issue.key)
            logger.info(
                "Triggered: %s → %s (type=pr_only, auto_merge=False)",
                issue.key, url,
            )
            triggered.append(
                TriggeredTicket(key=issue.key, url=url, auto_merge=False)
            )

        return triggered

    def poll_retry(
        self, seen_keys: set[str] | None = None,
    ) -> list[TriggeredTicket]:
        """Query Jira for retry triggered tickets only."""
        if seen_keys is None:
            seen_keys = set()
        triggered: list[TriggeredTicket] = []

        for issue in self.jira.search_issues(
            self._build_retry_jql(), fields=["key"], max_results=50,
        ):
            if issue.key in seen_keys:
                continue
            comments = self.jira.get_comments(issue.key)
            trigger_type, retry_count = self.parse_trigger_type(comments)
            if trigger_type is None:
                logger.info("%s: retry not valid, skipping", issue.key)
                continue
            seen_keys.add(issue.key)
            url = self.jira.build_browse_url(issue.key)
            auto_merge = trigger_type in ("do_it", "retry")
            logger.info(
                "Triggered: %s → %s (type=%s, retry_count=%d, auto_merge=%s)",
                issue.key, url, trigger_type, retry_count, auto_merge,
            )
            triggered.append(
                TriggeredTicket(
                    key=issue.key, url=url,
                    auto_merge=auto_merge, retry_count=retry_count,
                )
            )

        return triggered

    def _base_jql(self) -> str:
        """Common JQL predicates shared by all trigger queries.

        NOTE: ``NOT comment ~`` is intentionally omitted.  Jira Server's
        full-text search does not reliably honour negated comment
        searches (verified empirically).  Deduplication is handled by
        cursor-based tracking in the Dagster sensor layer.
        """
        if self.config.jira_labels:
            quoted = ', '.join(f'"{l}"' for l in self.config.jira_labels)
            labels_clause = f'labels in ({quoted}) AND '
        else:
            labels_clause = ''
        return (
            f'{labels_clause}'
            'status != Closed AND status != Done AND status != Resolved '
            f'AND updated >= -{self.config.query_span_days}d '
        )

    def _build_do_it_jql(self) -> str:
        """JQL for \"do it\" triggers."""
        return (
            self._base_jql()
            + f'AND comment ~ "\\"{self.config.trigger_comment}\\"" '
        )

    def _build_pr_only_jql(self) -> str:
        """JQL for \"pull request\" triggers."""
        return (
            self._base_jql()
            + f'AND comment ~ "\\"{self.config.trigger_pr_only}\\"" '
        )

    def _build_retry_jql(self) -> str:
        """JQL for retry triggers (requires a terminal marker: failed or done)."""
        return (
            self._base_jql()
            + f'AND comment ~ "\\"{self.config.trigger_retry}\\"" '
            f'AND (comment ~ "\\"{self.config.failed_marker}\\"" '
            f'OR comment ~ "\\"{self.config.processed_marker}\\"") '
        )

    def _build_scheduled_jql(self, users: list[str]) -> str:
        """JQL for scheduled triggers — tickets assigned to or reported by given users."""
        quoted = ", ".join(f'"{u}"' for u in users)
        return (
            self._base_jql()
            + f'AND (assignee in ({quoted}) OR reporter in ({quoted})) '
        )

    def poll_scheduled(
        self,
        users: list[str],
        seen_keys: set[str] | None = None,
    ) -> list[TriggeredTicket]:
        """Query Jira for tickets assigned to / reported by *users*.

        Returns tickets with ``auto_merge=False`` (PR-only).
        """
        if seen_keys is None:
            seen_keys = set()
        triggered: list[TriggeredTicket] = []

        for issue in self.jira.search_issues(
            self._build_scheduled_jql(users), fields=["key"], max_results=50,
        ):
            if issue.key in seen_keys:
                continue
            seen_keys.add(issue.key)
            url = self.jira.build_browse_url(issue.key)
            logger.info(
                "Triggered: %s → %s (type=scheduled, auto_merge=False)",
                issue.key, url,
            )
            triggered.append(
                TriggeredTicket(key=issue.key, url=url, auto_merge=False)
            )

        return triggered

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
        has_done = any(
            self.config.processed_marker in c.body for c in comments
        )
        has_terminal = has_failed or has_done
        retry_count = self._count_retry_comments(comments)

        # Check if retry is requested and valid
        if retry_count > 0 and has_terminal:
            if retry_count <= self.config.max_retry_count:
                if self._is_retry_newer_than_terminal(comments):
                    return "retry", retry_count
                # Retry comment exists but is older than last terminal marker
                return None, 0
            # Max retries exceeded
            return None, 0

        # If ticket has a terminal marker but no valid retry, reject
        if has_terminal:
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

    def _is_retry_newer_than_terminal(
        self,
        comments: list[JiraComment],
    ) -> bool:
        """Check if the most recent retry comment is newer than the last terminal marker.

        A terminal marker is either a done or failed marker. The retry
        comment must be posted after the most recent terminal marker to
        be considered valid. This prevents infinite re-triggering after
        a pipeline completes.

        Args:
            comments: List of JiraComment objects (assumed ordered by creation time)

        Returns:
            True if retry is newer than the last terminal marker, False otherwise
        """
        last_retry_idx = -1
        last_terminal_idx = -1

        for idx, c in enumerate(comments):
            if self.config.trigger_retry in c.body:
                last_retry_idx = idx
            if (self.config.failed_marker in c.body
                    or self.config.processed_marker in c.body):
                last_terminal_idx = idx

        # Retry must be newer (higher index) than the last terminal marker
        return last_retry_idx > last_terminal_idx

    # ------------------------------------------------------------------
    # Combined polling (for unified comment sensor)
    # ------------------------------------------------------------------

    def poll_comment_triggers(
        self,
    ) -> list[tuple[str, str, str]]:
        """Run 3 JQL queries and return deduplicated candidate tickets.

        Returns a list of ``(ticket_key, ticket_url, trigger_hint)`` tuples.
        ``trigger_hint`` is "do_it", "pr_only", or "retry_candidate" based
        on which JQL matched.  Deduplication by ticket key ensures a ticket
        appears at most once.  Retry is checked first so that a ticket with
        both an old trigger comment and a newer retry always gets the
        ``retry_candidate`` hint.
        """
        logger.info("Querying Jira for comment-triggered tickets (3 queries)")
        candidates: list[tuple[str, str, str]] = []
        seen_keys: set[str] = set()

        # retry (highest priority – a retry supersedes old trigger comments)
        for issue in self.jira.search_issues(
            self._build_retry_jql(), fields=["key"], max_results=50,
        ):
            if issue.key not in seen_keys:
                seen_keys.add(issue.key)
                url = self.jira.build_browse_url(issue.key)
                candidates.append((issue.key, url, "retry_candidate"))

        # do_it
        for issue in self.jira.search_issues(
            self._build_do_it_jql(), fields=["key"], max_results=50,
        ):
            if issue.key not in seen_keys:
                seen_keys.add(issue.key)
                url = self.jira.build_browse_url(issue.key)
                candidates.append((issue.key, url, "do_it"))

        # pr_only
        for issue in self.jira.search_issues(
            self._build_pr_only_jql(), fields=["key"], max_results=50,
        ):
            if issue.key not in seen_keys:
                seen_keys.add(issue.key)
                url = self.jira.build_browse_url(issue.key)
                candidates.append((issue.key, url, "pr_only"))

        logger.info("%d unique candidate(s) from 3 queries", len(candidates))
        return candidates

    def poll_scheduled_candidates(
        self,
        users: list[str],
    ) -> list[tuple[str, str]]:
        """Run the scheduled JQL query and return candidate tickets.

        Returns a list of ``(ticket_key, ticket_url)`` tuples.
        """
        candidates: list[tuple[str, str]] = []
        for issue in self.jira.search_issues(
            self._build_scheduled_jql(users), fields=["key"], max_results=50,
        ):
            url = self.jira.build_browse_url(issue.key)
            candidates.append((issue.key, url))
        return candidates


# ---------------------------------------------------------------------------
# Default WIP staleness threshold
# ---------------------------------------------------------------------------
DEFAULT_WIP_STALE_HOURS = int(os.environ.get("STARY_WIP_STALE_HOURS", "3"))


# ---------------------------------------------------------------------------
# Ticket State Validator
# ---------------------------------------------------------------------------


class TicketStateValidator:
    """Validates ticket eligibility by inspecting Jira comment history.

    Comments are the ground truth for ticket state. The cursor is only
    an optimization layer to avoid unnecessary comment fetches.
    """

    def __init__(self, config: TriggerConfig | None = None):
        self.config = config or TriggerConfig()

    def determine_state(
        self,
        comments: list[JiraComment],
        wip_stale_hours: int = DEFAULT_WIP_STALE_HOURS,
    ) -> TicketState:
        """Derive the ticket state from its comment history.

        Scans comments in order (oldest → newest) and tracks the last
        marker of each type.  The latest marker determines the state.
        """
        last_wip_idx = -1
        last_done_idx = -1
        last_failed_idx = -1
        last_wip_created: str | None = None

        for idx, c in enumerate(comments):
            if self.config.wip_marker in c.body:
                last_wip_idx = idx
                last_wip_created = c.created
            if self.config.processed_marker in c.body:
                last_done_idx = idx
            if self.config.failed_marker in c.body:
                last_failed_idx = idx

        last_terminal_idx = max(last_done_idx, last_failed_idx)

        # No stary markers at all → IDLE
        if last_wip_idx == -1 and last_terminal_idx == -1:
            return TicketState.IDLE

        # Terminal marker is the latest → DONE or FAILED
        if last_terminal_idx > last_wip_idx:
            if last_done_idx >= last_failed_idx:
                return TicketState.DONE
            return TicketState.FAILED

        # WIP is the latest marker (no terminal after it)
        if last_wip_idx > last_terminal_idx:
            if self._is_wip_stale(last_wip_created, wip_stale_hours):
                return TicketState.WIP_STALE
            return TicketState.WIP

        return TicketState.IDLE

    def is_eligible(
        self,
        state: TicketState,
        trigger_type: str,
    ) -> bool:
        """Check if a ticket in the given state can be triggered.

        Uses the eligibility matrix:
        - IDLE:      do_it ✓, pr_only ✓, retry ✗, scheduled ✓
        - WIP:       all ✗
        - WIP_STALE: do_it ✓, pr_only ✓, retry ✗, scheduled ✗
        - DONE:      retry ✓ (if valid), rest ✗
        - FAILED:    retry ✓ (if valid), rest ✗
        """
        matrix = {
            TicketState.IDLE: {"do_it", "pr_only", "scheduled"},
            TicketState.WIP: set(),
            TicketState.WIP_STALE: {"do_it", "pr_only"},
            TicketState.DONE: {"retry"},
            TicketState.FAILED: {"retry"},
        }
        return trigger_type in matrix.get(state, set())

    def resolve_trigger(
        self,
        comments: list[JiraComment],
        trigger_hint: str,
    ) -> tuple[str | None, int, bool]:
        """Resolve the actual trigger type, retry count, and auto_merge.

        Args:
            comments: Full comment list for the ticket
            trigger_hint: Which JQL matched ("do_it", "pr_only", "retry_candidate")

        Returns:
            (trigger_type, retry_count, auto_merge) or (None, 0, False) if invalid.
        """
        state = self.determine_state(comments)

        if trigger_hint == "retry_candidate":
            return self._resolve_retry(comments, state)

        if trigger_hint == "do_it":
            if self.is_eligible(state, "do_it"):
                return "do_it", 0, True
            return None, 0, False

        if trigger_hint == "pr_only":
            if self.is_eligible(state, "pr_only"):
                return "pr_only", 0, False
            return None, 0, False

        return None, 0, False

    def resolve_scheduled(
        self,
        comments: list[JiraComment],
    ) -> bool:
        """Check if a ticket is eligible for scheduled processing.

        A ticket is eligible only if it has NEVER been touched by stary
        (no wip, done, or failed markers exist at all).
        """
        state = self.determine_state(comments)
        return self.is_eligible(state, "scheduled")

    def _resolve_retry(
        self,
        comments: list[JiraComment],
        state: TicketState,
    ) -> tuple[str | None, int, bool]:
        """Validate and resolve a retry trigger."""
        if not self.is_eligible(state, "retry"):
            return None, 0, False

        retry_count = sum(
            1 for c in comments if self.config.trigger_retry in c.body
        )
        if retry_count > self.config.max_retry_count:
            return None, 0, False

        if not self._is_retry_newer_than_terminal(comments):
            return None, 0, False

        auto_merge = self._resolve_retry_auto_merge(comments)
        return "retry", retry_count, auto_merge

    def _resolve_retry_auto_merge(
        self,
        comments: list[JiraComment],
    ) -> bool:
        """Determine auto_merge for retry by looking at the original trigger.

        Scans comments before the last terminal marker to find whether
        the original trigger was "do it" (auto_merge=True) or
        "pull request" (auto_merge=False).
        """
        # Find the last terminal marker index
        last_terminal_idx = -1
        for idx, c in enumerate(comments):
            if (self.config.processed_marker in c.body
                    or self.config.failed_marker in c.body):
                last_terminal_idx = idx

        # Scan backwards from terminal to find the most recent original trigger
        for idx in range(last_terminal_idx - 1, -1, -1):
            body = comments[idx].body
            if self.config.trigger_comment in body:
                return True   # "do it" → auto_merge
            if self.config.trigger_pr_only in body:
                return False  # "pull request" → no auto_merge

        # No original trigger found → default to True
        return True

    def _is_retry_newer_than_terminal(
        self,
        comments: list[JiraComment],
    ) -> bool:
        """Check if the most recent retry comment is newer than the last terminal."""
        last_retry_idx = -1
        last_terminal_idx = -1

        for idx, c in enumerate(comments):
            if self.config.trigger_retry in c.body:
                last_retry_idx = idx
            if (self.config.failed_marker in c.body
                    or self.config.processed_marker in c.body):
                last_terminal_idx = idx

        return last_retry_idx > last_terminal_idx

    @staticmethod
    def _is_wip_stale(
        wip_created: str | None,
        stale_hours: int,
    ) -> bool:
        """Check if a WIP marker is older than the staleness threshold."""
        if not wip_created:
            return False
        try:
            wip_time = datetime.fromisoformat(wip_created)
            if wip_time.tzinfo is None:
                wip_time = wip_time.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - wip_time).total_seconds() / 3600
            return age_hours > stale_hours
        except (ValueError, TypeError):
            return False


