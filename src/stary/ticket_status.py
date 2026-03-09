"""Ticket status marker operations for Jira.

Manages WIP and done markers on Jira tickets to prevent re-processing
and provide status visibility.

This module focuses on status marking logic. It uses JiraAdapter for
actual API calls.

The marker comments use the faceless/service account
``sys_qaplatformbot`` for production automation.
"""

from dataclasses import dataclass
from typing import Protocol

from stary.config import DEFAULT_BOT_ACCOUNT

# ---------------------------------------------------------------------------
# Default marker strings
# ---------------------------------------------------------------------------
DEFAULT_WIP_MARKER = f"[~{DEFAULT_BOT_ACCOUNT}] stary:wip"
DEFAULT_DONE_MARKER = f"[~{DEFAULT_BOT_ACCOUNT}] stary:done"
DEFAULT_FAILED_MARKER = f"[~{DEFAULT_BOT_ACCOUNT}] stary:failed"
DEFAULT_RETRY_MARKER = f"[~{DEFAULT_BOT_ACCOUNT}] retry"

# Maximum number of retry attempts allowed per ticket
MAX_RETRY_COUNT = 3


# ---------------------------------------------------------------------------
# Protocol for Jira operations (allows easy testing)
# ---------------------------------------------------------------------------


class JiraClient(Protocol):
    """Protocol defining required Jira operations for status marking."""

    def add_comment(self, issue_key: str, body: str) -> object:
        """Add a comment to an issue."""
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class StatusMarkerConfig:
    """Configuration for ticket status markers."""

    wip_marker: str = DEFAULT_WIP_MARKER
    done_marker: str = DEFAULT_DONE_MARKER
    failed_marker: str = DEFAULT_FAILED_MARKER
    retry_marker: str = DEFAULT_RETRY_MARKER
    max_retry_count: int = MAX_RETRY_COUNT
    mention_user: str | None = None


# ---------------------------------------------------------------------------
# Ticket Status Marker
# ---------------------------------------------------------------------------


class TicketStatusMarker:
    """Manages WIP/done status markers on Jira tickets.

    Uses Jira wiki markup for comment formatting.
    """

    def __init__(
        self,
        jira: JiraClient,
        config: StatusMarkerConfig | None = None,
    ):
        """Initialize the status marker.

        Args:
            jira: Jira client (JiraAdapter or compatible object)
            config: Status marker configuration
        """
        self.jira = jira
        self.config = config or StatusMarkerConfig()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_marker(self, marker: str, mention_user: str | None = None) -> str:
        """Return *marker* with the mention replaced when *mention_user* is set.

        If *mention_user* is provided (non-empty), the default bot mention
        ``[~sys_qaplatformbot]`` in *marker* is replaced with
        ``[~mention_user]``.  Otherwise the marker is returned unchanged.
        """
        user = mention_user or self.config.mention_user
        if not user:
            return marker
        return marker.replace(
            f"[~{DEFAULT_BOT_ACCOUNT}]", f"[~{user}]"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mark_wip(
        self,
        ticket_key: str,
        dagster_run_url: str | None = None,
        mention_user: str | None = None,
    ) -> None:
        """Add a WIP marker comment to the ticket.

        The WIP marker prevents the ticket from being picked up again
        while the agent pipeline is still running.

        Args:
            ticket_key: Jira issue key (e.g., "PROJ-123")
            dagster_run_url: Optional URL to the Dagster pipeline run
            mention_user: Optional Jira username to mention instead of the bot
        """
        comment_body = self.format_wip_comment(dagster_run_url, mention_user=mention_user)
        self.jira.add_comment(ticket_key, comment_body)
        print(f"[TicketStatusMarker] Marked {ticket_key} as WIP.")

    def mark_done(
        self,
        ticket_key: str,
        pr_url: str,
        status: str,
        mention_user: str | None = None,
    ) -> None:
        """Add a done marker comment with pipeline results.

        The done marker excludes the ticket from future sensor polls.

        Args:
            ticket_key: Jira issue key (e.g., "PROJ-123")
            pr_url: URL to the created pull request
            status: Pipeline status (e.g., "APPROVED", "CHANGES_REQUESTED")
            mention_user: Optional Jira username to mention instead of the bot
        """
        comment_body = self.format_done_comment(pr_url, status, mention_user=mention_user)
        self.jira.add_comment(ticket_key, comment_body)
        print(f"[TicketStatusMarker] Marked {ticket_key} as done (status={status}).")

    def mark_failed(
        self,
        ticket_key: str,
        failed_step: str,
        error_message: str,
        dagster_run_url: str | None = None,
        mention_user: str | None = None,
    ) -> None:
        """Add a failure marker comment when the pipeline fails.

        The failure marker indicates which step failed and provides a
        short error summary. Full traceback is available in Dagster logs.

        Args:
            ticket_key: Jira issue key (e.g., "PROJ-123")
            failed_step: Name of the op/step that failed
            error_message: Short error description (not full traceback)
            dagster_run_url: Optional URL to the Dagster pipeline run
            mention_user: Optional Jira username to mention instead of the bot
        """
        comment_body = self.format_failed_comment(
            failed_step, error_message, dagster_run_url, mention_user=mention_user
        )
        self.jira.add_comment(ticket_key, comment_body)
        print(
            f"[TicketStatusMarker] Marked {ticket_key} as failed "
            f"(step={failed_step})."
        )

    # ------------------------------------------------------------------
    # Comment formatting
    # ------------------------------------------------------------------

    def format_wip_comment(
        self,
        dagster_run_url: str | None = None,
        mention_user: str | None = None,
    ) -> str:
        """Build the WIP comment body.

        Uses Jira wiki markup. When a dagster_run_url is available,
        a clickable link is appended.

        Args:
            dagster_run_url: Optional URL to the Dagster pipeline run
            mention_user: Optional Jira username to mention instead of the bot

        Returns:
            Formatted comment body
        """
        marker = self._resolve_marker(self.config.wip_marker, mention_user)
        lines = [
            marker,
            "Pipeline has been triggered and is currently in progress.",
        ]
        if dagster_run_url:
            lines.append(f"[View live pipeline status|{dagster_run_url}]")
        return "\n".join(lines)

    def format_done_comment(
        self,
        pr_url: str,
        status: str,
        mention_user: str | None = None,
    ) -> str:
        """Build the done comment body.

        Args:
            pr_url: URL to the created pull request
            status: Pipeline status
            mention_user: Optional Jira username to mention instead of the bot

        Returns:
            Formatted comment body
        """
        marker = self._resolve_marker(self.config.done_marker, mention_user)
        return (
            f"{marker}\n"
            f"Status: {status}\n"
            f"PR: {pr_url}\n"
            f"Processed by stary automation."
        )

    def format_failed_comment(
        self,
        failed_step: str,
        error_message: str,
        dagster_run_url: str | None = None,
        mention_user: str | None = None,
    ) -> str:
        """Build the failure comment body.

        Args:
            failed_step: Name of the step that failed
            error_message: Short error description
            dagster_run_url: Optional URL to the Dagster pipeline run
            mention_user: Optional Jira username to mention instead of the bot

        Returns:
            Formatted comment body
        """
        marker = self._resolve_marker(self.config.failed_marker, mention_user)
        lines = [
            marker,
            f"*Failed step:* {failed_step}",
            "*Error:*",
            "{noformat}",
            error_message,
            "{noformat}",
            "",
            "Pipeline execution failed. Check Dagster logs for full details.",
        ]
        if dagster_run_url:
            lines.append(f"[View pipeline run|{dagster_run_url}]")
        return "\n".join(lines)
