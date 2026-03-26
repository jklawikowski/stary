"""Ticket status marker operations for Jira.

Manages WIP and done markers on Jira tickets to prevent re-processing
and provide status visibility.

This module focuses on status marking logic. It uses JiraAdapter for
actual API calls.

The marker comments use the faceless/service account
``sys_qaplatformbot`` for production automation.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from opentelemetry import trace

from stary.telemetry import tracer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default marker strings
# ---------------------------------------------------------------------------
DEFAULT_WIP_MARKER = "[~sys_qaplatformbot] stary:wip"
DEFAULT_DONE_MARKER = "[~sys_qaplatformbot] stary:done"
DEFAULT_FAILED_MARKER = "[~sys_qaplatformbot] stary:failed"
DEFAULT_RETRY_MARKER = "[~sys_qaplatformbot] retry"

# Maximum number of retry attempts allowed per ticket
MAX_RETRY_COUNT = 3


# ---------------------------------------------------------------------------
# Ticket state (derived from comment history)
# ---------------------------------------------------------------------------


class TicketState(Enum):
    """State of a ticket derived from its Jira comment markers.

    Ground truth is the comment history — NOT a volatile cursor.
    """

    IDLE = "idle"              # No stary markers at all
    WIP = "wip"                # stary:wip posted, no terminal after
    WIP_STALE = "wip_stale"    # stary:wip with no terminal, older than threshold
    DONE = "done"              # stary:done is the latest terminal marker
    FAILED = "failed"          # stary:failed is the latest terminal marker


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
    # Public API
    # ------------------------------------------------------------------

    @tracer.start_as_current_span("ticket_status.mark_wip")
    def mark_wip(
        self,
        ticket_key: str,
        dagster_run_url: str | None = None,
    ) -> None:
        """Add a WIP marker comment to the ticket.

        The WIP marker prevents the ticket from being picked up again
        while the agent pipeline is still running.

        Args:
            ticket_key: Jira issue key (e.g., "PROJ-123")
            dagster_run_url: Optional URL to the Dagster pipeline run
        """
        trace.get_current_span().set_attribute("ticket.key", ticket_key)
        comment_body = self.format_wip_comment(dagster_run_url)
        self.jira.add_comment(ticket_key, comment_body)
        logger.info("Marked %s as WIP", ticket_key)

    @tracer.start_as_current_span("ticket_status.mark_done")
    def mark_done(
        self,
        ticket_key: str,
        pr_url: str,
        status: str,
        reviews: list[dict] | None = None,
    ) -> None:
        """Add a done marker comment with pipeline results.

        The done marker excludes the ticket from future sensor polls.

        Args:
            ticket_key: Jira issue key (e.g., "PROJ-123")
            pr_url: URL to the created pull request
            status: Pipeline status (e.g., "APPROVED", "CHANGES_REQUESTED")
            reviews: Optional list of review dicts (parallel to pr_urls)
        """
        trace.get_current_span().set_attribute("ticket.key", ticket_key)
        comment_body = self.format_done_comment(pr_url, status, reviews)
        self.jira.add_comment(ticket_key, comment_body)
        logger.info("Marked %s as done (status=%s)", ticket_key, status)

    @tracer.start_as_current_span("ticket_status.mark_failed")
    def mark_failed(
        self,
        ticket_key: str,
        failed_step: str,
        error_message: str,
        dagster_run_url: str | None = None,
    ) -> None:
        """Add a failure marker comment when the pipeline fails.

        The failure marker indicates which step failed and provides a
        short error summary. Full traceback is available in Dagster logs.

        Args:
            ticket_key: Jira issue key (e.g., "PROJ-123")
            failed_step: Name of the op/step that failed
            error_message: Short error description (not full traceback)
            dagster_run_url: Optional URL to the Dagster pipeline run
        """
        trace.get_current_span().set_attribute("ticket.key", ticket_key)
        comment_body = self.format_failed_comment(
            failed_step, error_message, dagster_run_url
        )
        self.jira.add_comment(ticket_key, comment_body)
        logger.info(
            "Marked %s as failed (step=%s)", ticket_key, failed_step,
        )

    # ------------------------------------------------------------------
    # Comment formatting
    # ------------------------------------------------------------------

    def format_wip_comment(
        self,
        dagster_run_url: str | None = None,
    ) -> str:
        """Build the WIP comment body.

        Uses Jira wiki markup. When a dagster_run_url is available,
        a clickable link is appended.

        Args:
            dagster_run_url: Optional URL to the Dagster pipeline run

        Returns:
            Formatted comment body
        """
        lines = [
            self.config.wip_marker,
            "Pipeline has been triggered and is currently in progress.",
        ]
        if dagster_run_url:
            lines.append(f"[View live pipeline status|{dagster_run_url}]")
        return "\n".join(lines)

    def format_done_comment(
        self,
        pr_url: str,
        status: str,
        reviews: list[dict] | None = None,
    ) -> str:
        """Build the done comment body.

        Args:
            pr_url: URL to the created pull request(s), comma-separated
            status: Pipeline status
            reviews: Optional list of review dicts (parallel to pr_urls)

        Returns:
            Formatted comment body
        """
        pr_urls = [u.strip() for u in pr_url.split(",") if u.strip()]
        lines = [
            self.config.done_marker,
            f"Status: {status}",
        ]
        for idx, url in enumerate(pr_urls):
            approved = None
            if reviews and idx < len(reviews):
                approved = reviews[idx].get("approved")
            emoticon = "(/)" if approved else "(!)" if approved is not None else ""
            lines.append(f"{emoticon} {url}" if emoticon else url)
        lines.append("Processed by STARY.")
        return "\n".join(lines)

    def format_failed_comment(
        self,
        failed_step: str,
        error_message: str,
        dagster_run_url: str | None = None,
    ) -> str:
        """Build the failure comment body.

        Args:
            failed_step: Name of the step that failed
            error_message: Short error description
            dagster_run_url: Optional URL to the Dagster pipeline run

        Returns:
            Formatted comment body
        """
        lines = [
            self.config.failed_marker,
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
