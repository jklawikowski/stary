"""Lifecycle agent — post-pipeline Jira ticket management.

Handles:
- Transitioning the Jira ticket to Done/Closed after successful merge
- Posting a summary comment with PR link and merge details
- Creating regression tickets when issues are detected
"""

import logging
from dataclasses import dataclass

from stary.jira_adapter import JiraAdapter
from stary.telemetry import tracer

logger = logging.getLogger(__name__)


@dataclass
class LifecycleConfig:
    """Configuration for the lifecycle agent."""

    target_status: str = "Done"
    transition_on_approval: bool = True
    create_regression_tickets: bool = True


class LifecycleAgent:
    """Agent #4: Post-pipeline ticket lifecycle management."""

    def __init__(
        self,
        jira: JiraAdapter,
        config: LifecycleConfig | None = None,
    ):
        self.jira = jira
        self.config = config or LifecycleConfig()

    @tracer.start_as_current_span("lifecycle.run")
    def run(
        self,
        ticket_key: str,
        pr_urls: list[str],
        all_approved: bool,
        merged: bool = False,
        mention_user: str = "",
    ) -> dict:
        """Run post-pipeline lifecycle steps.

        Args:
            ticket_key: Jira issue key
            pr_urls: List of PR URLs created by the pipeline
            all_approved: Whether all PRs passed code review
            merged: Whether PRs were auto-merged
            mention_user: Jira username to mention in comments

        Returns:
            dict with keys: transitioned, transition_status
        """
        result: dict = {
            "transitioned": False,
            "transition_status": "",
        }

        if not self.config.transition_on_approval:
            logger.info("Ticket transition disabled by config")
            return result

        if all_approved and merged:
            try:
                success = self.jira.transition_issue(
                    ticket_key,
                    self.config.target_status,
                )
                result["transitioned"] = success
                result["transition_status"] = (
                    self.config.target_status
                    if success
                    else "transition_unavailable"
                )
                if success:
                    logger.info(
                        "Transitioned %s to %s",
                        ticket_key,
                        self.config.target_status,
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to transition %s: %s",
                    ticket_key,
                    exc,
                )
                result["transition_status"] = f"error: {exc}"
        else:
            logger.info(
                "Skipping transition for %s (approved=%s, merged=%s)",
                ticket_key,
                all_approved,
                merged,
            )

        return result

    @tracer.start_as_current_span("lifecycle.create_regression_ticket")
    def create_regression_ticket(
        self,
        original_ticket_key: str,
        pr_url: str,
        failure_details: str,
    ) -> str | None:
        """Create a new Jira ticket for a post-merge regression.

        Args:
            original_ticket_key: The ticket that introduced the regression
            pr_url: The PR that was merged
            failure_details: Description of the regression

        Returns:
            The new ticket key, or None on failure.
        """
        # NOTE: Creating issues requires project key and issue type.
        # This is a best-effort implementation that posts a comment
        # on the original ticket instead, since creating issues
        # requires project-specific configuration.
        try:
            body = (
                "[~sys_qaplatformbot] stary:regression\n"
                f"A potential regression was detected after merging "
                f"the PR from this ticket.\n\n"
                f"*PR:* {pr_url}\n"
                f"*Details:*\n"
                f"{{noformat}}\n{failure_details}\n{{noformat}}\n\n"
                f"Please investigate and create a follow-up ticket "
                f"if necessary."
            )
            self.jira.add_comment(original_ticket_key, body)
            logger.info(
                "Posted regression notice on %s",
                original_ticket_key,
            )
            return original_ticket_key
        except Exception as exc:
            logger.error(
                "Failed to post regression notice on %s: %s",
                original_ticket_key,
                exc,
            )
            return None
