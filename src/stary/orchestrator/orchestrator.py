"""Orchestrator: owns the main loop.  Triggers the Sensor, decides whether
to run the agent pipeline, and drives TaskReader -> Planner -> Implementer -> Reviewer
for each triggered ticket."""

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

from stary.agents import TaskReader, Planner, Implementer, Reviewer
from stary.inference import InferenceClient, get_inference_client
from stary.jira_adapter import JiraAdapter
from stary.sensor import TriggerConfig, TriggerDetector
from stary.ticket_status import StatusMarkerConfig, TicketStatusMarker

POLL_INTERVAL = int(os.environ.get("SENSOR_POLL_INTERVAL", "60"))


class Orchestrator:
    def __init__(
        self,
        inference_client: InferenceClient | None = None,
        repo_path: str | None = None,
        jira_base_url: str | None = None,
        jira_token: str | None = None,
        github_token: str | None = None,
        git_user_name: str | None = None,
        git_user_email: str | None = None,
        poll_interval: int | None = None,
    ):
        # Create shared inference client
        self._inference = inference_client or get_inference_client()

        # Create shared JiraAdapter
        self._jira = JiraAdapter(
            base_url=jira_base_url,
            token=jira_token,
        )

        # Create trigger detector and status marker
        self._trigger_detector = TriggerDetector(self._jira)
        self._status_marker = TicketStatusMarker(self._jira)

        # Create agents with shared inference client
        self.task_reader = TaskReader(
            inference_client=self._inference,
            jira_adapter=self._jira,
        )
        self.planner = Planner(
            inference_client=self._inference,
            github_token=github_token,
            git_user_name=git_user_name,
            git_user_email=git_user_email,
        )
        self.implementer = Implementer(
            inference_client=self._inference,
            github_token=github_token,
            git_user_name=git_user_name,
            git_user_email=git_user_email,
        )
        self.reviewer = Reviewer(
            inference_client=self._inference,
            github_token=github_token,
        )
        self.poll_interval = poll_interval or POLL_INTERVAL

    # ------------------------------------------------------------------
    # Run a single ticket through TaskReader → Planner → Implementer → Reviewer
    # ------------------------------------------------------------------

    def run(self, ticket_input: str, auto_merge: bool = True) -> dict:
        """Run the full agent pipeline for one ticket.

        Args:
            ticket_input: A Jira ticket URL or a legacy XML file path.
                Passed straight to TaskReader.
            auto_merge: If True, merge PR automatically when code review passes.
                If False, leave PR open even when approved.
        """

        # Step 1 – interpret Jira ticket
        logger.info("=" * 60)
        logger.info("STEP 1: Reading Jira ticket")
        logger.info("=" * 60)
        task_input = self.task_reader.run(ticket_input)
        logger.info(
            "TaskReader produced %d task(s) for implementation",
            len(task_input["tasks"]),
        )

        logger.debug("TaskReader output: %s", list(task_input.keys()))

        # Step 2 – plan & validate against repo
        logger.info("=" * 60)
        logger.info("STEP 2: Planning & validating tasks")
        logger.info("=" * 60)
        planner_output = self.planner.run(task_input)
        logger.info(
            "Planner produced %d step(s)",
            len(planner_output.get("steps", [])),
        )

        # Step 3 – implement feature (generate code, commit, push, PR)
        logger.info("=" * 60)
        logger.info("STEP 3: Implementing feature")
        logger.info("=" * 60)
        pr_url = self.implementer.run(planner_output)
        logger.info("Implementer PR: %s", pr_url)

        # Step 4 – code review
        logger.info("=" * 60)
        logger.info("STEP 4: Code review (auto_merge=%s)", auto_merge)
        logger.info("=" * 60)
        review = self.reviewer.run(pr_url, auto_merge=auto_merge)
        logger.info(
            "Reviewer verdict: %s",
            "APPROVED" if review.get("approved") else "CHANGES REQUESTED",
        )

        return {
            "task_input": task_input,
            "planner_output": planner_output,
            "pr_url": pr_url,
            "review": review,
        }

    # ------------------------------------------------------------------
    # Sensor-driven loop
    # ------------------------------------------------------------------

    def poll_once(self) -> list[str]:
        """Run one sensor poll cycle.  Returns list of processed ticket keys."""
        triggered = self._trigger_detector.poll()

        if not triggered:
            logger.info("No triggered tickets — nothing to do.")
            return []

        logger.info("%d ticket(s) to process", len(triggered))
        processed: list[str] = []

        for ticket in triggered:
            key = ticket.key
            url = ticket.url
            auto_merge = ticket.auto_merge
            logger.info(
                ">>> Processing %s: %s (auto_merge=%s)",
                key, url, auto_merge,
            )
            try:
                self._status_marker.mark_wip(key)
                result = self.run(url, auto_merge=auto_merge)
                pr_url = result.get("pr_url", "N/A")
                review = result.get("review", {})
                verdict = "APPROVED" if review.get("approved") else "CHANGES_REQUESTED"
                logger.info("<<< %s pipeline finished", key)
                self._status_marker.mark_done(key, pr_url=pr_url, status=verdict)
                processed.append(key)
            except Exception as exc:
                logger.error("<<< %s pipeline FAILED: %s", key, exc)
                self._status_marker.mark_done(key, pr_url="N/A", status=f"FAILED: {exc}")

        return processed

    def run_forever(self) -> None:
        """Poll the sensor in a loop and process triggered tickets."""
        logger.info(
            "Starting sensor loop — polling every %ds", self.poll_interval,
        )

        while True:
            try:
                self.poll_once()
            except requests.RequestException as exc:
                logger.error("Jira request error: %s", exc)
            except Exception as exc:
                logger.error("Unexpected error: %s", exc, exc_info=True)

            logger.debug("Sleeping %ds", self.poll_interval)
            time.sleep(self.poll_interval)
