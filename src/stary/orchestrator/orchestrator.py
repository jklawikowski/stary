"""Orchestrator: owns the main loop.  Triggers the Sensor, decides whether
to run the agent pipeline, and drives Agent1 -> Agent2 -> Agent3 for each
triggered ticket."""

import os
import time

import requests

from stary.agents import JiraReaderAgent, ImplementerAgent, ReviewerAgent
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
        self.agent1 = JiraReaderAgent(
            inference_client=self._inference,
            jira_adapter=self._jira,
        )
        self.agent2 = ImplementerAgent(
            inference_client=self._inference,
            github_token=github_token,
            git_user_name=git_user_name,
            git_user_email=git_user_email,
        )
        self.agent3 = ReviewerAgent(
            inference_client=self._inference,
            github_token=github_token,
        )
        self.poll_interval = poll_interval or POLL_INTERVAL

    # ------------------------------------------------------------------
    # Run a single ticket through agent1 → agent2 → agent3
    # ------------------------------------------------------------------

    def run(self, ticket_input: str, auto_merge: bool = True) -> dict:
        """Run the full agent pipeline for one ticket.

        Args:
            ticket_input: A Jira ticket URL or a legacy XML file path.
                Passed straight to Agent 1.
            auto_merge: If True, merge PR automatically when code review passes.
                If False, leave PR open even when approved.
        """

        # Step 1 – interpret Jira ticket
        print("=" * 60)
        print("STEP 1: Reading Jira ticket")
        print("=" * 60)
        task_input = self.agent1.run(ticket_input)
        print(f"[Orchestrator] Agent 1 produced {len(task_input['tasks'])} task(s) for implementation.")

        print(task_input)
        # Step 2 – implement feature (clone, branch, commit, push)
        print("\n" + "=" * 60)
        print("STEP 2: Implementing feature")
        print("=" * 60)
        pr_url = self.agent2.run(task_input)
        print(f"[Orchestrator] Agent 2 pr: {pr_url}")

        # Step 3 – code review
        print("\n" + "=" * 60)
        print(f"STEP 3: Code review (auto_merge={auto_merge})")
        print("=" * 60)
        review = self.agent3.run(pr_url, auto_merge=auto_merge)
        print(f"[Orchestrator] Agent 3 verdict: {'APPROVED' if review.get('approved') else 'CHANGES REQUESTED'}")

        return {
            "task_input": task_input,
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
            print("[Orchestrator] No triggered tickets \u2014 nothing to do.")
            return []

        print(f"[Orchestrator] {len(triggered)} ticket(s) to process.")
        processed: list[str] = []

        for ticket in triggered:
            key = ticket.key
            url = ticket.url
            auto_merge = ticket.auto_merge
            trigger_author = ticket.trigger_author
            print(f"\n[Orchestrator] >>> Processing {key}: {url} (auto_merge={auto_merge})")
            try:
                # In the standalone orchestrator path there is no Dagster
                # run ID available, so the WIP comment will not contain a
                # Dagster link (dagster_run_url=None).  The Dagster-managed
                # pipeline (mark_ticket_wip op) handles this automatically.
                self._status_marker.mark_wip(key, trigger_author=trigger_author)
                result = self.run(url, auto_merge=auto_merge)
                pr_url = result.get("pr_url", "N/A")
                review = result.get("review", {})
                verdict = "APPROVED" if review.get("approved") else "CHANGES_REQUESTED"
                print(f"[Orchestrator] <<< {key} pipeline finished.")
                self._status_marker.mark_done(
                    key, pr_url=pr_url, status=verdict,
                    trigger_author=trigger_author,
                )
                processed.append(key)
            except Exception as exc:
                print(f"[Orchestrator] <<< {key} pipeline FAILED: {exc}")
                self._status_marker.mark_done(
                    key, pr_url="N/A", status=f"FAILED: {exc}",
                    trigger_author=trigger_author,
                )

        return processed

    def run_forever(self) -> None:
        """Poll the sensor in a loop and process triggered tickets."""
        print(f"[Orchestrator] Starting sensor loop \u2014 polling every {self.poll_interval}s")

        while True:
            try:
                self.poll_once()
            except requests.RequestException as exc:
                print(f"[Orchestrator] Jira request error: {exc}")
            except Exception as exc:
                print(f"[Orchestrator] Unexpected error: {exc}")

            print(f"[Orchestrator] Sleeping {self.poll_interval}s \u2026\n")
            time.sleep(self.poll_interval)
