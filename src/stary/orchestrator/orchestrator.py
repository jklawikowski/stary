"""Orchestrator: owns the main loop.  Triggers the Sensor, decides whether
to run the agent pipeline, and drives Agent1 -> Agent2 -> Agent3 for each
triggered ticket."""

import os
import time

import requests

from stary.agents import JiraReaderAgent, ImplementerAgent, ReviewerAgent
from stary.sensor import Sensor

POLL_INTERVAL = int(os.environ.get("SENSOR_POLL_INTERVAL", "60"))


class Orchestrator:
    def __init__(
        self,
        inference_url: str | None = None,
        agent1_inference_url: str | None = None,
        agent2_inference_url: str | None = None,
        agent3_inference_url: str | None = None,
        repo_path: str | None = None,
        jira_base_url: str | None = None,
        jira_token: str | None = None,
        github_token: str | None = None,
        git_user_name: str | None = None,
        git_user_email: str | None = None,
        poll_interval: int | None = None,
    ):
        self.sensor = Sensor(
            jira_base_url=jira_base_url,
            jira_token=jira_token,
        )
        self.agent1 = JiraReaderAgent(
            inference_url=agent1_inference_url or inference_url,
            jira_base_url=jira_base_url,
            jira_token=jira_token,
        )
        self.agent2 = ImplementerAgent(
            inference_url=agent2_inference_url or inference_url,
            github_token=github_token,
            git_user_name=git_user_name,
            git_user_email=git_user_email,
        )
        self.agent3 = ReviewerAgent(
            inference_url=agent3_inference_url or inference_url,
        )
        self.poll_interval = poll_interval or POLL_INTERVAL

    # ------------------------------------------------------------------
    # Run a single ticket through agent1 → agent2 → agent3
    # ------------------------------------------------------------------

    def run(self, ticket_input: str) -> dict:
        """Run the full agent pipeline for one ticket.

        Args:
            ticket_input: A Jira ticket URL or a legacy XML file path.
                Passed straight to Agent 1.
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
        print("STEP 3: Code review")
        print("=" * 60)
        review = self.agent3.run(pr_url)
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
        triggered = self.sensor.poll()

        if not triggered:
            print("[Orchestrator] No triggered tickets — nothing to do.")
            return []

        print(f"[Orchestrator] {len(triggered)} ticket(s) to process.")
        processed: list[str] = []

        for ticket in triggered:
            key = ticket["ticket_key"]
            url = ticket["ticket_url"]
            print(f"\n[Orchestrator] >>> Processing {key}: {url}")
            try:
                self.sensor.mark_as_wip(key)
                result = self.run(url)
                pr_url = result.get("pr_url", "N/A")
                review = result.get("review", {})
                verdict = "APPROVED" if review.get("approved") else "CHANGES_REQUESTED"
                print(f"[Orchestrator] <<< {key} pipeline finished.")
                self.sensor.mark_as_done(key, pr_url=pr_url, status=verdict)
                processed.append(key)
            except Exception as exc:
                print(f"[Orchestrator] <<< {key} pipeline FAILED: {exc}")
                self.sensor.mark_as_done(key, pr_url="N/A", status=f"FAILED: {exc}")

        return processed

    def run_forever(self) -> None:
        """Poll the sensor in a loop and process triggered tickets."""
        print(f"[Orchestrator] Starting sensor loop — polling every {self.poll_interval}s")

        while True:
            try:
                self.poll_once()
            except requests.RequestException as exc:
                print(f"[Orchestrator] Jira request error: {exc}")
            except Exception as exc:
                print(f"[Orchestrator] Unexpected error: {exc}")

            print(f"[Orchestrator] Sleeping {self.poll_interval}s …\n")
            time.sleep(self.poll_interval)
