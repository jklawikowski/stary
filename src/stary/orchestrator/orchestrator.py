"""Orchestrator: owns the main loop.  Triggers the Sensor, decides whether
to run the agent pipeline, and drives TaskReader -> Planner -> Implementer -> Reviewer
for each triggered ticket.

Supports multi-repo tickets: TaskReader tags each task with a repo_url,
the orchestrator groups tasks by repo, and runs a Planner → Implementer
pipeline per repo.  PRs are cross-linked and reviewed individually.
"""

import logging
import os
import time
from collections import defaultdict

import requests

logger = logging.getLogger(__name__)

from stary.agents import TaskReader, Planner, Implementer, Reviewer
from stary.github_adapter import GitHubAdapter
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
        self._inference = inference_client or get_inference_client()

        self._jira = JiraAdapter(
            base_url=jira_base_url,
            token=jira_token,
        )

        self._github = GitHubAdapter(
            token=github_token,
            git_user_name=git_user_name,
            git_user_email=git_user_email,
        )

        self._trigger_detector = TriggerDetector(self._jira)
        self._status_marker = TicketStatusMarker(self._jira)

        self.task_reader = TaskReader(
            inference_client=self._inference,
            jira_adapter=self._jira,
        )
        self.planner = Planner(
            inference_client=self._inference,
            github=self._github,
        )
        self.implementer = Implementer(
            inference_client=self._inference,
            github=self._github,
        )
        self.reviewer = Reviewer(
            inference_client=self._inference,
            github=self._github,
        )
        self.poll_interval = poll_interval or POLL_INTERVAL

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_tasks_by_repo(task_input: dict) -> dict[str, dict]:
        """Split TaskReader output into per-repo task dicts.

        Returns a mapping of ``repo_url -> task_input-like dict`` where
        each dict contains only the tasks for that repo, plus the shared
        ticket metadata.
        """
        groups: dict[str, list[dict]] = defaultdict(list)
        for task in task_input.get("tasks", []):
            repo_url = task.get("repo_url", "")
            if not repo_url:
                logger.warning(
                    "Task '%s' has no repo_url — skipping",
                    task.get("title", "(untitled)"),
                )
                continue
            groups[repo_url].append(task)

        if not groups:
            raise ValueError(
                "[Orchestrator] No tasks with a repo_url found. "
                "Ensure the ticket description contains GitHub URLs.",
            )

        result: dict[str, dict] = {}
        for repo_url, tasks in groups.items():
            result[repo_url] = {
                "repo_url": repo_url,
                "ticket_id": task_input.get("ticket_id", "UNKNOWN"),
                "ticket_url": task_input.get("ticket_url", ""),
                "summary": task_input.get("summary", ""),
                "description": task_input.get("description", ""),
                "interpretation": task_input.get("interpretation", ""),
                "tasks": tasks,
            }

        return result

    def _cross_link_prs(self, pr_urls: list[str]) -> None:
        """Append related-PR links to each PR body."""
        if len(pr_urls) < 2:
            return

        for pr_url in pr_urls:
            others = [u for u in pr_urls if u != pr_url]
            links = "\n".join(f"- {u}" for u in others)
            body_addition = f"\n\n---\n**Related PRs:**\n{links}"

            try:
                owner, repo, pr_number = GitHubAdapter.parse_pr_url(pr_url)
                self._github.append_to_pr_body(
                    owner=owner,
                    repo=repo,
                    pr_number=pr_number,
                    text=body_addition,
                )
                logger.info("Cross-linked PR %s with %d other PR(s)", pr_url, len(others))
            except Exception as exc:
                logger.warning("Failed to cross-link PR %s: %s", pr_url, exc)

    # ------------------------------------------------------------------
    # Run a single ticket through the full pipeline
    # ------------------------------------------------------------------

    def run(self, ticket_input: str, auto_merge: bool = True) -> dict:
        """Run the full agent pipeline for one ticket.

        Supports multi-repo tickets: tasks are grouped by repo_url,
        a Planner → Implementer pipeline is run per repo, PRs are
        cross-linked, then each PR is reviewed.

        Args:
            ticket_input: A Jira ticket URL.
            auto_merge: If True, merge PR automatically when code review passes.
        """
        # Step 1 – interpret Jira ticket
        logger.info("=" * 60)
        logger.info("STEP 1: Reading Jira ticket")
        logger.info("=" * 60)
        task_input = self.task_reader.run(ticket_input)

        ticket_id = task_input.get("ticket_id", "UNKNOWN")
        tasks = task_input.get("tasks", [])
        repo_urls = sorted({t["repo_url"] for t in tasks if t.get("repo_url")})
        logger.info(
            "TaskReader: %s — %d task(s) across %d repo(s)",
            ticket_id, len(tasks), len(repo_urls),
        )
        for repo_url in repo_urls:
            repo_tasks = [t for t in tasks if t.get("repo_url") == repo_url]
            logger.info("  repo: %s — %d task(s)", repo_url, len(repo_tasks))

        # Step 2 – group tasks by repo
        repo_groups = self._group_tasks_by_repo(task_input)
        repo_count = len(repo_groups)

        # Step 3 – plan & implement per repo
        pr_urls: list[str] = []
        planner_outputs: list[dict] = []

        for idx, (repo_url, group) in enumerate(repo_groups.items(), 1):
            prefix = f"[repo {idx}/{repo_count}] {repo_url}"
            logger.info("=" * 60)
            logger.info("%s — %d task(s)", prefix, len(group["tasks"]))
            logger.info("=" * 60)

            # Plan
            logger.info("%s — planning & validating tasks", prefix)
            planner_output = self.planner.run(group)
            planner_outputs.append(planner_output)
            logger.info(
                "%s — branch=%s, %d step(s)",
                prefix,
                planner_output.get("branch_name", "UNKNOWN"),
                len(planner_output.get("steps", [])),
            )

            # Implement
            logger.info("%s — implementing feature", prefix)
            pr_url = self.implementer.run(planner_output)
            pr_urls.append(pr_url)
            logger.info("%s — PR created: %s", prefix, pr_url)

        # Step 4 – cross-link PRs (if multi-repo)
        self._cross_link_prs(pr_urls)

        # Step 5 – code review each PR
        pr_count = len(pr_urls)
        reviews: list[dict] = []
        for idx, pr_url in enumerate(pr_urls, 1):
            prefix = f"[PR {idx}/{pr_count}]"
            logger.info("-" * 60)
            logger.info("%s reviewing %s (auto_merge=%s)", prefix, pr_url, auto_merge)
            review = self.reviewer.run(pr_url, auto_merge=auto_merge)
            reviews.append(review)
            verdict = "APPROVED" if review.get("approved") else "CHANGES REQUESTED"
            logger.info("%s verdict: %s", prefix, verdict)

        logger.info(
            "Pipeline done — %d/%d PR(s) approved",
            sum(1 for r in reviews if r.get("approved")), pr_count,
        )

        return {
            "task_input": task_input,
            "planner_outputs": planner_outputs,
            "pr_urls": pr_urls,
            "reviews": reviews,
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
                pr_urls = result.get("pr_urls", [])
                pr_summary = ", ".join(pr_urls) if pr_urls else "N/A"
                reviews = result.get("reviews", [])
                all_approved = all(r.get("approved") for r in reviews) if reviews else False
                verdict = "APPROVED" if all_approved else "CHANGES_REQUESTED"
                logger.info("<<< %s pipeline finished", key)
                self._status_marker.mark_done(key, pr_url=pr_summary, status=verdict)
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
