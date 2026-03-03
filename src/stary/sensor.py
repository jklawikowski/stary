"""Sensor: queries Jira for open tickets whose comments contain a trigger
phrase.  Returns a list of triggered ticket URLs — nothing more.

The sensor has NO knowledge of orchestration, agents, or pipelines.
"""

import os
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL", "https://jira.devtools.intel.com")
JIRA_TOKEN = os.environ.get("JIRA_TOKEN", "")
TRIGGER_COMMENT = "[~jklawiko] do it"
TRIGGER_PR_ONLY = "[~jklawiko] pull request"
WIP_MARKER = "[~jklawiko] stary:wip"
PROCESSED_MARKER = "[~jklawiko] stary:done"


class Sensor:
    """Listens to Jira and reports which tickets have been triggered."""

    def __init__(
        self,
        jira_base_url: str | None = None,
        jira_token: str | None = None,
        trigger_comment: str | None = None,
        trigger_pr_only: str | None = None,
        processed_marker: str | None = None,
    ):
        self.jira_base_url = jira_base_url or JIRA_BASE_URL
        self.jira_token = jira_token or JIRA_TOKEN
        self.trigger_comment = trigger_comment or TRIGGER_COMMENT
        self.trigger_pr_only = trigger_pr_only or TRIGGER_PR_ONLY
        self.wip_marker = WIP_MARKER
        self.processed_marker = processed_marker or PROCESSED_MARKER

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def poll(self) -> list[dict]:
        """Query Jira and return triggered tickets.

        Returns a list of dicts, each containing:
            ``ticket_key``  – e.g. "PROJ-123"
            ``ticket_url``  – e.g. "https://jira.devtools.intel.com/browse/PROJ-123"
            ``auto_merge``  – True if triggered by "do it", False if "pull request"
        """
        print("[Sensor] Querying Jira for tickets with trigger comment …")
        issues = self._search_triggered_tickets()
        print(f"[Sensor] JQL returned {len(issues)} candidate(s).")

        triggered: list[dict] = []
        for issue in issues:
            key = issue["key"]
            trigger_type = self._get_trigger_type(key)
            if trigger_type is None:
                print(f"[Sensor] {key}: trigger comment not confirmed, skipping.")
                continue
            ticket_url = f"{self.jira_base_url}/browse/{key}"
            auto_merge = trigger_type == "do_it"
            print(f"[Sensor] Triggered: {key} → {ticket_url} (auto_merge={auto_merge})")
            triggered.append({
                "ticket_key": key,
                "ticket_url": ticket_url,
                "auto_merge": auto_merge,
            })

        print(f"[Sensor] {len(triggered)} ticket(s) confirmed.")
        return triggered

    def mark_as_wip(
        self,
        ticket_key: str,
        dagster_run_url: Optional[str] = None,
    ) -> None:
        """Add a WIP marker so the ticket is not picked up again while
        the agent pipeline is still running.

        If *dagster_run_url* is provided the comment will include a clickable
        link to the live Dagster pipeline run.
        """
        url = f"{self.jira_base_url}/rest/api/2/issue/{ticket_key}/comment"
        comment_body = self._format_wip_comment(dagster_run_url)
        body = {"body": comment_body}
        resp = requests.post(url, headers=self._headers(), json=body, timeout=120)
        resp.raise_for_status()
        print(f"[Sensor] Marked {ticket_key} as WIP.")

    def mark_as_done(self, ticket_key: str, pr_url: str, status: str) -> None:
        """Add a done-marker comment with pipeline results so the ticket
        is excluded on the next poll."""
        url = f"{self.jira_base_url}/rest/api/2/issue/{ticket_key}/comment"
        body = {
            "body": (
                f"{self.processed_marker}\n"
                f"Status: {status}\n"
                f"PR: {pr_url}\n"
                f"Processed by stary automation."
            ),
        }
        resp = requests.post(url, headers=self._headers(), json=body, timeout=120)
        resp.raise_for_status()
        print(f"[Sensor] Marked {ticket_key} as done (status={status}, pr={pr_url}).")

    # ------------------------------------------------------------------
    # WIP comment formatting
    # ------------------------------------------------------------------

    def _format_wip_comment(
        self,
        dagster_run_url: Optional[str] = None,
    ) -> str:
        """Build the WIP comment body.

        Uses Jira wiki markup.  When a *dagster_run_url* is available a
        clickable link is appended.
        """
        lines = [
            self.wip_marker,
            "Pipeline has been triggered and is currently in progress.",
        ]
        if dagster_run_url:
            lines.append(f"[View live pipeline status|{dagster_run_url}]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        if not self.jira_token:
            raise RuntimeError("JIRA_TOKEN environment variable is not set")
        return {
            "Authorization": f"Bearer {self.jira_token}",
            "Content-Type": "application/json",
        }

    def _search_triggered_tickets(self) -> list[dict]:
        # Search for either trigger comment ("do it" OR "pull request")
        jql = (
            'status != Closed AND status != Done AND status != Resolved '
            f'AND (comment ~ "\\"{self.trigger_comment}\\"" '
            f'OR comment ~ "\\"{self.trigger_pr_only}\\"") '
            f'AND NOT comment ~ "\\"{self.wip_marker}\\"" '
            f'AND NOT comment ~ "\\"{self.processed_marker}\\"" '
        )
        url = f"{self.jira_base_url}/rest/api/2/search"
        params = {"jql": jql, "fields": "key", "maxResults": 50}

        resp = requests.get(url, headers=self._headers(), params=params, timeout=120)
        if not resp.ok:
            print(
                f"[Sensor] Jira search failed: HTTP {resp.status_code}\n"
                f"  URL: {url}\n"
                f"  Response: {resp.text[:500]}"
            )
            resp.raise_for_status()
        try:
            data = resp.json()
        except requests.exceptions.JSONDecodeError:
            raise RuntimeError(
                f"Jira returned non-JSON response (HTTP {resp.status_code}). "
                f"First 500 chars: {resp.text[:500]}"
            )
        return data.get("issues", [])

    def _has_trigger_comment(self, issue_key: str) -> bool:
        """Check if issue has any trigger comment (legacy helper)."""
        return self._get_trigger_type(issue_key) is not None

    def _get_trigger_type(self, issue_key: str) -> str | None:
        """Determine which trigger comment is present.

        Returns:
            "do_it" if the full-flow trigger is found,
            "pr_only" if the PR-only trigger is found,
            None if no trigger is found.

        If both triggers are present, "do_it" takes precedence.
        """
        url = f"{self.jira_base_url}/rest/api/2/issue/{issue_key}/comment"
        resp = requests.get(url, headers=self._headers(), timeout=120)
        resp.raise_for_status()
        comments = resp.json().get("comments", [])

        has_do_it = any(self.trigger_comment in c.get("body", "") for c in comments)
        has_pr_only = any(self.trigger_pr_only in c.get("body", "") for c in comments)

        if has_do_it:
            return "do_it"
        if has_pr_only:
            return "pr_only"
        return None
