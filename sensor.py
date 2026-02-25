"""Sensor: queries Jira for open tickets whose comments contain a trigger
phrase.  Returns a list of triggered ticket URLs — nothing more.

The sensor has NO knowledge of orchestration, agents, or pipelines.
"""

import os

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL", "https://jira.devtools.intel.com")
JIRA_TOKEN = os.environ.get("JIRA_TOKEN", "")
TRIGGER_COMMENT = "[~jklawiko] do it"
WIP_MARKER = "[~jklawiko] stary:wip"
PROCESSED_MARKER = "[~jklawiko] stary:done"


class Sensor:
    """Listens to Jira and reports which tickets have been triggered."""

    def __init__(
        self,
        jira_base_url: str | None = None,
        jira_token: str | None = None,
        trigger_comment: str | None = None,
        processed_marker: str | None = None,
    ):
        self.jira_base_url = jira_base_url or JIRA_BASE_URL
        self.jira_token = jira_token or JIRA_TOKEN
        self.trigger_comment = trigger_comment or TRIGGER_COMMENT
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
        """
        print("[Sensor] Querying Jira for tickets with trigger comment …")
        issues = self._search_triggered_tickets()
        print(f"[Sensor] JQL returned {len(issues)} candidate(s).")

        triggered: list[dict] = []
        for issue in issues:
            key = issue["key"]
            if not self._has_trigger_comment(key):
                print(f"[Sensor] {key}: trigger comment not confirmed, skipping.")
                continue
            ticket_url = f"{self.jira_base_url}/browse/{key}"
            print(f"[Sensor] Triggered: {key} → {ticket_url}")
            triggered.append({"ticket_key": key, "ticket_url": ticket_url})

        print(f"[Sensor] {len(triggered)} ticket(s) confirmed.")
        return triggered

    def mark_as_wip(self, ticket_key: str) -> None:
        """Add a WIP marker so the ticket is not picked up again while
        the agent pipeline is still running."""
        url = f"{self.jira_base_url}/rest/api/2/issue/{ticket_key}/comment"
        body = {"body": f"{self.wip_marker}\nPipeline in progress."}
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
        jql = (
            'status != Closed AND status != Done AND status != Resolved '
            f'AND comment ~ "\\"{self.trigger_comment}\\"" '
            f'AND NOT comment ~ "\\"{self.wip_marker}\\"" '
            f'AND NOT comment ~ "\\"{self.processed_marker}\\"" '
        )
        url = f"{self.jira_base_url}/rest/api/2/search"
        params = {"jql": jql, "fields": "key", "maxResults": 50}

        resp = requests.get(url, headers=self._headers(), params=params, timeout=120)
        resp.raise_for_status()
        return resp.json().get("issues", [])

    def _has_trigger_comment(self, issue_key: str) -> bool:
        url = f"{self.jira_base_url}/rest/api/2/issue/{issue_key}/comment"
        resp = requests.get(url, headers=self._headers(), timeout=120)
        resp.raise_for_status()
        comments = resp.json().get("comments", [])
        return any(self.trigger_comment in c.get("body", "") for c in comments)
