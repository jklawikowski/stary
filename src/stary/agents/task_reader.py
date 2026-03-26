"""TaskReader – Reads a Jira ticket via tool-calling, interprets the
description via LLM, and prepares a structured task list for downstream
agents.

Each task is tagged with its target repository URL so the orchestrator
can group tasks by repo and run independent pipelines per repository.

The LLM uses tools to fetch ticket data from Jira and can browse
related tickets or comments for additional context."""

from __future__ import annotations

import logging
import os

from opentelemetry import trace

logger = logging.getLogger(__name__)

from stary.agents.tools import make_jira_tools
from stary.inference import InferenceClient, get_inference_client
from stary.jira_adapter import JiraAdapter
from stary.telemetry import tracer

JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL", "https://jira.devtools.intel.com")
JIRA_TOKEN = os.environ.get("JIRA_TOKEN", "")

_SYSTEM_PROMPT = """\
You are an expert software architect and Jira ticket analyst.

You have tools available to fetch Jira ticket data. Use them to gather
all information you need about the ticket before producing your analysis.

Your workflow:
1. Use the fetch_ticket tool to get the ticket's summary and description.
2. Optionally use get_comments to see any discussion or clarifications.
3. Optionally use search_issues to find related tickets for context.
4. Analyse the ticket and decompose it into concrete implementation tasks.
5. Identify the TARGET REPOSITORY URL(s) from the ticket description.

CRITICAL RULES:
- Do NOT produce your final JSON until you have fetched the ticket data.
- Your FINAL response must be a single, valid JSON object — nothing else.
- Do NOT include markdown fences, commentary, or prose in your final answer.

## Repository URL identification — READ CAREFULLY

The ticket description may contain one or MORE GitHub URLs pointing to
different repositories. A single ticket can require changes across
multiple repos (e.g. a backend repo and a frontend repo).

For EACH task you produce, you MUST identify which repository it
belongs to. Assign the correct `repo_url` to every task.

Rules for identifying repository URLs:
- Look for GitHub URLs (github.com/<owner>/<repo>).
- Distinguish between repos that are actual TARGETS for implementation
  versus repos mentioned only as references, examples, or documentation.
- Users sometimes paste a URL that points DEEPER than the repo root,
  e.g. `github.com/org/repo/tree/main/src/module` or
  `github.com/org/repo/blob/dev/README.md`. You MUST normalise these
  to the repository root: `https://github.com/org/repo`.
- Strip any trailing slashes, fragment identifiers, or query parameters.
- Return ONLY the base repo URL in the form `https://github.com/<owner>/<repo>`.
- If a task has no identifiable repository, set its repo_url to an
  empty string.

Your final JSON must follow this schema:
{
  "interpretation": "<one-paragraph summary of what the ticket asks for>",
  "tasks": [
    {
      "repo_url": "<normalised GitHub repo URL for this task>",
      "title": "<task title>",
      "detail": "<detailed technical description>"
    },
    ...
  ]
}
"""


class TaskReader:
    """Fetches a Jira ticket using tool-calling, sends its description to
    an LLM for interpretation, and produces a structured task list for the
    next agent in the pipeline.

    Each task carries its own ``repo_url`` so the orchestrator can group
    tasks by repository and run independent Planner/Implementer pipelines.
    """

    def __init__(
        self,
        inference_client: InferenceClient | None = None,
        jira_base_url: str | None = None,
        jira_token: str | None = None,
        jira_adapter: JiraAdapter | None = None,
    ):
        self._inference = inference_client or get_inference_client()
        self.jira_base_url = jira_base_url or JIRA_BASE_URL
        self.jira_token = jira_token or JIRA_TOKEN
        self._jira = jira_adapter or JiraAdapter(
            base_url=self.jira_base_url,
            token=self.jira_token,
        )

    @tracer.start_as_current_span("task_reader.run")
    def run(self, ticket_input: str) -> dict:
        """Interpret a ticket and return structured input for downstream agents.

        Args:
            ticket_input: A Jira ticket URL
                (e.g. ``https://jira.devtools.intel.com/browse/PROJ-123``)

        Returns:
            dict with keys:
                ticket_id, ticket_url, summary, description,
                interpretation, tasks (each task has repo_url, title, detail).
        """
        issue_key = JiraAdapter.extract_issue_key(ticket_input)
        span = trace.get_current_span()
        span.set_attribute("ticket.key", issue_key)
        logger.info("Processing ticket %s", issue_key)

        tools = make_jira_tools(self._jira)

        user_message = (
            f"Analyse Jira ticket **{issue_key}**.\n\n"
            f"Start by fetching the ticket with the fetch_ticket tool "
            f"(issue_key: {issue_key}), then decompose it into concrete "
            f"implementation tasks. Tag each task with the repo_url of "
            f"the repository it should be implemented in."
        )

        try:
            llm_output = self._inference.chat_json_with_tools(
                system=_SYSTEM_PROMPT,
                user=user_message,
                tools=tools,
                temperature=0.2,
                timeout=900.0,
            )
        except Exception as exc:
            logger.error("LLM failed: %s", exc)
            llm_output = {}

        # Fetch ticket data directly as fallback for metadata
        try:
            issue = self._jira.get_issue(issue_key, fields=["summary", "description"])
            summary = issue.summary
            description = issue.description
        except Exception:
            summary = llm_output.get("interpretation", issue_key)
            description = ""

        tasks = llm_output.get("tasks", [])

        # Ensure every task has a repo_url field (default to empty)
        for task in tasks:
            task.setdefault("repo_url", "")

        if not tasks:
            logger.warning("No tasks produced, using fallback")
            tasks = [{"repo_url": "", "title": summary, "detail": description}]

        repo_urls = {t["repo_url"] for t in tasks if t["repo_url"]}
        logger.info(
            "Produced %d task(s) across %d repo(s) for %s",
            len(tasks), len(repo_urls), issue_key,
        )

        return {
            "ticket_id": issue_key,
            "ticket_url": ticket_input if ticket_input.startswith("http") else "",
            "summary": summary,
            "description": description,
            "interpretation": llm_output.get("interpretation", ""),
            "tasks": tasks,
        }
