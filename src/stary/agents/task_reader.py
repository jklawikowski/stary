"""TaskReader – Reads a Jira ticket via tool-calling, interprets the
description via LLM, and prepares a structured prompt with task
descriptions for downstream agents.

The LLM uses tools to fetch ticket data from Jira and can browse
related tickets or comments for additional context."""

from __future__ import annotations

import os

from stary.agents.tools import make_jira_tools
from stary.inference import InferenceClient, ToolDefinition, get_inference_client
from stary.jira_adapter import JiraAdapter

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

CRITICAL RULES:
- Do NOT produce your final JSON until you have fetched the ticket data.
- Your FINAL response must be a single, valid JSON object — nothing else.
- Do NOT include markdown fences, commentary, or prose in your final answer.

Your final JSON must follow this schema:
{
  "interpretation": "<one-paragraph summary of what the ticket asks for>",
  "tasks": [
    {"title": "<task title>", "detail": "<detailed technical description>"},
    ...
  ],
  "implementer_prompt": "<a full prompt that another AI agent can use to implement all tasks; be specific and include all technical details>"
}
"""


class TaskReader:
    """Fetches a Jira ticket using tool-calling, sends its description to
    an LLM for interpretation, and produces a structured task list for the
    next agent in the pipeline."""

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

    def run(self, ticket_input: str) -> dict:
        """Interpret a ticket and return structured input for downstream agents.

        Args:
            ticket_input: A Jira ticket URL
                (e.g. ``https://jira.devtools.intel.com/browse/PROJ-123``)

        Returns:
            {
                "ticket_id": str,
                "ticket_url": str,
                "summary": str,
                "description": str,
                "interpretation": str,
                "tasks": [{"title": str, "detail": str}, ...],
                "implementer_prompt": str,
            }
        """
        issue_key = JiraAdapter.extract_issue_key(ticket_input)
        print(f"[TaskReader] Processing ticket {issue_key}")

        # Build tools for Jira access
        tools = make_jira_tools(self._jira)

        user_message = (
            f"Analyse Jira ticket **{issue_key}**.\n\n"
            f"Start by fetching the ticket with the fetch_ticket tool "
            f"(issue_key: {issue_key}), then decompose it into concrete "
            f"implementation tasks."
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
            print(f"[TaskReader] LLM failed: {exc}")
            llm_output = {}

        # Fetch ticket data directly as fallback for metadata
        try:
            issue = self._jira.get_issue(issue_key, fields=["summary", "description"])
            summary = issue.summary
            description = issue.description
        except Exception:
            summary = llm_output.get("interpretation", issue_key)
            description = ""

        result = {
            "ticket_id": issue_key,
            "ticket_url": ticket_input if ticket_input.startswith("http") else "",
            "summary": summary,
            "description": description,
            "interpretation": llm_output.get("interpretation", ""),
            "tasks": llm_output.get("tasks", []),
            "implementer_prompt": llm_output.get("implementer_prompt", ""),
        }

        if not result["tasks"]:
            print("[TaskReader] WARNING: No tasks produced, using fallback.")
            result["tasks"] = [{"title": summary, "detail": description}]
            result["implementer_prompt"] = (
                f"Implement the following based on this Jira ticket.\n"
                f"Summary: {summary}\nDescription: {description}\n"
            )

        print(f"[TaskReader] Produced {len(result['tasks'])} task(s) for {issue_key}.")
        return result
