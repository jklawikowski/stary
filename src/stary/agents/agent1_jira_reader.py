"""Agent 1: Reads a Jira ticket by URL (or from a legacy XML file),
interprets the description via LLM, and prepares a structured prompt with
task descriptions for downstream agents."""

import json
import os
import re
import xml.etree.ElementTree as ET

import requests

INFERENCE_URL = os.environ.get(
    "AGENT1_INFERENCE_URL",
    os.environ.get("INFERENCE_URL", "http://localhost:8080/v1/chat/completions"),
)

JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL", "https://jira.devtools.intel.com")
JIRA_TOKEN = os.environ.get("JIRA_TOKEN", "")

# System prompt that instructs the LLM how to analyse a Jira ticket.
_SYSTEM_PROMPT = """\
You are an expert software architect and Jira ticket analyst.
You will receive a Jira ticket consisting of a summary and a free-form
description.  Your job is to autonomously decompose the ticket into a
complete set of concrete, actionable implementation tasks.  Do NOT expect
pre-defined sub-tasks — YOU must infer every task from the description.

CRITICAL RULES — you MUST follow all of these:
- Do NOT use any tools or function calls. You have no tools available.
- Do NOT attempt to browse, search, read files, or gather additional data.
- Do NOT converse, ask questions, or produce any text outside the JSON.
- Work ONLY with the information provided in the user message.
- Your ENTIRE response must be a single, valid JSON object — nothing else.

Follow these steps:
1. Understand the intent, scope, and acceptance criteria of the ticket.
2. Identify every piece of work required: new files, functions, tests,
   configuration changes, documentation, etc.
3. For each task provide:
   - "title": a short imperative title (e.g. "Create login endpoint").
   - "detail": a thorough technical description of what needs to be done,
     including API contracts, data models, edge cases, and acceptance
     criteria.
4. Return ONLY valid JSON with the following schema (no markdown fences,
   no commentary, no prose — ONLY the JSON object):
{
  "interpretation": "<one-paragraph summary of what the ticket asks for>",
  "tasks": [
    {"title": "<task title>", "detail": "<detailed technical description>"},
    ...
  ],
  "implementer_prompt": "<a full prompt that another AI agent can use to implement all the tasks above; be specific and include all technical details>"
}
"""


class JiraReaderAgent:
    """Fetches a Jira ticket by URL, sends its description to an LLM for
    interpretation, and produces a structured task list + implementer prompt
    for the next agent in the pipeline."""

    def __init__(
        self,
        inference_url: str | None = None,
        jira_base_url: str | None = None,
        jira_token: str | None = None,
    ):
        self.inference_url = inference_url or INFERENCE_URL
        self.jira_base_url = jira_base_url or JIRA_BASE_URL
        self.jira_token = jira_token or JIRA_TOKEN

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def run(self, ticket_input: str) -> dict:
        """Interpret a ticket and return structured input for Agent 2.

        Args:
            ticket_input: A Jira ticket URL
                (e.g. ``https://jira.devtools.intel.com/browse/PROJ-123``)
                **or** a path to a legacy XML file.

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
        # 1. Obtain ticket fields -------------------------------------------
        if ticket_input.startswith("http"):
            ticket_id, summary, description = self._fetch_from_jira(ticket_input)
            ticket_url = ticket_input
        else:
            # Legacy XML path
            ticket_id, summary, description = self._parse_xml(ticket_input)
            ticket_url = ""
        print(f"[Agent1] Parsed ticket {ticket_id}")

        # 2. Call LLM to interpret description and generate tasks ------------
        llm_output = self._interpret_with_llm(ticket_id, summary, description)
        print(llm_output)

        # 3. Assemble result for Agent 2 -------------------------------------
        result = {
            "ticket_id": ticket_id,
            "ticket_url": ticket_url,
            "summary": summary,
            "description": description,
            "interpretation": llm_output.get("interpretation", ""),
            "tasks": llm_output.get("tasks", []),
            "implementer_prompt": llm_output.get("implementer_prompt", ""),
        }

        print(f"[Agent1] LLM produced {len(result['tasks'])} task(s) for ticket {ticket_id}.")
        return result

    # ------------------------------------------------------------------
    # Jira REST API
    # ------------------------------------------------------------------

    def _jira_headers(self) -> dict:
        if not self.jira_token:
            raise RuntimeError("JIRA_TOKEN environment variable is not set")
        return {
            "Authorization": f"Bearer {self.jira_token}",
            "Content-Type": "application/json",
        }

    def _fetch_from_jira(self, ticket_url: str) -> tuple[str, str, str]:
        """Fetch ticket data from Jira REST API given a browse URL.

        Extracts the issue key from the URL and calls
        ``/rest/api/2/issue/{key}`` to retrieve summary and description.

        Returns (ticket_id, summary, description).
        """
        # URL form: https://jira.devtools.intel.com/browse/PROJ-123
        issue_key = ticket_url.rstrip("/").rsplit("/", 1)[-1]
        api_url = f"{self.jira_base_url}/rest/api/2/issue/{issue_key}"
        params = {"fields": "summary,description"}

        print(f"[Agent1] Fetching ticket {issue_key} from Jira …")
        resp = requests.get(
            api_url, headers=self._jira_headers(), params=params, timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        ticket_id = data.get("key", issue_key)
        fields = data.get("fields", {})
        summary = fields.get("summary", "")
        description = fields.get("description", "")
        return ticket_id, summary, description

    # ------------------------------------------------------------------
    # Legacy XML parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_xml(xml_path: str):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        ticket_id = root.findtext("id", default="UNKNOWN")
        summary = root.findtext("summary", default="")
        description = root.findtext("description", default="")

        return ticket_id, summary, description

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _interpret_with_llm(
        self,
        ticket_id: str,
        summary: str,
        description: str,
    ) -> dict:
        """Send the ticket info to the LLM and return parsed JSON.

        The LLM is responsible for breaking the ticket down into concrete
        implementation tasks — no pre-defined tasks are provided."""

        user_message = (
            f"Jira Ticket: {ticket_id}\n"
            f"Summary: {summary}\n\n"
            f"Description:\n{description}\n\n"
            "Analyse the ticket, break it down into concrete implementation "
            "tasks, and return the JSON as instructed."
        )

        payload = {
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.2,
            "tool_choice": "none",
        }
        url = self.inference_url
        print(f"[Agent1] Calling LLM at {url} …")

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            print(f"[Agent1] LLM response data: {data}")

            content = data["choices"][0]["message"].get("content")
            if not content:
                print("[Agent1] LLM returned tool_calls or empty content instead of JSON. Retrying is needed.")
                raise ValueError("LLM did not return text content — likely attempted tool calls.")
            # Strip possible markdown code fences (even with surrounding prose)
            content = content.strip()
            fence_match = re.search(r"```(?:json)?\s*\n(.*)```", content, re.DOTALL)
            if fence_match:
                content = fence_match.group(1).strip()
            elif content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
                content = content.strip()

            return json.loads(content)

        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
            print(f"[Agent1] Failed to parse LLM response: {exc}")
            return {
                "interpretation": description,
                "tasks": [{"title": summary, "detail": description}],
                "implementer_prompt": (
                    f"Implement the following based on this Jira ticket.\n"
                    f"Summary: {summary}\nDescription: {description}\n"
                ),
            }

