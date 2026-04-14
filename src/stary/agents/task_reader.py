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

from stary.agents.tools import make_github_read_tools, make_jenkins_tools, make_jira_tools
from stary.github_adapter import GITHUB_TOKEN, GitHubAdapter
from stary.inference import InferenceClient, get_inference_client
from stary.jenkins_adapter import (
    JENKINS_ALLOWED_HOSTS,
    JENKINS_PASSWORD,
    JENKINS_USERNAME,
    JenkinsAdapter,
)
from stary.jira_adapter import JiraAdapter
from stary.telemetry import tracer

JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL", "https://jira.devtools.intel.com")
JIRA_TOKEN = os.environ.get("JIRA_TOKEN", "")

_SYSTEM_PROMPT = """\
You are an expert software architect and Jira ticket analyst.

You have tools available to fetch Jira ticket data and explore related
context. You MUST follow the structured exploration workflow below
before producing your analysis.

## MANDATORY exploration workflow — follow this order

### Phase 1: Fetch the target ticket
1. Use fetch_ticket to get the ticket's full metadata: summary,
   description, status, priority, labels, components, linked issues,
   subtasks, and epic/parent information.

### Phase 2: Explore the Jira graph (MANDATORY — do ALL of these)
2. Use get_linked_issues to fetch ALL linked tickets (blockers,
   related, duplicates). Read carefully — these are human-curated
   relationships and the highest-signal context available.
3. If the ticket has a parent epic (shown in the Epic/Parent field
   from fetch_ticket), use get_epic_children to list stories in
   that epic. Scan the LIST to understand scope, but do NOT fetch
   full descriptions of every child — only the overview matters here.
4. If the ticket has subtasks, use get_subtasks to fetch them.
   Subtasks often represent a human-curated decomposition that
   should inform your own task breakdown.
5. Drill-down budget: pick at most 3 linked or sibling tickets that
   are MOST relevant (same component, blocking, or clearly overlapping
   scope) and use fetch_ticket to read their descriptions.
   Do NOT fetch more than 3 — the overview list from steps 2-4 is
   sufficient for the rest.

### Phase 3: Comments and resolved precedent
6. Use get_comments to see discussion, clarifications, and decisions.
7. Use find_similar_resolved to find previously resolved tickets in
   the same project and components. Study these to understand:
   - What repos were touched for similar work.
   - What implementation patterns were used.
   - What level of effort was involved.

### Phase 4: External context (GitHub, Jenkins)
8. If the ticket description or comments contain GitHub file URLs
   (github.com/.../blob/...), use the GitHub tools to read the
   referenced source code — this gives you critical context about the
   codebase being modified:
   a. Use fetch_github_file to read files linked in the ticket.
      Line-range anchors (e.g. #L460 or #L10-L25) are handled
      automatically — only the relevant lines plus surrounding
      context are returned.
   b. Use list_github_directory to explore the directory structure
      around the referenced files when you need more context.
9. If the ticket description or comments contain Jenkins URLs, use the
   Jenkins tools to gather build context:
   a. Use fetch_jenkins_build to get build status and parameters.
   b. Use search_jenkins_log with error-related patterns (e.g. "error",
      "failed", "exception", "traceback") BEFORE fetching the full log.
      This is critical — Jenkins logs can be very large.
   c. Use fetch_jenkins_log (with tail_lines) only if you need more
      context around the errors found by search_jenkins_log.
   d. Use fetch_jenkins_test_report to get test pass/fail details.

### Phase 5: Synthesise and decompose
10. Only AFTER completing phases 1-4, analyse the ticket and decompose
    it into concrete implementation tasks.
11. Identify the TARGET REPOSITORY URL(s) from the ticket description,
    linked tickets, and resolved precedent.
12. Use search_issues with raw JQL ONLY if the above steps left
    significant gaps in your understanding.

CRITICAL RULES:
- Do NOT produce your final JSON until you have completed phases 1-3.
- Do NOT skip phase 2 — the Jira graph provides important context for
  avoiding duplication and understanding scope.
- The TARGET TICKET itself (description, code links, CI links) is your
  PRIMARY source of truth. The Jira graph is supplementary context.
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
      "title": "<short task title>",
      "detail": "<detailed technical description of what to implement>",
      "target_files": ["<repo-relative file paths identified from GitHub URLs or ticket description>"],
      "references": [
        {"type": "jira|github|jenkins", "url": "<source URL>", "relevance": "<one-line why this matters>"}
      ],
      "acceptance_criteria": ["<concrete done-condition>", "..."]
    },
    ...
  ]
}

Field guidance:
- `target_files`: repo-relative paths (e.g. "src/utils/parser.py") you
  identified from GitHub blob URLs, code references, or ticket description.
  These are HINTS for the planner — they do not need to be exhaustive.
  Use an empty list if no specific files were identified.
- `references`: Jira tickets, GitHub URLs, or Jenkins build URLs that
  informed this task. Include only the most relevant (max 5 per task).
  Use an empty list if no external references apply.
- `acceptance_criteria`: measurable conditions for considering this task
  done. Derive them from the ticket description, comments, and linked
  issues. Use an empty list only if the ticket provides no criteria.
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
        jenkins_allowed_hosts: list[str] | None = None,
        jenkins_username: str | None = None,
        jenkins_password: str | None = None,
        jenkins_adapter: JenkinsAdapter | None = None,
        github_adapter: GitHubAdapter | None = None,
        github_token: str | None = None,
    ):
        self._inference = inference_client or get_inference_client()
        self.jira_base_url = jira_base_url or JIRA_BASE_URL
        self.jira_token = jira_token or JIRA_TOKEN
        self._jira = jira_adapter or JiraAdapter(
            base_url=self.jira_base_url,
            token=self.jira_token,
        )
        # Jenkins integration is optional — only active when hosts are configured.
        self.jenkins_allowed_hosts = jenkins_allowed_hosts or JENKINS_ALLOWED_HOSTS
        self.jenkins_username = jenkins_username or JENKINS_USERNAME
        self.jenkins_password = jenkins_password or JENKINS_PASSWORD
        if jenkins_adapter is not None:
            self._jenkins: JenkinsAdapter | None = jenkins_adapter
        elif self.jenkins_allowed_hosts:
            self._jenkins = JenkinsAdapter(
                allowed_hosts=self.jenkins_allowed_hosts,
                username=self.jenkins_username,
                password=self.jenkins_password,
            )
        else:
            self._jenkins = None

        # GitHub integration — active when a token is available.
        _gh_token = github_token or GITHUB_TOKEN
        if github_adapter is not None:
            self._github: GitHubAdapter | None = github_adapter
        elif _gh_token:
            self._github = GitHubAdapter(token=_gh_token)
        else:
            self._github = None

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
        if self._jenkins is not None:
            tools += make_jenkins_tools(self._jenkins)
        if self._github is not None:
            tools += make_github_read_tools(self._github)

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
                timeout=1800.0,
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

        # Ensure every task has required fields (defaults for optional ones)
        for task in tasks:
            task.setdefault("repo_url", "")
            task.setdefault("target_files", [])
            task.setdefault("references", [])
            task.setdefault("acceptance_criteria", [])
            if not task.get("detail"):
                logger.warning(
                    "Task '%s' has empty detail — Planner will rely on title only",
                    task.get("title", "(untitled)"),
                )

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
