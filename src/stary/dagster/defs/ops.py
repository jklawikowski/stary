"""Dagster ops for the Stary agent pipeline.

Each op wraps one agent step, matching the pattern from the qa-platform
orchestrator (src/vistula/dagster/defs/ops.py).

Ops:
    read_jira_ticket   – Agent 1: parse Jira ticket via LLM (Copilot SDK)
    implement_feature  – Agent 2: clone repo, generate code, push & create PR
    review_code        – Agent 3: review PR via LLM, post comments, merge
    mark_ticket_wip    – Sensor helper: mark Jira ticket as WIP
    mark_ticket_done   – Sensor helper: mark Jira ticket as done
"""

import os
from typing import Any, Dict

from dagster import Field, In, Nothing, OpExecutionContext, Out, op

from stary.config import build_dagster_run_url, get_dagster_base_url


# ---------------------------------------------------------------------------
# Agent 1: Read Jira Ticket
# ---------------------------------------------------------------------------

@op(
    description="Read and interpret a Jira ticket using Agent 1 (JiraReaderAgent) via Copilot SDK.",
    config_schema={
        "ticket_url": str,
        "jira_base_url": str,
        "jira_token": str,
    },
    ins={"after": In(Nothing)},
    out=Out(Dict),
)
def read_jira_ticket(context: OpExecutionContext) -> Dict[str, Any]:
    """Fetch a Jira ticket and break it down into implementation tasks."""
    from stary.agents import JiraReaderAgent

    cfg = context.op_config
    agent = JiraReaderAgent(
        jira_base_url=cfg["jira_base_url"],
        jira_token=cfg["jira_token"],
    )

    ticket_url = cfg["ticket_url"]
    context.log.info("Agent 1: reading ticket %s", ticket_url)
    task_input = agent.run(ticket_url)

    # Diagnostic: log structure and key fields so we can spot corruption
    context.log.info(
        "Agent 1: produced %d task(s) for ticket %s",
        len(task_input.get("tasks", [])),
        task_input.get("ticket_id", "UNKNOWN"),
    )
    context.log.info(
        "Agent 1 output keys: %s | interpretation length: %d | "
        "implementer_prompt length: %d | description length: %d",
        list(task_input.keys()),
        len(task_input.get("interpretation", "")),
        len(task_input.get("implementer_prompt", "")),
        len(task_input.get("description", "")),
    )
    context.log.info(f"Agent 1 implementer prompt: {task_input.get('implementer_prompt', '')}")
    for i, t in enumerate(task_input.get("tasks", [])):
        context.log.info(
            "  Task %d: %s (detail: %d chars)",
            i + 1,
            t.get("title", "<no title>"),
            len(t.get("detail", "")),
        )
    return task_input


# ---------------------------------------------------------------------------
# Agent 2: Implement Feature
# ---------------------------------------------------------------------------

@op(
    description="Implement features by cloning the repo, generating code via Copilot SDK, and creating a PR.",
    ins={"task_input": In(Dict)},
    out=Out(str),
)
def implement_feature(context: OpExecutionContext, task_input: Dict) -> str:
    """Clone the target repo, generate code, commit/push, and create a PR."""
    from stary.agents import ImplementerAgent

    agent = ImplementerAgent()

    context.log.info(
        "Agent 2: implementing ticket %s",
        task_input.get("ticket_id", "UNKNOWN"),
    )
    # Diagnostic: log what Agent 2 received from Agent 1
    context.log.info(
        "Agent 2 input keys: %s | tasks: %d | interpretation: %d chars | "
        "implementer_prompt: %d chars | description: %d chars",
        list(task_input.keys()),
        len(task_input.get("tasks", [])),
        len(task_input.get("interpretation", "")),
        len(task_input.get("implementer_prompt", "")),
        len(task_input.get("description", "")),
    )
    pr_url = agent.run(task_input)
    context.log.info("Agent 2: PR created at %s", pr_url)
    return pr_url


# ---------------------------------------------------------------------------
# Agent 3: Review Code
# ---------------------------------------------------------------------------

@op(
    description="Review the PR using Agent 3 (ReviewerAgent) – Copilot SDK-powered code review.",
    config_schema={
        "auto_merge": Field(bool, default_value=True, is_required=False),
    },
    ins={"pr_url": In(str)},
    out=Out(Dict),
)
def review_code(context: OpExecutionContext, pr_url: str) -> Dict[str, Any]:
    """Perform an automated code review on the PR and optionally merge."""
    from stary.agents import ReviewerAgent

    cfg = context.op_config
    agent = ReviewerAgent()
    auto_merge = cfg.get("auto_merge", True)

    context.log.info("Agent 3: reviewing PR %s (auto_merge=%s)", pr_url, auto_merge)
    review = agent.run(pr_url, auto_merge=auto_merge)
    verdict = "APPROVED" if review.get("approved") else "CHANGES REQUESTED"
    context.log.info("Agent 3: verdict = %s", verdict)
    return review


# ---------------------------------------------------------------------------
# Sensor helpers: Jira ticket status markers
# ---------------------------------------------------------------------------

@op(
    description="Mark a Jira ticket as work-in-progress.",
    config_schema={
        "ticket_key": str,
        "jira_base_url": str,
        "jira_token": str,
    },
    out=Out(Nothing),
)
def mark_ticket_wip(context: OpExecutionContext) -> None:
    """Add a WIP marker comment on the Jira ticket.

    If ``DAGSTER_BASE_URL`` is configured the WIP comment will include a
    clickable link to the current Dagster run.
    """
    from stary.jira_adapter import JiraAdapter
    from stary.ticket_status import TicketStatusMarker

    cfg = context.op_config
    jira = JiraAdapter(base_url=cfg["jira_base_url"], token=cfg["jira_token"])
    status_marker = TicketStatusMarker(jira)

    # Build Dagster run URL when possible
    dagster_base_url = get_dagster_base_url()
    run_id = context.run_id
    dagster_run_url = build_dagster_run_url(dagster_base_url, run_id)

    status_marker.mark_wip(cfg["ticket_key"], dagster_run_url=dagster_run_url)
    context.log.info(
        "Marked %s as WIP (dagster_run_url=%s)",
        cfg["ticket_key"],
        dagster_run_url or "N/A",
    )


@op(
    description="Mark a Jira ticket as done with pipeline results.",
    config_schema={
        "ticket_key": str,
        "status": str,
        "jira_base_url": str,
        "jira_token": str,
    },
    ins={"pr_url": In(str), "review_result": In(Dict)},
    out=Out(Nothing),
)
def mark_ticket_done(context: OpExecutionContext, pr_url: str, review_result: Dict) -> None:
    """Add a done-marker comment on the Jira ticket."""
    from stary.jira_adapter import JiraAdapter
    from stary.ticket_status import TicketStatusMarker

    cfg = context.op_config
    jira = JiraAdapter(base_url=cfg["jira_base_url"], token=cfg["jira_token"])
    status_marker = TicketStatusMarker(jira)
    status_marker.mark_done(cfg["ticket_key"], pr_url=pr_url, status=cfg["status"])
    context.log.info(
        "Marked %s as done (status=%s, pr=%s)",
        cfg["ticket_key"],
        cfg["status"],
        pr_url,
    )
