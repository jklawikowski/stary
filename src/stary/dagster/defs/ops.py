"""Dagster ops for the Stary agent pipeline.

Each op wraps one agent step, matching the pattern from the qa-platform
orchestrator (src/vistula/dagster/defs/ops.py).

Ops:
    read_jira_ticket   – TaskReader: parse Jira ticket via LLM (Copilot SDK)
    plan_tasks         – Planner: clone repo, validate & align tasks via LLM
    implement_feature  – Implementer: generate code via LLM, push & create PR
    review_code        – Reviewer: review PR via LLM, post comments, merge
    mark_ticket_wip    – Sensor helper: mark Jira ticket as WIP
    mark_ticket_done   – Sensor helper: mark Jira ticket as done
"""

import os
from typing import Any, Dict

from dagster import Field, In, Nothing, OpExecutionContext, Out, op

from stary.config import build_dagster_run_url, get_dagster_base_url


# ---------------------------------------------------------------------------
# TaskReader: Read Jira Ticket
# ---------------------------------------------------------------------------

@op(
    description="Read and interpret a Jira ticket using TaskReader via Copilot SDK.",
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
    from stary.agents import TaskReader

    cfg = context.op_config
    agent = TaskReader(
        jira_base_url=cfg["jira_base_url"],
        jira_token=cfg["jira_token"],
    )

    ticket_url = cfg["ticket_url"]
    context.log.info("TaskReader: reading ticket %s", ticket_url)
    task_input = agent.run(ticket_url)

    context.log.info(
        "TaskReader: produced %d task(s) for ticket %s",
        len(task_input.get("tasks", [])),
        task_input.get("ticket_id", "UNKNOWN"),
    )
    context.log.info(
        "TaskReader output keys: %s | interpretation length: %d | "
        "implementer_prompt length: %d | description length: %d",
        list(task_input.keys()),
        len(task_input.get("interpretation", "")),
        len(task_input.get("implementer_prompt", "")),
        len(task_input.get("description", "")),
    )
    context.log.info(f"TaskReader implementer prompt: {task_input.get('implementer_prompt', '')}")
    for i, t in enumerate(task_input.get("tasks", [])):
        context.log.info(
            "  Task %d: %s (detail: %d chars)",
            i + 1,
            t.get("title", "<no title>"),
            len(t.get("detail", "")),
        )
    return task_input


# ---------------------------------------------------------------------------
# Planner: Validate & align tasks against repo
# ---------------------------------------------------------------------------

@op(
    description="Clone repo, scan context, and validate/align tasks via Planner LLM call.",
    ins={"task_input": In(Dict)},
    out=Out(Dict),
)
def plan_tasks(context: OpExecutionContext, task_input: Dict) -> Dict[str, Any]:
    """Clone the target repo, gather context, and validate tasks via LLM."""
    from stary.agents import Planner

    agent = Planner()

    context.log.info(
        "Planner: validating ticket %s",
        task_input.get("ticket_id", "UNKNOWN"),
    )
    planner_output = agent.run(task_input)
    context.log.info(
        "Planner: validated=%s, branch=%s, steps=%d",
        bool(planner_output.get("steps")),
        planner_output.get("branch_name", "UNKNOWN"),
        len(planner_output.get("steps", [])),
    )
    context.log.info("Planner validation notes: %s", planner_output.get("validation_notes", ""))
    for i, step in enumerate(planner_output.get("steps", [])):
        context.log.info(
            "  Step %d: %s (target_files: %s)",
            i + 1,
            step.get("title", "<no title>"),
            step.get("target_files", []),
        )
    return planner_output


# ---------------------------------------------------------------------------
# Implementer: Generate code, push & create PR
# ---------------------------------------------------------------------------

@op(
    description="Generate code via Implementer LLM call, commit/push, and create a PR.",
    ins={"planner_output": In(Dict)},
    out=Out(str),
)
def implement_feature(context: OpExecutionContext, planner_output: Dict) -> str:
    """Generate code from planner output, commit/push, and create a PR."""
    from stary.agents import Implementer

    agent = Implementer()

    ticket_id = planner_output.get("ticket_id", "UNKNOWN")
    context.log.info(
        "Implementer: implementing ticket %s",
        ticket_id,
    )
    pr_url = agent.run(planner_output)
    context.log.info("Implementer: PR created at %s", pr_url)
    return pr_url


# ---------------------------------------------------------------------------
# Reviewer: Review Code
# ---------------------------------------------------------------------------

@op(
    description="Review the PR using Reviewer \u2013 Copilot SDK-powered code review.",
    config_schema={
        "auto_merge": Field(bool, default_value=True, is_required=False),
    },
    ins={"pr_url": In(str)},
    out=Out(Dict),
)
def review_code(context: OpExecutionContext, pr_url: str) -> Dict[str, Any]:
    """Perform an automated code review on the PR and optionally merge."""
    from stary.agents import Reviewer

    cfg = context.op_config
    agent = Reviewer()
    auto_merge = cfg.get("auto_merge", True)

    context.log.info("Reviewer: reviewing PR %s (auto_merge=%s)", pr_url, auto_merge)
    review = agent.run(pr_url, auto_merge=auto_merge)
    verdict = "APPROVED" if review.get("approved") else "CHANGES REQUESTED"
    context.log.info("Reviewer: verdict = %s", verdict)
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
        "trigger_author": Field(str, default_value="", is_required=False),
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

    mention_user = cfg.get("trigger_author") or None
    status_marker.mark_wip(cfg["ticket_key"], dagster_run_url=dagster_run_url, mention_user=mention_user)
    context.log.info(
        "Marked %s as WIP (dagster_run_url=%s, mention_user=%s)",
        cfg["ticket_key"],
        dagster_run_url or "N/A",
        mention_user or "N/A",
    )


@op(
    description="Mark a Jira ticket as done with pipeline results.",
    config_schema={
        "ticket_key": str,
        "status": str,
        "jira_base_url": str,
        "jira_token": str,
        "trigger_author": Field(str, default_value="", is_required=False),
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
    mention_user = cfg.get("trigger_author") or None
    status_marker.mark_done(cfg["ticket_key"], pr_url=pr_url, status=cfg["status"], mention_user=mention_user)
    context.log.info(
        "Marked %s as done (status=%s, pr=%s, mention_user=%s)",
        cfg["ticket_key"],
        cfg["status"],
        pr_url,
        mention_user or "N/A",
    )
