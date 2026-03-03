"""Dagster ops for the Stary agent pipeline.

Each op wraps one agent step, matching the pattern from the qa-platform
orchestrator (src/vistula/dagster/defs/ops.py).

Ops:
    read_jira_ticket   – Agent 1: parse Jira ticket via LLM
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
    description="Read and interpret a Jira ticket using Agent 1 (JiraReaderAgent).",
    config_schema={
        "ticket_url": str,
        "inference_url": str,
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
        inference_url=cfg["inference_url"],
        jira_base_url=cfg["jira_base_url"],
        jira_token=cfg["jira_token"],
    )

    ticket_url = cfg["ticket_url"]
    context.log.info("Agent 1: reading ticket %s", ticket_url)
    task_input = agent.run(ticket_url)
    context.log.info(
        "Agent 1: produced %d task(s) for ticket %s",
        len(task_input.get("tasks", [])),
        task_input.get("ticket_id", "UNKNOWN"),
    )
    return task_input


# ---------------------------------------------------------------------------
# Agent 2: Implement Feature
# ---------------------------------------------------------------------------

@op(
    description="Implement features by cloning the repo, generating code via LLM, and creating a PR.",
    config_schema={
        "inference_url": str,
    },
    ins={"task_input": In(Dict)},
    out=Out(str),
)
def implement_feature(context: OpExecutionContext, task_input: Dict) -> str:
    """Clone the target repo, generate code, commit/push, and create a PR."""
    from stary.agents import ImplementerAgent

    cfg = context.op_config
    agent = ImplementerAgent(inference_url=cfg["inference_url"])

    context.log.info(
        "Agent 2: implementing ticket %s",
        task_input.get("ticket_id", "UNKNOWN"),
    )
    pr_url = agent.run(task_input)
    context.log.info("Agent 2: PR created at %s", pr_url)
    return pr_url


# ---------------------------------------------------------------------------
# Agent 3: Review Code
# ---------------------------------------------------------------------------

@op(
    description="Review the PR using Agent 3 (ReviewerAgent) – LLM-powered code review.",
    config_schema={
        "inference_url": str,
        "auto_merge": Field(bool, default_value=True, is_required=False),
    },
    ins={"pr_url": In(str)},
    out=Out(Dict),
)
def review_code(context: OpExecutionContext, pr_url: str) -> Dict[str, Any]:
    """Perform an automated code review on the PR and optionally merge."""
    from stary.agents import ReviewerAgent

    cfg = context.op_config
    agent = ReviewerAgent(inference_url=cfg["inference_url"])
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
    from stary.sensor import Sensor

    cfg = context.op_config
    s = Sensor(jira_base_url=cfg["jira_base_url"], jira_token=cfg["jira_token"])

    # Build Dagster run URL when possible
    dagster_base_url = get_dagster_base_url()
    run_id = context.run_id
    dagster_run_url = build_dagster_run_url(dagster_base_url, run_id)

    s.mark_as_wip(cfg["ticket_key"], dagster_run_url=dagster_run_url)
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
    from stary.sensor import Sensor

    cfg = context.op_config
    s = Sensor(jira_base_url=cfg["jira_base_url"], jira_token=cfg["jira_token"])
    s.mark_as_done(cfg["ticket_key"], pr_url=pr_url, status=cfg["status"])
    context.log.info(
        "Marked %s as done (status=%s, pr=%s)",
        cfg["ticket_key"],
        cfg["status"],
        pr_url,
    )
