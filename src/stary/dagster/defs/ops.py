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
from opentelemetry import trace

from stary.config import build_dagster_run_url, get_dagster_base_url
from stary.telemetry import tracer


# ---------------------------------------------------------------------------
# TaskReader: Read Jira Ticket
# ---------------------------------------------------------------------------

@op(
    description="Read and interpret a Jira ticket using TaskReader via Copilot SDK.",
    config_schema={
        "ticket_url": str,
        "jira_base_url": str,
    },
    ins={"after": In(Nothing)},
    out=Out(Dict),
)
def read_jira_ticket(context: OpExecutionContext) -> Dict[str, Any]:
    """Fetch a Jira ticket and break it down into implementation tasks."""
    from stary.agents import TaskReader

    cfg = context.op_config
    jira_token = os.environ.get("JIRA_TOKEN", "")
    if not jira_token:
        raise RuntimeError("JIRA_TOKEN environment variable is not set")

    with tracer.start_as_current_span("dagster.op.read_jira_ticket") as span:
        span.set_attribute("ticket.url", cfg["ticket_url"])
        agent = TaskReader(
            jira_base_url=cfg["jira_base_url"],
            jira_token=jira_token,
        )

        ticket_url = cfg["ticket_url"]
        context.log.info("TaskReader: reading ticket %s", ticket_url)
        task_input = agent.run(ticket_url)

        ticket_id = task_input.get("ticket_id", "UNKNOWN")
        span.set_attribute("ticket.key", ticket_id)
        tasks = task_input.get("tasks", [])
        repo_urls = sorted({t["repo_url"] for t in tasks if t.get("repo_url")})

        context.log.info(
            "TaskReader: %s — %d task(s) across %d repo(s)",
            ticket_id, len(tasks), len(repo_urls),
        )
        for repo_url in repo_urls:
            repo_tasks = [t for t in tasks if t.get("repo_url") == repo_url]
            context.log.info("  repo: %s — %d task(s)", repo_url, len(repo_tasks))
            for t in repo_tasks:
                context.log.info(
                    "    - %s (detail: %d chars)",
                    t.get("title", "<no title>"),
                    len(t.get("detail", "")),
                )
    return task_input


# ---------------------------------------------------------------------------
# Planner: Validate & align tasks against repo
# ---------------------------------------------------------------------------

@op(
    description="Group tasks by repo, clone each, validate/align tasks via Planner LLM.",
    ins={"task_input": In(Dict)},
    out=Out(Dict),
)
def plan_tasks(context: OpExecutionContext, task_input: Dict) -> Dict[str, Any]:
    """Group tasks by repo URL and run the Planner on each repo group.

    Returns a dict with a ``repo_plans`` list — one planner_output per repo.
    """
    from collections import defaultdict

    from stary.agents import Planner

    ticket_id = task_input.get("ticket_id", "UNKNOWN")

    # Group tasks by repo_url
    groups: dict[str, list[dict]] = defaultdict(list)
    for task in task_input.get("tasks", []):
        repo_url = task.get("repo_url", "")
        if not repo_url:
            context.log.warning("Task '%s' has no repo_url — skipping", task.get("title"))
            continue
        groups[repo_url].append(task)

    if not groups:
        raise RuntimeError("No tasks with a repo_url found in TaskReader output.")

    repo_count = len(groups)
    context.log.info(
        "Planner: %s — %d repo(s) to plan", ticket_id, repo_count,
    )

    repo_plans: list[dict] = []
    for idx, (repo_url, tasks) in enumerate(groups.items(), 1):
        prefix = f"[repo {idx}/{repo_count}] {repo_url}"
        context.log.info("%s", "=" * 60)
        context.log.info("%s — %d task(s)", prefix, len(tasks))
        context.log.info("%s", "=" * 60)

        group_input = {
            "repo_url": repo_url,
            "ticket_id": ticket_id,
            "summary": task_input.get("summary", ""),
            "tasks": tasks,
        }
        agent = Planner()
        planner_output = agent.run(group_input)
        repo_plans.append(planner_output)

        context.log.info(
            "%s — branch=%s, %d step(s)",
            prefix,
            planner_output.get("branch_name", "UNKNOWN"),
            len(planner_output.get("steps", [])),
        )
        for i, step in enumerate(planner_output.get("steps", []), 1):
            context.log.info(
                "  step %d: %s", i, step.get("prompt", "")[:80],
            )

    return {"repo_plans": repo_plans, "ticket_id": ticket_id}


# ---------------------------------------------------------------------------
# Implementer: Generate code, push & create PR
# ---------------------------------------------------------------------------

@op(
    description="Generate code via Implementer LLM call per repo, commit/push, and create PRs.",
    ins={"plan_result": In(Dict)},
    out=Out(Dict),
)
def implement_feature(context: OpExecutionContext, plan_result: Dict) -> Dict[str, Any]:
    """Implement each repo plan and return all PR URLs."""
    from stary.agents import Implementer

    repo_plans = plan_result.get("repo_plans", [])
    ticket_id = plan_result.get("ticket_id", "UNKNOWN")
    repo_count = len(repo_plans)

    context.log.info(
        "Implementer: %s — %d repo(s) to implement", ticket_id, repo_count,
    )

    pr_urls: list[str] = []
    for idx, plan in enumerate(repo_plans, 1):
        repo_url = plan.get("repo_url", "UNKNOWN")
        steps = plan.get("steps", [])
        prefix = f"[repo {idx}/{repo_count}] {repo_url}"
        context.log.info("%s", "=" * 60)
        context.log.info("%s — %d step(s)", prefix, len(steps))
        context.log.info("%s", "=" * 60)

        agent = Implementer()
        pr_url = agent.run(plan)
        pr_urls.append(pr_url)
        context.log.info("%s — PR created: %s", prefix, pr_url)

    context.log.info(
        "Implementer: %s — all %d repo(s) done, %d PR(s) created",
        ticket_id, repo_count, len(pr_urls),
    )
    return {"pr_urls": pr_urls, "ticket_id": ticket_id}


# ---------------------------------------------------------------------------
# Reviewer: Review Code
# ---------------------------------------------------------------------------

@op(
    description="Review all PRs using Reviewer – Copilot SDK-powered code review.",
    config_schema={
        "auto_merge": Field(bool, default_value=True, is_required=False),
    },
    ins={"impl_result": In(Dict)},
    out=Out(Dict),
)
def review_code(context: OpExecutionContext, impl_result: Dict) -> Dict[str, Any]:
    """Perform automated code review on all PRs and optionally merge."""
    from stary.agents import Reviewer

    cfg = context.op_config
    auto_merge = cfg.get("auto_merge", True)
    pr_urls = impl_result.get("pr_urls", [])

    pr_count = len(pr_urls)
    context.log.info("Reviewer: %d PR(s) to review", pr_count)

    reviews: list[dict] = []
    for idx, pr_url in enumerate(pr_urls, 1):
        prefix = f"[PR {idx}/{pr_count}]"
        context.log.info("%s", "-" * 60)
        context.log.info("%s reviewing %s (auto_merge=%s)", prefix, pr_url, auto_merge)

        agent = Reviewer()
        review = agent.run(pr_url, auto_merge=auto_merge)
        reviews.append(review)
        verdict = "APPROVED" if review.get("approved") else "CHANGES REQUESTED"
        context.log.info("%s verdict: %s", prefix, verdict)

    all_approved = all(r.get("approved") for r in reviews) if reviews else False
    context.log.info(
        "Reviewer: done — %d/%d approved",
        sum(1 for r in reviews if r.get("approved")), pr_count,
    )
    return {"reviews": reviews, "all_approved": all_approved, "pr_urls": pr_urls}


# ---------------------------------------------------------------------------
# Lifecycle: Transition Jira ticket status
# ---------------------------------------------------------------------------

@op(
    description="Transition Jira ticket status after successful pipeline completion.",
    config_schema={
        "ticket_key": str,
        "jira_base_url": str,
    },
    ins={"review_result": In(Dict)},
    out=Out(Dict),
)
def manage_ticket_lifecycle(
    context: OpExecutionContext, review_result: Dict,
) -> Dict[str, Any]:
    """Run Agent #4: transition ticket if all PRs approved and merged."""
    from stary.agents.lifecycle import LifecycleAgent
    from stary.jira_adapter import JiraAdapter

    cfg = context.op_config
    jira_token = os.environ.get("JIRA_TOKEN", "")
    if not jira_token:
        raise RuntimeError("JIRA_TOKEN environment variable is not set")

    with tracer.start_as_current_span("dagster.op.lifecycle") as span:
        span.set_attribute("ticket.key", cfg["ticket_key"])
        jira = JiraAdapter(base_url=cfg["jira_base_url"], token=jira_token)
        agent = LifecycleAgent(jira=jira)

        pr_urls = review_result.get("pr_urls", [])
        reviews = review_result.get("reviews", [])
        all_approved = all(r.get("approved") for r in reviews) if reviews else False
        any_merged = any(r.get("merged") for r in reviews) if reviews else False

        result = agent.run(
            ticket_key=cfg["ticket_key"],
            pr_urls=pr_urls,
            all_approved=all_approved,
            merged=any_merged,
        )
        context.log.info(
            "Lifecycle for %s: transitioned=%s",
            cfg["ticket_key"], result.get("transitioned"),
        )
    return {**review_result, **result}


# ---------------------------------------------------------------------------
# Sensor helpers: Jira ticket status markers
# ---------------------------------------------------------------------------

@op(
    description="Mark a Jira ticket as work-in-progress.",
    config_schema={
        "ticket_key": str,
        "jira_base_url": str,
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
    jira_token = os.environ.get("JIRA_TOKEN", "")
    if not jira_token:
        raise RuntimeError("JIRA_TOKEN environment variable is not set")

    with tracer.start_as_current_span("dagster.op.mark_ticket_wip") as span:
        span.set_attribute("ticket.key", cfg["ticket_key"])
        jira = JiraAdapter(base_url=cfg["jira_base_url"], token=jira_token)
        status_marker = TicketStatusMarker(jira)

        # Build Dagster run URL when possible
        dagster_base_url = get_dagster_base_url()
        run_id = context.run_id
        dagster_run_url = build_dagster_run_url(dagster_base_url, run_id)

        status_marker.mark_wip(
            cfg["ticket_key"],
            dagster_run_url=dagster_run_url,
            mention_user=cfg.get("trigger_author", ""),
        )
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
        "trigger_author": Field(str, default_value="", is_required=False),
    },
    ins={"review_result": In(Dict)},
    out=Out(Nothing),
)
def mark_ticket_done(context: OpExecutionContext, review_result: Dict) -> None:
    """Add a done-marker comment on the Jira ticket."""
    from stary.jira_adapter import JiraAdapter
    from stary.ticket_status import TicketStatusMarker

    cfg = context.op_config
    jira_token = os.environ.get("JIRA_TOKEN", "")
    if not jira_token:
        raise RuntimeError("JIRA_TOKEN environment variable is not set")

    with tracer.start_as_current_span("dagster.op.mark_ticket_done") as span:
        span.set_attribute("ticket.key", cfg["ticket_key"])
        jira = JiraAdapter(base_url=cfg["jira_base_url"], token=jira_token)
        status_marker = TicketStatusMarker(jira)

        pr_urls = review_result.get("pr_urls", [])
        reviews = review_result.get("reviews", [])
        pr_summary = ", ".join(pr_urls) if pr_urls else "N/A"
        status_marker.mark_done(
            cfg["ticket_key"],
            pr_url=pr_summary,
            status=cfg["status"],
            reviews=reviews or None,
            mention_user=cfg.get("trigger_author", ""),
        )
        context.log.info(
            "Marked %s as done (status=%s, prs=%s)",
            cfg["ticket_key"],
            cfg["status"],
            pr_summary,
        )
