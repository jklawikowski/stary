"""Dagster jobs for the Stary agent pipeline.

Mirrors the pattern from the qa-platform orchestrator
(src/vistula/dagster/defs/jobs.py).

Jobs:
    stary_pipeline          – Full agent pipeline: read ticket → plan → implement → review
    stary_pipeline_with_wip – Same pipeline but with Jira WIP/done markers
"""

import logging
import os
from datetime import timedelta

from dagster import MAX_RUNTIME_SECONDS_TAG, graph, job

from stary.dagster.defs.ops import (
    implement_feature,
    mark_ticket_done,
    mark_ticket_wip,
    plan_tasks,
    read_jira_ticket,
    review_code,
)

DEFAULT_TIMEOUT = f"{timedelta(minutes=30).total_seconds()}"
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core pipeline graph: TaskReader → Planner → Implementer → Reviewer
# ---------------------------------------------------------------------------

@graph
def stary_pipeline_graph() -> None:
    """Full Stary agent pipeline.

    TaskReader → Planner (per repo) → Implementer (per repo) → Reviewer
    """
    task_input = read_jira_ticket()
    plan_result = plan_tasks(task_input)
    impl_result = implement_feature(plan_result)
    review_code(impl_result)


stary_pipeline = stary_pipeline_graph.to_job(
    name="stary_pipeline",
    description=(
        "End-to-end Stary pipeline: reads a Jira ticket, plans & validates tasks, "
        "implements features via LLM-generated code, creates a PR, and performs "
        "an automated code review."
    ),
    tags={MAX_RUNTIME_SECONDS_TAG: DEFAULT_TIMEOUT},
)


# ---------------------------------------------------------------------------
# Pipeline with Jira WIP/done markers (sensor-driven)
# ---------------------------------------------------------------------------

@graph
def stary_pipeline_with_markers_graph() -> None:
    """Stary pipeline with Jira WIP/done status markers.

    mark_ticket_wip → TaskReader → Planner → Implementer → Reviewer → mark_ticket_done
    """
    wip_done = mark_ticket_wip()
    task_input = read_jira_ticket(after=wip_done)
    plan_result = plan_tasks(task_input)
    impl_result = implement_feature(plan_result)
    review_result = review_code(impl_result)
    mark_ticket_done(review_result=review_result)


stary_pipeline_with_markers = stary_pipeline_with_markers_graph.to_job(
    name="stary_pipeline_with_markers",
    description=(
        "Sensor-driven Stary pipeline with Jira WIP/done markers. "
        "Marks ticket as WIP, runs the full agent pipeline, then marks done."
    ),
    tags={MAX_RUNTIME_SECONDS_TAG: f"{timedelta(hours=1).total_seconds()}"},
)
