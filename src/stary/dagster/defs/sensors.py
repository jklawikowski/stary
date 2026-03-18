"""Dagster sensors for the Stary agent pipeline.

Mirrors the pattern from the qa-platform orchestrator
(src/vistula/dagster/defs/sensors.py).

Sensors:
    jira_do_it_sensor    – Polls Jira for "do it" triggered tickets
    jira_pr_only_sensor  – Polls Jira for "pull request" triggered tickets
    jira_retry_sensor    – Polls Jira for retry triggered tickets
"""

import logging
import os
from typing import Generator

from dagster import (
    RunFailureSensorContext,
    RunRequest,
    run_failure_sensor,
    sensor,
)

from stary.config import build_dagster_run_url, get_dagster_base_url
from stary.dagster.defs.jobs import stary_pipeline, stary_pipeline_with_markers
from stary.jira_adapter import JiraAdapter
from stary.sensor import TriggerConfig, TriggerDetector, TriggeredTicket

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (sourced from environment)
# ---------------------------------------------------------------------------
JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL", "https://jira.devtools.intel.com")
JIRA_TOKEN = os.environ.get("JIRA_TOKEN", "")


def _build_trigger_config() -> TriggerConfig:
    """Build TriggerConfig from environment variables."""
    query_span_days = int(os.environ.get("STARY_QUERY_SPAN_DAYS", "1"))
    jira_labels_raw = os.environ.get("STARY_JIRA_LABELS", "")
    jira_labels = [l.strip() for l in jira_labels_raw.split(",") if l.strip()] or None
    return TriggerConfig(query_span_days=query_span_days, jira_labels=jira_labels)


def _yield_run_requests(
    triggered: list[TriggeredTicket],
    jira_base_url: str,
) -> Generator:
    """Yield RunRequests for a list of triggered tickets."""
    for ticket in triggered:
        ticket_key = ticket.key
        ticket_url = ticket.url
        auto_merge = ticket.auto_merge
        retry_count = ticket.retry_count

        run_key = f"stary-{ticket_key}-{retry_count}"

        run_config = {
            "ops": {
                "mark_ticket_wip": {
                    "config": {
                        "ticket_key": ticket_key,
                        "jira_base_url": jira_base_url,
                    }
                },
                "read_jira_ticket": {
                    "config": {
                        "ticket_url": ticket_url,
                        "jira_base_url": jira_base_url,
                    }
                },
                "review_code": {
                    "config": {
                        "auto_merge": auto_merge,
                    }
                },
                "mark_ticket_done": {
                    "config": {
                        "ticket_key": ticket_key,
                        "status": "COMPLETED",
                        "jira_base_url": jira_base_url,
                    }
                },
            }
        }

        yield RunRequest(
            run_key=run_key,
            run_config=run_config,
            tags={"ticket_key": ticket_key, "retry_count": str(retry_count)},
        )


# ---------------------------------------------------------------------------
# Jira ticket sensors (split by trigger type)
# ---------------------------------------------------------------------------

@sensor(
    job=stary_pipeline_with_markers,
    name="jira_do_it_sensor",
    minimum_interval_seconds=1800,
    description=(
        "Polls Jira for tickets with the 'do it' trigger comment. "
        "Yields a RunRequest for each triggered ticket (auto_merge=True)."
    ),
)
def jira_do_it_sensor() -> Generator:
    """Dagster sensor for 'do it' triggers."""
    jira_base_url = os.environ.get("JIRA_BASE_URL", JIRA_BASE_URL)
    jira_token = os.environ.get("JIRA_TOKEN", JIRA_TOKEN)

    jira = JiraAdapter(base_url=jira_base_url, token=jira_token)
    detector = TriggerDetector(jira, config=_build_trigger_config())
    triggered = detector.poll_do_it()

    yield from _yield_run_requests(triggered, jira_base_url)


@sensor(
    job=stary_pipeline_with_markers,
    name="jira_pr_only_sensor",
    minimum_interval_seconds=1800,
    description=(
        "Polls Jira for tickets with the 'pull request' trigger comment. "
        "Yields a RunRequest for each triggered ticket (auto_merge=False)."
    ),
)
def jira_pr_only_sensor() -> Generator:
    """Dagster sensor for 'pull request' triggers."""
    jira_base_url = os.environ.get("JIRA_BASE_URL", JIRA_BASE_URL)
    jira_token = os.environ.get("JIRA_TOKEN", JIRA_TOKEN)

    jira = JiraAdapter(base_url=jira_base_url, token=jira_token)
    detector = TriggerDetector(jira, config=_build_trigger_config())
    triggered = detector.poll_pr_only()

    yield from _yield_run_requests(triggered, jira_base_url)


@sensor(
    job=stary_pipeline_with_markers,
    name="jira_retry_sensor",
    minimum_interval_seconds=1800,
    description=(
        "Polls Jira for tickets with the 'retry' trigger comment. "
        "Yields a RunRequest for each triggered ticket after validating retry count."
    ),
)
def jira_retry_sensor() -> Generator:
    """Dagster sensor for 'retry' triggers."""
    jira_base_url = os.environ.get("JIRA_BASE_URL", JIRA_BASE_URL)
    jira_token = os.environ.get("JIRA_TOKEN", JIRA_TOKEN)

    jira = JiraAdapter(base_url=jira_base_url, token=jira_token)
    detector = TriggerDetector(jira, config=_build_trigger_config())
    triggered = detector.poll_retry()

    yield from _yield_run_requests(triggered, jira_base_url)


# ---------------------------------------------------------------------------
# Failure sensor
# ---------------------------------------------------------------------------


def _extract_root_cause(error) -> str:
    """Extract the root cause error message from a Dagster error.

    Dagster wraps the actual exception in DagsterExecutionStepExecutionError.
    We want the underlying cause, not the wrapper.

    Args:
        error: SerializableErrorInfo from Dagster

    Returns:
        The root cause error message
    """
    if not error:
        return "Unknown error"

    # The error has a cause chain - traverse to find the root cause
    # SerializableErrorInfo has: message, stack, cls_name, cause, context
    current = error
    while current.cause:
        current = current.cause

    # Build a meaningful message from the root cause
    # Format: "ExceptionClass: message"
    cls_name = current.cls_name if hasattr(current, "cls_name") else ""
    message = current.message if hasattr(current, "message") else str(current)

    if cls_name and message:
        # Remove the cls_name prefix if it's already in the message
        if message.startswith(cls_name):
            return message
        return f"{cls_name}: {message}"
    elif message:
        return message
    elif cls_name:
        return cls_name
    return "Unknown error"


def _extract_failure_info(context: RunFailureSensorContext) -> tuple[str, str]:
    """Extract failed step name and error message from failure context.

    The context.failure_event is the RUN_FAILURE event, not STEP_FAILURE.
    We need to query the instance for step failure events to get details.

    Returns:
        (failed_step, error_message) tuple
    """
    from dagster import DagsterEventType

    failed_step = "unknown_step"
    error_message = "Unknown error"

    run_id = context.dagster_run.run_id

    try:
        # Query instance for STEP_FAILURE events
        records = context.instance.get_records_for_run(
            run_id=run_id,
            of_type=DagsterEventType.STEP_FAILURE,
        ).records

        if records:
            # Get the last step failure event
            last_failure = records[-1]
            event_log_entry = last_failure.event_log_entry
            dagster_event = event_log_entry.dagster_event

            if dagster_event:
                # Get step name
                if dagster_event.step_key:
                    failed_step = dagster_event.step_key

                # Get error from event_specific_data
                event_data = dagster_event.event_specific_data
                if event_data and hasattr(event_data, "error"):
                    error = event_data.error
                    if error:
                        # Extract the root cause, not the Dagster wrapper
                        error_message = _extract_root_cause(error)
                        # Truncate if too long (keep it short for Jira comment)
                        if len(error_message) > 300:
                            error_message = error_message[:297] + "..."
    except Exception as e:
        logger.warning("Could not extract failure details from run %s: %s", run_id, e)

    return failed_step, error_message


@run_failure_sensor(
    monitored_jobs=[stary_pipeline_with_markers, stary_pipeline],
    name="monitor_stary_failures",
)
def monitor_stary_failures(context: RunFailureSensorContext) -> None:
    """Monitor Stary pipeline failures and post failure info to Jira.

    Extracts the failed step and error message, then adds a failure
    marker comment to the Jira ticket. Full traceback is available
    in Dagster logs.
    """
    dagster_run = context.dagster_run
    dagster_base_url = get_dagster_base_url()
    run_id = dagster_run.run_id
    dagster_run_url = build_dagster_run_url(dagster_base_url, run_id)
    # Fallback to legacy DAGIT_UI_URL if DAGSTER_BASE_URL is not set
    if not dagster_run_url:
        dagit_url = os.getenv("DAGIT_UI_URL", "http://localhost:3000")
        dagster_run_url = f"{dagit_url}/runs/{run_id}"
    pipeline_name = dagster_run.job_name

    logger.error(
        "Dagster job failed: %s | Run URL: %s",
        pipeline_name,
        dagster_run_url,
    )

    # Extract ticket_key from run config
    run_config = dagster_run.run_config or {}
    ops_config = run_config.get("ops", {})
    ticket_key = None

    # Try to get ticket_key from mark_ticket_wip or mark_ticket_done config
    for op_name in ("mark_ticket_wip", "mark_ticket_done"):
        op_config = ops_config.get(op_name, {}).get("config", {})
        if "ticket_key" in op_config:
            ticket_key = op_config["ticket_key"]
            break

    if not ticket_key:
        logger.warning(
            "Cannot post failure to Jira: ticket_key not found in run_config"
        )
        return

    # Extract failure details
    failed_step, error_message = _extract_failure_info(context)

    # Get Jira credentials from environment
    jira_base_url = ops_config.get("mark_ticket_wip", {}).get(
        "config", {}
    ).get("jira_base_url") or os.environ.get("JIRA_BASE_URL", JIRA_BASE_URL)
    jira_token = os.environ.get("JIRA_TOKEN", JIRA_TOKEN)
    if not jira_token:
        logger.error("Cannot post failure to Jira: JIRA_TOKEN not set in environment")
        return

    # Post failure marker to Jira
    try:
        from stary.jira_adapter import JiraAdapter
        from stary.ticket_status import TicketStatusMarker

        jira = JiraAdapter(base_url=jira_base_url, token=jira_token)
        status_marker = TicketStatusMarker(jira)
        status_marker.mark_failed(
            ticket_key=ticket_key,
            failed_step=failed_step,
            error_message=error_message,
            dagster_run_url=dagster_run_url,
        )
        logger.info(
            "Posted failure marker to Jira ticket %s (step: %s)",
            ticket_key,
            failed_step,
        )
    except Exception as exc:
        logger.error(
            "Failed to post failure marker to Jira ticket %s: %s",
            ticket_key,
            exc,
        )
