"""Dagster sensors for the Stary agent pipeline.

Sensors:
    stary_comment_sensor     – Unified sensor for do_it + pr_only + retry triggers
    stary_users_sensor        – Polls for tickets assigned to / reported by configured users
    monitor_stary_failures   – Monitors pipeline failures and posts to Jira
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Generator

from dagster import (
    RunFailureSensorContext,
    RunRequest,
    SensorEvaluationContext,
    run_failure_sensor,
    sensor,
)

from stary.config import build_dagster_run_url, get_dagster_base_url
from stary.dagster.defs.jobs import stary_pipeline, stary_pipeline_with_markers
from stary.jira_adapter import JiraAdapter
from stary.sensor import TriggerConfig, TriggerDetector, TicketStateValidator, TriggeredTicket
from stary.telemetry import tracer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (sourced from environment)
# ---------------------------------------------------------------------------
JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL", "https://jira.devtools.intel.com")
JIRA_TOKEN = os.environ.get("JIRA_TOKEN", "")


# ---------------------------------------------------------------------------
# Cursor helpers – persist processed ticket keys across evaluations
# ---------------------------------------------------------------------------
CURSOR_RETENTION_DAYS = 7


def _load_cursor(
    cursor_raw: str | None,
) -> dict[str, str]:
    """Deserialize the cursor JSON and prune entries older than retention."""
    if not cursor_raw:
        return {}
    try:
        data: dict[str, str] = json.loads(cursor_raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    cutoff = (datetime.now(timezone.utc) - timedelta(days=CURSOR_RETENTION_DAYS)).isoformat()
    return {k: v for k, v in data.items() if v > cutoff}


def _save_cursor(
    processed: dict[str, str],
) -> str:
    """Serialize the processed-tickets dict to a cursor string."""
    return json.dumps(processed)


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
                        "trigger_author": ticket.trigger_author,
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
                "manage_ticket_lifecycle": {
                    "config": {
                        "ticket_key": ticket_key,
                        "jira_base_url": jira_base_url,
                    }
                },
                "mark_ticket_done": {
                    "config": {
                        "ticket_key": ticket_key,
                        "status": "COMPLETED",
                        "jira_base_url": jira_base_url,
                        "trigger_author": ticket.trigger_author,
                    }
                },
            }
        }

        yield RunRequest(
            run_key=run_key,
            run_config=run_config,
            tags={
                "ticket_key": ticket_key,
                "retry_count": str(retry_count),
                "trigger_type": "retry" if retry_count > 0 else "comment",
            },
        )


# ---------------------------------------------------------------------------
# Unified comment-triggered sensor (replaces do_it + pr_only + retry)
# ---------------------------------------------------------------------------

@sensor(
    job=stary_pipeline_with_markers,
    name="stary_comment_sensor",
    minimum_interval_seconds=3600,
    description=(
        "Unified sensor for do_it, pr_only, and retry triggers. "
        "Runs 3 separate JQL queries, deduplicates candidates, "
        "validates ticket state via comment history, and yields "
        "RunRequests for eligible tickets."
    ),
)
def stary_comment_sensor(context: SensorEvaluationContext) -> Generator:
    """Unified Dagster sensor for comment-based triggers."""
    span = tracer.start_span("dagster.sensor.comment_sensor")
    try:
        jira_base_url = os.environ.get("JIRA_BASE_URL", JIRA_BASE_URL)
        jira_token = os.environ.get("JIRA_TOKEN", JIRA_TOKEN)

        jira = JiraAdapter(base_url=jira_base_url, token=jira_token)
        config = _build_trigger_config()
        detector = TriggerDetector(jira, config=config)
        validator = TicketStateValidator(config=config)

        # Run 3 JQL queries, deduplicated
        candidates = detector.poll_comment_triggers()

        # Load shared cursor
        processed = _load_cursor(context.cursor)
        now = datetime.now(timezone.utc).isoformat()

        new_triggered: list[TriggeredTicket] = []

        for ticket_key, ticket_url, trigger_hint in candidates:
            # Cursor optimization: skip recently processed (non-retry)
            if trigger_hint != "retry_candidate" and ticket_key in processed:
                logger.info("Skipping %s (cursor hit at %s)", ticket_key, processed[ticket_key])
                continue

            # Fetch comments and validate state
            comments = jira.get_comments(ticket_key)
            trigger_type, retry_count, auto_merge, trigger_author = (
                validator.resolve_trigger(
                    comments, trigger_hint,
                )
            )

            if trigger_type is None:
                logger.info(
                    "%s: not eligible (hint=%s, state determined from comments)",
                    ticket_key, trigger_hint,
                )
                continue

            # For retry, check cursor with retry-count key
            cursor_key = f"{ticket_key}-{retry_count}" if trigger_type == "retry" else ticket_key
            if cursor_key in processed:
                logger.info("Skipping %s (cursor key %s hit)", ticket_key, cursor_key)
                continue

            logger.info(
                "Triggered: %s (type=%s, retry=%d, auto_merge=%s)",
                ticket_key, trigger_type, retry_count, auto_merge,
            )
            processed[cursor_key] = now
            new_triggered.append(
                TriggeredTicket(
                    key=ticket_key,
                    url=ticket_url,
                    auto_merge=auto_merge,
                    retry_count=retry_count,
                    trigger_author=trigger_author,
                )
            )

        context.update_cursor(_save_cursor(processed))
        yield from _yield_run_requests(new_triggered, jira_base_url)
    finally:
        span.end()


def _parse_schedule_users() -> list[str]:
    """Parse STARY_SCHEDULE_USERS env var into a list of Jira usernames."""
    raw = os.environ.get("STARY_SCHEDULE_USERS", "")
    return [u.strip() for u in raw.split(",") if u.strip()]


@sensor(
    job=stary_pipeline_with_markers,
    name="stary_users_sensor",
    minimum_interval_seconds=21600,
    description=(
        "Polls Jira every ~6 hours for tickets assigned to or reported "
        "by users listed in STARY_SCHEDULE_USERS. Only processes tickets "
        "that have NEVER been touched by stary (no wip/done/failed markers)."
    ),
)
def stary_users_sensor(context: SensorEvaluationContext) -> Generator:
    """Dagster sensor for user-based automatic ticket discovery."""
    users = _parse_schedule_users()
    if not users:
        logger.info("STARY_SCHEDULE_USERS is empty — skipping")
        return

    span = tracer.start_span("dagster.sensor.users_sensor")
    try:
        jira_base_url = os.environ.get("JIRA_BASE_URL", JIRA_BASE_URL)
        jira_token = os.environ.get("JIRA_TOKEN", JIRA_TOKEN)

        jira = JiraAdapter(base_url=jira_base_url, token=jira_token)
        config = _build_trigger_config()
        detector = TriggerDetector(jira, config=config)
        validator = TicketStateValidator(config=config)

        candidates = detector.poll_scheduled_candidates(users)

        processed = _load_cursor(context.cursor)
        now = datetime.now(timezone.utc).isoformat()

        new_triggered: list[TriggeredTicket] = []

        for ticket_key, ticket_url in candidates:
            if ticket_key in processed:
                logger.info("Skipping %s (cursor hit at %s)", ticket_key, processed[ticket_key])
                continue

            # Fetch comments and verify ticket was never touched by stary
            comments = jira.get_comments(ticket_key)
            if not validator.resolve_scheduled(comments):
                logger.info("%s: already handled by stary, skipping", ticket_key)
                # Still add to cursor so we don't re-fetch comments next time
                processed[ticket_key] = now
                continue

            logger.info("Triggered: %s (type=scheduled, auto_merge=False)", ticket_key)
            processed[ticket_key] = now
            new_triggered.append(
                TriggeredTicket(
                    key=ticket_key,
                    url=ticket_url,
                    auto_merge=False,
                )
            )

        context.update_cursor(_save_cursor(processed))
        yield from _yield_run_requests(new_triggered, jira_base_url)
    finally:
        span.end()


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

    # Extract trigger_author from run config
    trigger_author = ""
    for op_name in ("mark_ticket_wip", "mark_ticket_done"):
        op_config = ops_config.get(op_name, {}).get("config", {})
        if "trigger_author" in op_config:
            trigger_author = op_config["trigger_author"]
            break

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

        with tracer.start_as_current_span("dagster.sensor.failure_marker") as span:
            span.set_attribute("ticket.key", ticket_key)
            jira = JiraAdapter(base_url=jira_base_url, token=jira_token)
            status_marker = TicketStatusMarker(jira)
            status_marker.mark_failed(
                ticket_key=ticket_key,
                failed_step=failed_step,
                error_message=error_message,
                dagster_run_url=dagster_run_url,
                mention_user=trigger_author,
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
