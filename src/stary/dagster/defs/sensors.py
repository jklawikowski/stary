"""Dagster sensors for the Stary agent pipeline.

Mirrors the pattern from the qa-platform orchestrator
(src/vistula/dagster/defs/sensors.py).

Sensors:
    jira_ticket_sensor – Polls Jira for triggered tickets and yields RunRequests
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

from stary.dagster.defs.jobs import stary_pipeline, stary_pipeline_with_markers

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (sourced from environment)
# ---------------------------------------------------------------------------
JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL", "https://jira.devtools.intel.com")
JIRA_TOKEN = os.environ.get("JIRA_TOKEN", "")
INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://localhost:8080/v1/chat/completions")
AGENT1_INFERENCE_URL = os.environ.get("AGENT1_INFERENCE_URL", "")
AGENT2_INFERENCE_URL = os.environ.get("AGENT2_INFERENCE_URL", "")
AGENT3_INFERENCE_URL = os.environ.get("AGENT3_INFERENCE_URL", "")


def _get_inference_url(agent_var: str) -> str:
    """Return agent-specific URL if set, otherwise fall back to INFERENCE_URL."""
    return os.environ.get(agent_var, "") or INFERENCE_URL


# ---------------------------------------------------------------------------
# Jira ticket sensor
# ---------------------------------------------------------------------------

@sensor(
    job=stary_pipeline_with_markers,
    name="jira_ticket_sensor",
    minimum_interval_seconds=60,
    description=(
        "Polls Jira for tickets with the trigger comment. "
        "Yields a RunRequest for each triggered ticket."
    ),
)
def jira_ticket_sensor() -> Generator:
    """Dagster sensor that replaces Orchestrator.run_forever().

    Uses the Sensor class to query Jira for triggered tickets and
    yields a RunRequest for each one, passing all necessary configuration
    to the ops via run_config.
    """
    from stary.sensor import Sensor

    jira_base_url = os.environ.get("JIRA_BASE_URL", JIRA_BASE_URL)
    jira_token = os.environ.get("JIRA_TOKEN", JIRA_TOKEN)

    s = Sensor(jira_base_url=jira_base_url, jira_token=jira_token)
    triggered = s.poll()

    for ticket in triggered:
        ticket_key = ticket["ticket_key"]
        ticket_url = ticket["ticket_url"]

        run_config = {
            "ops": {
                "mark_ticket_wip": {
                    "config": {
                        "ticket_key": ticket_key,
                        "jira_base_url": jira_base_url,
                        "jira_token": jira_token,
                    }
                },
                "read_jira_ticket": {
                    "config": {
                        "ticket_url": ticket_url,
                        "inference_url": _get_inference_url("AGENT1_INFERENCE_URL"),
                        "jira_base_url": jira_base_url,
                        "jira_token": jira_token,
                    }
                },
                "implement_feature": {
                    "config": {
                        "inference_url": _get_inference_url("AGENT2_INFERENCE_URL"),
                    }
                },
                "review_code": {
                    "config": {
                        "inference_url": _get_inference_url("AGENT3_INFERENCE_URL"),
                    }
                },
                "mark_ticket_done": {
                    "config": {
                        "ticket_key": ticket_key,
                        "status": "COMPLETED",
                        "jira_base_url": jira_base_url,
                        "jira_token": jira_token,
                    }
                },
            }
        }

        yield RunRequest(
            run_key=f"stary-{ticket_key}",
            run_config=run_config,
            tags={"ticket_key": ticket_key},
        )


# ---------------------------------------------------------------------------
# Failure sensor
# ---------------------------------------------------------------------------

@run_failure_sensor
def monitor_stary_failures(context: RunFailureSensorContext) -> None:
    """Monitor Stary pipeline failures.

    Logs failure details. Can be extended to send notifications
    (e.g. MS Teams, Slack, email).
    """
    dagit_url = os.getenv("DAGIT_UI_URL", "http://localhost:3000")
    run_id = context.pipeline_run.run_id
    run_url = f"{dagit_url}/runs/{run_id}"
    pipeline_name = context.pipeline_run.pipeline_name

    logger.error(
        "Dagster job failed: %s | Run URL: %s",
        pipeline_name,
        run_url,
    )
