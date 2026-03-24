"""Dagster code location entry point for Stary.

This module is the single entry point loaded by dagster via:
    dagster api grpc -m stary.dagster.definitions

It mirrors the pattern from the qa-platform orchestrator:
    src/vistula/dagster/definitions.py
"""

import json
import logging
import logging.config
import os
from pathlib import Path

from dagster import Definitions

from stary.dagster.defs.jobs import stary_pipeline, stary_pipeline_with_markers
from stary.dagster.defs.sensors import (
    jira_do_it_sensor,
    jira_pr_only_sensor,
    jira_retry_sensor,
    jira_scheduled_trigger,
    monitor_stary_failures,
)
from stary.telemetry import init_telemetry

logger = logging.getLogger(__name__)

_LOGGING_JSON = Path(__file__).resolve().parent.parent.parent.parent / "dagster" / "logging.json"


def configure_logging() -> None:
    """Configure logging for the dagster code location.

    Loads ``dagster/logging.json`` when available; falls back to
    ``basicConfig`` otherwise.  The ``STARY_LOG_LEVEL`` environment
    variable overrides the level for the ``stary`` logger.
    """
    if _LOGGING_JSON.is_file():
        with open(_LOGGING_JSON) as fh:
            config = json.load(fh)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=os.getenv("STARY_LOG_LEVEL", "INFO"),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

    # Allow runtime override of the stary logger level
    override = os.getenv("STARY_LOG_LEVEL")
    if override:
        logging.getLogger("stary").setLevel(override)


def create_definitions() -> Definitions:
    """Create Dagster definitions.

    Used as an entrypoint to Dagster code location that is called only once.

    :returns: Dagster definitions
    """
    configure_logging()
    init_telemetry()
    return Definitions(
        jobs=[stary_pipeline, stary_pipeline_with_markers],
        sensors=[jira_do_it_sensor, jira_pr_only_sensor, jira_retry_sensor, monitor_stary_failures],
        schedules=[jira_scheduled_trigger],
    )


defs = create_definitions()
