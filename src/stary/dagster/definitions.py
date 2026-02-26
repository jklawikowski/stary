"""Dagster code location entry point for Stary.

This module is the single entry point loaded by dagster via:
    dagster api grpc -m stary.dagster.definitions

It mirrors the pattern from the qa-platform orchestrator:
    src/vistula/dagster/definitions.py
"""

import logging
import os

from dagster import Definitions

from stary.dagster.defs.jobs import stary_pipeline, stary_pipeline_with_markers
from stary.dagster.defs.sensors import jira_ticket_sensor, monitor_stary_failures

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure logging for the dagster code location."""
    logging.basicConfig(
        level=os.getenv("STARY_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


def create_definitions() -> Definitions:
    """Create Dagster definitions.

    Used as an entrypoint to Dagster code location that is called only once.

    :returns: Dagster definitions
    """
    configure_logging()
    return Definitions(
        jobs=[stary_pipeline, stary_pipeline_with_markers],
        sensors=[jira_ticket_sensor, monitor_stary_failures],
    )


defs = create_definitions()
