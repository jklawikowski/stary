#!/usr/bin/env python3
"""Entry point – run the orchestrator.

Usage:
    # Sensor-driven loop (polls Jira continuously):
    python main.py

    # Single poll cycle:
    python main.py --once

    # Legacy: run pipeline for a single XML file or ticket URL:
    python main.py sample_ticket_multiply.xml
    python main.py https://jira.devtools.intel.com/browse/PROJ-123

Environment variables:
    COPILOT_GITHUB_TOKEN or GH_TOKEN: GitHub token for Copilot SDK authentication
    COPILOT_MODEL: Model to use (default: gpt-4o)
    JIRA_BASE_URL: Jira server URL
    JIRA_TOKEN: Jira API token
    GITHUB_TOKEN: GitHub token for PR operations
"""

import logging
import logging.config
import json
import sys
import os
from pathlib import Path
from stary.orchestrator import Orchestrator
from stary.telemetry import init_telemetry

logger = logging.getLogger(__name__)

_LOGGING_JSON = Path(__file__).resolve().parent.parent.parent / "dagster" / "logging.json"


def _configure_logging() -> None:
    """Bootstrap logging for the standalone CLI entry point."""
    if _LOGGING_JSON.is_file():
        with open(_LOGGING_JSON) as fh:
            config = json.load(fh)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=os.getenv("STARY_LOG_LEVEL", "INFO"),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )
    override = os.getenv("STARY_LOG_LEVEL")
    if override:
        stary_logger = logging.getLogger("stary")
        stary_logger.setLevel(override)
        for name in logging.root.manager.loggerDict:
            if name.startswith("stary."):
                logging.getLogger(name).setLevel(override)


def main():
    _configure_logging()
    init_telemetry()
    orch = Orchestrator(
        repo_path=os.environ.get("REPO_PATH"),
    )

    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if args:
        # Direct ticket input (XML path or URL)
        ticket_input = args[0]
        result = orch.run(ticket_input)
        logger.info("=" * 60)
        logger.info("FINAL RESULT")
        logger.info("=" * 60)
    elif "--once" in sys.argv:
        try:
            orch.poll_once()
        except Exception as exc:
            logger.error("poll_once failed: %s", exc)
            sys.exit(1)
    else:
        orch.run_forever()


if __name__ == "__main__":
    main()
