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

import sys
import os
from stary.orchestrator import Orchestrator


def main():
    orch = Orchestrator(
        repo_path=os.environ.get("REPO_PATH"),
    )

    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if args:
        # Direct ticket input (XML path or URL)
        ticket_input = args[0]
        result = orch.run(ticket_input)
        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
    elif "--once" in sys.argv:
        try:
            orch.poll_once()
        except Exception as exc:
            print(f"[main] poll_once failed: {exc}")
            sys.exit(1)
    else:
        orch.run_forever()


if __name__ == "__main__":
    main()
