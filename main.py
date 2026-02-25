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
"""

import sys
import os
from orchestrator import Orchestrator


def main():
    orch = Orchestrator(
        inference_url=os.environ.get("INFERENCE_URL"),
        agent1_inference_url=os.environ.get("AGENT1_INFERENCE_URL"),
        agent2_inference_url=os.environ.get("AGENT2_INFERENCE_URL"),
        agent3_inference_url=os.environ.get("AGENT3_INFERENCE_URL"),
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
