"""GitHub Copilot SDK inference backend.

This module implements the InferenceClient interface using the
GitHub Copilot SDK for LLM completions.

Each ``chat()`` call spins up a fresh ``asyncio.run()`` with its own
CopilotClient so there is zero interaction with the caller's event-loop
or signal handlers (which was causing silent kills under Dagster's
multiprocess executor).

Configuration (environment variables):
    COPILOT_GITHUB_TOKEN or GH_TOKEN: GitHub token for authentication
    COPILOT_MODEL: Model to use (default: gpt-4o)
"""

import asyncio
import os
import traceback

from stary.inference.base import BaseInferenceClient


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gpt-4o"


def _get_github_token() -> str:
    """Get GitHub token for Copilot SDK authentication."""
    token = os.environ.get("COPILOT_GITHUB_TOKEN") or os.environ.get("GH_TOKEN", "")
    if not token:
        raise RuntimeError(
            "No GitHub token found for Copilot SDK. "
            "Set COPILOT_GITHUB_TOKEN or GH_TOKEN environment variable."
        )
    return token


def _get_model() -> str:
    """Get model name from environment or use default."""
    return os.environ.get("COPILOT_MODEL", DEFAULT_MODEL).strip()


# ---------------------------------------------------------------------------
# Standalone async helper – runs inside ``asyncio.run()``
# ---------------------------------------------------------------------------

async def _run_chat(
    github_token: str,
    model: str,
    full_prompt: str,
    timeout: float,
    disable_tools: bool = True,
) -> str:
    """Create a CopilotClient, send a prompt, collect the response, tear down.

    This function is designed to be called via ``asyncio.run()`` so it
    owns the entire event-loop lifecycle and never leaks handles.
    """
    from typing import Any

    from copilot import CopilotClient

    client = CopilotClient({"github_token": github_token})

    print("[CopilotInference] Starting CopilotClient\u2026")
    await client.start()
    print("[CopilotInference] CopilotClient started.")

    try:
        session_cfg: dict[str, Any] = {
            "model": model,
            "reasoning_effort": "high",
        }
        if disable_tools:
            # Use a whitelist that matches no real tool name so the
            # server exposes zero built-in tools to the model.
            session_cfg["available_tools"] = ["_disabled_"]
        print(f"[CopilotInference] Creating session (model={model}, tools={'off' if disable_tools else 'on'}, reasoning_effort=high)\u2026")
        session = await client.create_session(session_cfg)
        print("[CopilotInference] Session created.")

        done = asyncio.Event()
        result_content: list[str] = []
        error_msg: list[str] = []
        all_events: list[str] = []

        def on_event(event):
            etype = event.type.value if hasattr(event.type, "value") else str(event.type)
            all_events.append(etype)

            if etype == "assistant.message":
                if hasattr(event.data, "content"):
                    result_content.append(event.data.content)
                else:
                    print(
                        f"[CopilotInference] assistant.message without content: {event.data!r}"
                    )
            elif etype == "session.idle":
                done.set()
            elif etype == "error":
                detail = getattr(event.data, "message", None) or repr(event.data)
                error_msg.append(detail)
                print(f"[CopilotInference] SDK error event: {detail}")
                done.set()

        session.on(on_event)

        print(f"[CopilotInference] Sending prompt ({len(full_prompt)} chars, model={model})\u2026")
        await session.send({"prompt": full_prompt})
        print("[CopilotInference] Prompt sent, waiting for response\u2026")

        try:
            await asyncio.wait_for(done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"[CopilotInference] Timeout after {timeout}s. Events: {all_events}")
            await session.destroy()
            raise RuntimeError(
                f"Copilot SDK timed out after {timeout}s. Events: {all_events}"
            )

        await session.destroy()

        if error_msg:
            print(f"[CopilotInference] Events trace: {all_events}")
            raise RuntimeError(f"Copilot SDK error: {error_msg[0]}")

        if not result_content:
            print(f"[CopilotInference] No content received. Events: {all_events}")
            return ""

        combined = "".join(result_content)
        print(
            f"[CopilotInference] Response: {len(combined)} chars, "
            f"{len(result_content)} msg(s), events: {all_events}"
        )
        return combined

    finally:
        print("[CopilotInference] Stopping CopilotClient\u2026")
        await client.stop()
        print("[CopilotInference] CopilotClient stopped.")


class CopilotInferenceClient(BaseInferenceClient):
    """Copilot SDK-based inference client (fully synchronous public API).

    Every ``chat()`` call runs the Copilot SDK inside a throwaway
    ``asyncio.run()`` so there are no leftover event-loops or signal
    handlers that could interfere with Dagster's process management.
    """

    def __init__(
        self,
        github_token: str | None = None,
        model: str | None = None,
        disable_tools: bool = True,
    ):
        self.github_token = github_token or _get_github_token()
        self.model = model or _get_model()
        self.disable_tools = disable_tools
        print(f"[CopilotInference] Configured with model={self.model}")

    # ------------------------------------------------------------------
    # Public API (synchronous)
    # ------------------------------------------------------------------

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        timeout: float = 300.0,
    ) -> str:
        """Send a chat completion request.

        Args:
            system: System prompt
            user: User message
            temperature: Sampling temperature (0.0-1.0)
            timeout: Request timeout in seconds

        Returns:
            The assistant's response text
        """
        full_prompt = f"{system}\n\n---\n\n{user}"

        try:
            return asyncio.run(
                _run_chat(
                    self.github_token, self.model, full_prompt, timeout,
                    disable_tools=self.disable_tools,
                )
            )
        except Exception as exc:
            print(f"[CopilotInference] chat() failed: {exc}")
            traceback.print_exc()
            raise
