"""GitHub Copilot SDK inference backend.

This module implements the InferenceClient interface using the
GitHub Copilot SDK for LLM completions, including native tool-calling
support.

Each ``chat()`` / ``chat_with_tools()`` call spins up a fresh
``asyncio.run()`` with its own CopilotClient so there is zero
interaction with the caller's event-loop or signal handlers.

Configuration (environment variables):
    COPILOT_GITHUB_TOKEN or GH_TOKEN: GitHub token for authentication
    COPILOT_MODEL: Model to use (default: gpt-4o)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

from stary.inference.base import BaseInferenceClient, ToolDefinition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gpt-4o"


def _get_github_token() -> str:
    token = os.environ.get("COPILOT_GITHUB_TOKEN") or os.environ.get("GH_TOKEN", "")
    if not token:
        raise RuntimeError(
            "No GitHub token found for Copilot SDK. "
            "Set COPILOT_GITHUB_TOKEN or GH_TOKEN environment variable."
        )
    return token


def _get_model() -> str:
    return os.environ.get("COPILOT_MODEL", DEFAULT_MODEL).strip()


# ---------------------------------------------------------------------------
# Standalone async helper – simple chat (no tools)
# ---------------------------------------------------------------------------

async def _run_chat(
    github_token: str,
    model: str,
    full_prompt: str,
    timeout: float,
    disable_tools: bool = True,
) -> str:
    """Create a CopilotClient, send a prompt, collect the response, tear down."""
    from typing import Any
    from copilot import CopilotClient

    client = CopilotClient({"github_token": github_token})
    logger.info("Starting CopilotClient")
    await client.start()
    logger.info("CopilotClient started")

    try:
        session_cfg: dict[str, Any] = {
            "model": model,
            "reasoning_effort": "medium",
            "on_permission_request": lambda req, inv: {"kind": "approved"},
        }
        if disable_tools:
            session_cfg["available_tools"] = ["_disabled_"]
        logger.info(
            "Creating session (model=%s, tools=%s)",
            model, "off" if disable_tools else "on",
        )
        session = await client.create_session(session_cfg)
        logger.info("Session created")

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
            elif etype == "session.idle":
                done.set()
            elif etype in ("error", "session.error"):
                detail = getattr(event.data, "message", None) or repr(event.data)
                error_msg.append(detail)
                logger.error("SDK error event (%s): %s", etype, detail)
                done.set()

        session.on(on_event)

        logger.info("Sending prompt (%d chars)", len(full_prompt))
        logger.debug("Prompt content:\n%s", full_prompt)
        await session.send({"prompt": full_prompt})

        try:
            await asyncio.wait_for(done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Timeout after %ss. Events: %s", timeout, all_events,
            )
            await session.destroy()
            raise RuntimeError(f"Copilot SDK timed out after {timeout}s")

        await session.destroy()

        if error_msg:
            raise RuntimeError(f"Copilot SDK error: {error_msg[0]}")

        combined = "".join(result_content)
        logger.info(
            "Response: %d chars, %d msg(s), events: %s",
            len(combined), len(result_content), all_events,
        )
        logger.debug("Response content:\n%s", combined)
        return combined

    finally:
        await client.stop()


# ---------------------------------------------------------------------------
# Async helper – chat with custom tools
# ---------------------------------------------------------------------------

def _make_sdk_tool(tool_def: ToolDefinition):
    """Convert a ToolDefinition into a ``copilot.types.Tool``.

    The SDK expects Tool objects with a handler that receives a
    ``ToolInvocation`` dict and returns a ``ToolResult`` TypedDict.
    """
    from copilot.types import Tool as SdkTool, ToolResult

    def _handler(invocation):
        raw_args = invocation.get("arguments") or {}
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                raw_args = {}
        try:
            logger.debug("Tool call: %s(%s)", tool_def.name, raw_args)
            result_text = tool_def.handler(**raw_args)
            logger.debug(
                "Tool result: %s() -> %d chars", tool_def.name, len(result_text),
            )
            return ToolResult(textResultForLlm=result_text, resultType="success")
        except Exception as exc:
            logger.error("Tool error (%s): %s", tool_def.name, exc)
            return ToolResult(
                textResultForLlm=f"Error executing tool '{tool_def.name}': {exc}",
                resultType="failure",
                error=str(exc),
            )

    schema = tool_def.to_openai_schema()
    params = schema.get("function", {}).get("parameters")
    return SdkTool(
        name=tool_def.name,
        description=tool_def.description,
        handler=_handler,
        parameters=params,
    )


async def _run_chat_with_tools(
    github_token: str,
    model: str,
    system: str,
    user: str,
    tools: list[ToolDefinition],
    timeout: float,
    max_iterations: int,
) -> str:
    """Run a tool-calling conversation via the Copilot SDK.

    Converts ToolDefinition objects to native SDK Tool objects and lets
    the SDK handle the tool-execution loop internally via
    ``send_and_wait()``.
    """
    from typing import Any
    from copilot import CopilotClient

    client = CopilotClient({"github_token": github_token})
    await client.start()

    sdk_tools = [_make_sdk_tool(t) for t in tools]

    try:
        session_cfg: dict[str, Any] = {
            "model": model,
            "reasoning_effort": "medium",
            "tools": sdk_tools,
            "available_tools": [],  # disable built-in tools
            "on_permission_request": lambda req, inv: {"kind": "approved"},
            "system_message": {"mode": "replace", "content": system},
        }
        logger.info(
            "Creating tool session (model=%s, %d custom tool(s))",
            model, len(tools),
        )
        session = await client.create_session(session_cfg)

        logger.info("Sending prompt (%d chars) with tools", len(user))
        logger.debug("System prompt:\n%s", system)
        logger.debug("User prompt:\n%s", user)
        response = await session.send_and_wait(
            {"prompt": user},
            timeout=timeout,
        )

        final = ""
        if response and hasattr(response.data, "content"):
            final = response.data.content or ""

        await session.destroy()
        logger.info("Tool session done: %d chars", len(final))
        logger.debug("Tool session response:\n%s", final)
        return final

    finally:
        await client.stop()


# ---------------------------------------------------------------------------
# Synchronous client class
# ---------------------------------------------------------------------------

class CopilotInferenceClient(BaseInferenceClient):
    """Copilot SDK-based inference client (fully synchronous public API).

    Every call runs the SDK inside a throwaway ``asyncio.run()`` so
    there are no leftover event-loops or signal handlers.
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
        logger.info("Configured with model=%s", self.model)

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        timeout: float = 300.0,
    ) -> str:
        full_prompt = f"{system}\n\n---\n\n{user}"
        t0 = time.monotonic()
        try:
            result = asyncio.run(
                _run_chat(
                    self.github_token, self.model, full_prompt, timeout,
                    disable_tools=self.disable_tools,
                )
            )
            logger.info("chat() completed in %.1fs", time.monotonic() - t0)
            return result
        except Exception as exc:
            logger.error(
                "chat() failed after %.1fs: %s",
                time.monotonic() - t0, exc, exc_info=True,
            )
            raise

    def chat_with_tools(
        self,
        system: str,
        user: str,
        tools: list[ToolDefinition],
        temperature: float = 0.2,
        timeout: float = 900.0,
        max_iterations: int = 30,
    ) -> str:
        """Native SDK tool-calling (overrides prompt-based fallback)."""
        t0 = time.monotonic()
        try:
            result = asyncio.run(
                _run_chat_with_tools(
                    self.github_token, self.model,
                    system, user, tools,
                    timeout, max_iterations,
                )
            )
            logger.info(
                "chat_with_tools() completed in %.1fs", time.monotonic() - t0,
            )
            return result
        except Exception as exc:
            logger.error(
                "chat_with_tools() failed after %.1fs: %s",
                time.monotonic() - t0, exc, exc_info=True,
            )
            raise
