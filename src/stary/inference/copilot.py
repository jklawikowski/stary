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
import os
import traceback

from stary.inference.base import BaseInferenceClient, ToolDefinition


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
    print("[CopilotInference] Starting CopilotClient\u2026")
    await client.start()
    print("[CopilotInference] CopilotClient started.")

    try:
        session_cfg: dict[str, Any] = {
            "model": model,
            "reasoning_effort": "medium",
            "on_permission_request": lambda req, inv: {"kind": "approved"},
        }
        if disable_tools:
            session_cfg["available_tools"] = ["_disabled_"]
        print(f"[CopilotInference] Creating session (model={model}, tools={'off' if disable_tools else 'on'})\u2026")
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
            elif etype == "session.idle":
                done.set()
            elif etype in ("error", "session.error"):
                detail = getattr(event.data, "message", None) or repr(event.data)
                error_msg.append(detail)
                print(f"[CopilotInference] SDK error event ({etype}): {detail}")
                done.set()

        session.on(on_event)

        print(f"[CopilotInference] Sending prompt ({len(full_prompt)} chars)\u2026")
        await session.send({"prompt": full_prompt})

        try:
            await asyncio.wait_for(done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"[CopilotInference] Timeout after {timeout}s. Events: {all_events}")
            await session.destroy()
            raise RuntimeError(f"Copilot SDK timed out after {timeout}s")

        await session.destroy()

        if error_msg:
            raise RuntimeError(f"Copilot SDK error: {error_msg[0]}")

        combined = "".join(result_content)
        print(
            f"[CopilotInference] Response: {len(combined)} chars, "
            f"{len(result_content)} msg(s), events: {all_events}"
        )
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
            result_text = tool_def.handler(**raw_args)
            print(f"[CopilotInference]   {tool_def.name}() -> {len(result_text)} chars")
            return ToolResult(textResultForLlm=result_text, resultType="success")
        except Exception as exc:
            print(f"[CopilotInference] Tool error ({tool_def.name}): {exc}")
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
        print(
            f"[CopilotInference] Creating tool session "
            f"(model={model}, {len(tools)} custom tool(s))\u2026"
        )
        session = await client.create_session(session_cfg)

        print(f"[CopilotInference] Sending prompt ({len(user)} chars) with tools\u2026")
        response = await session.send_and_wait(
            {"prompt": user},
            timeout=timeout,
        )

        final = ""
        if response and hasattr(response.data, "content"):
            final = response.data.content or ""

        await session.destroy()
        print(f"[CopilotInference] Tool session done: {len(final)} chars")
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
        print(f"[CopilotInference] Configured with model={self.model}")

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        timeout: float = 300.0,
    ) -> str:
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
        try:
            return asyncio.run(
                _run_chat_with_tools(
                    self.github_token, self.model,
                    system, user, tools,
                    timeout, max_iterations,
                )
            )
        except Exception as exc:
            print(f"[CopilotInference] chat_with_tools() failed: {exc}")
            traceback.print_exc()
            raise
