"""Base interface for inference clients.

This module defines the InferenceClient protocol that all inference
backends must implement. Agents depend ONLY on this interface.

Includes tool-calling support: agents define ``ToolDefinition`` objects
with handler callbacks, and the inference client manages the multi-turn
tool-call loop automatically.

To add a new inference backend:
1. Create a new module in stary/inference/ (e.g., openai.py)
2. Implement a class that satisfies the InferenceClient protocol
3. Register it in the factory (stary/inference/factory.py)
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool-calling data types
# ---------------------------------------------------------------------------

@dataclass
class ToolParam:
    """Single parameter for a tool."""
    name: str
    type: str
    description: str
    required: bool = True
    enum: list[str] | None = None


@dataclass
class ToolDefinition:
    """Definition of a tool the LLM can call.

    ``handler`` is invoked by the inference client when the LLM
    requests this tool.  It receives keyword arguments matching
    ``parameters`` and must return a string result.
    """
    name: str
    description: str
    parameters: list[ToolParam]
    handler: Callable[..., str]

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI-style function-calling tool schema."""
        props: dict[str, dict] = {}
        required: list[str] = []
        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            props[p.name] = prop
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class InferenceClient(Protocol):
    """Protocol defining the inference client interface.

    All inference backends must implement these methods.
    Agents should type-hint with this protocol, not concrete classes.
    """

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        timeout: float = 300.0,
    ) -> str:
        """Send a chat completion request and return the text response."""
        ...

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        timeout: float = 300.0,
    ) -> dict:
        """Send a chat completion request and parse the JSON response."""
        ...

    def chat_with_tools(
        self,
        system: str,
        user: str,
        tools: list[ToolDefinition],
        temperature: float = 0.2,
        timeout: float = 900.0,
        max_iterations: int = 30,
    ) -> str:
        """Run a multi-turn tool-calling loop.

        The LLM may call any of the supplied *tools*.  The inference
        client executes the tool handler and feeds the result back,
        repeating until the LLM produces a final text response (no
        more tool calls) or *max_iterations* is reached.

        Returns the LLM's final text response.
        """
        ...

    def chat_json_with_tools(
        self,
        system: str,
        user: str,
        tools: list[ToolDefinition],
        temperature: float = 0.2,
        timeout: float = 900.0,
        max_iterations: int = 30,
    ) -> dict:
        """Like ``chat_with_tools`` but parses the final response as JSON."""
        ...


class BaseInferenceClient(ABC):
    """Abstract base class with shared utilities for inference clients.

    Concrete implementations can inherit from this to get common
    JSON parsing and tool-calling logic.  They only need to implement
    ``chat()``.
    """

    @abstractmethod
    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        timeout: float = 300.0,
    ) -> str:
        """Send a chat completion request. Must be implemented by subclasses."""
        pass

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        timeout: float = 300.0,
    ) -> dict:
        """Send a chat completion request and parse JSON response."""
        _JSON_ENFORCE = (
            "\n\nYou MUST respond with ONLY a valid JSON object — no markdown "
            "fences, no commentary, no prose. Just the raw JSON."
        )
        content = self.chat(system + _JSON_ENFORCE, user, temperature, timeout)
        return self._parse_json_response(content)

    # ------------------------------------------------------------------
    # Tool-calling (default prompt-based implementation)
    # ------------------------------------------------------------------

    _TOOL_CALL_TAG = "tool_call"

    def chat_with_tools(
        self,
        system: str,
        user: str,
        tools: list[ToolDefinition],
        temperature: float = 0.2,
        timeout: float = 900.0,
        max_iterations: int = 30,
    ) -> str:
        """Multi-turn tool-calling loop (prompt-based fallback).

        Subclasses that have native tool-calling support (e.g. via an
        SDK) should override this for better reliability and efficiency.
        """
        tool_map = {t.name: t for t in tools}
        tools_desc = self._format_tools_for_prompt(tools)

        augmented_system = (
            system + "\n\n"
            "## Available Tools\n"
            "You have the following tools available. To call a tool, "
            "include a fenced block in your response using EXACTLY this "
            "format (you may include multiple blocks):\n\n"
            f"```{self._TOOL_CALL_TAG}\n"
            '{"tool": "<tool_name>", "arguments": {<args as JSON>}}\n'
            "```\n\n"
            "After each tool call you will receive the result and can "
            "continue working. When you have finished, provide your "
            f"FINAL answer WITHOUT any ```{self._TOOL_CALL_TAG}``` blocks.\n\n"
            + tools_desc
        )

        conversation = user
        t0 = time.monotonic()
        logger.debug("System prompt:\n%s", augmented_system)
        logger.debug("User prompt:\n%s", user)

        for iteration in range(max_iterations):
            response = self.chat(augmented_system, conversation, temperature, timeout)

            calls = self._extract_tool_calls_from_text(response)
            if not calls:
                logger.debug("Final response:\n%s", response)
                logger.info(
                    "chat_with_tools() done in %.1fs (%d iterations)",
                    time.monotonic() - t0, iteration + 1,
                )
                return response  # final answer

            # Execute tool calls and append results
            results_parts: list[str] = []
            for tc_name, tc_args in calls:
                logger.debug("Tool call: %s(%s)", tc_name, tc_args)
                tool = tool_map.get(tc_name)
                if tool:
                    try:
                        result = tool.handler(**tc_args)
                    except Exception as exc:
                        result = f"Error executing tool '{tc_name}': {exc}"
                else:
                    result = f"Unknown tool: '{tc_name}'"
                logger.debug(
                    "Tool result: %s -> %d chars", tc_name, len(result),
                )
                results_parts.append(
                    f"## Tool result: {tc_name}\n{result}"
                )

            conversation += (
                f"\n\n--- Assistant ---\n{response}\n\n"
                f"--- Tool Results ---\n"
                + "\n\n".join(results_parts)
                + "\n\nContinue with your task. You may call more tools "
                "or provide your final answer."
            )

            logger.info(
                "Tool iteration %d: %d call(s) executed",
                iteration + 1, len(calls),
            )

        # Exhausted iterations — return last response
        logger.warning(
            "chat_with_tools() exhausted %d iterations in %.1fs",
            max_iterations, time.monotonic() - t0,
        )
        return response  # type: ignore[possibly-undefined]

    def chat_json_with_tools(
        self,
        system: str,
        user: str,
        tools: list[ToolDefinition],
        temperature: float = 0.2,
        timeout: float = 900.0,
        max_iterations: int = 30,
    ) -> dict:
        """Like ``chat_with_tools`` but parses the final response as JSON."""
        _JSON_ENFORCE = (
            "\n\nWhen you provide your FINAL answer (no more tool calls), "
            "respond with ONLY valid JSON — no markdown fences, no "
            "commentary, no prose. Just the raw JSON object."
        )
        response = self.chat_with_tools(
            system + _JSON_ENFORCE, user, tools,
            temperature, timeout, max_iterations,
        )
        return self._parse_json_response(response)

    # ------------------------------------------------------------------
    # Tool-call helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_tools_for_prompt(tools: list[ToolDefinition]) -> str:
        """Render tool definitions as a human-readable description."""
        parts: list[str] = []
        for t in tools:
            params = ", ".join(
                f"{p.name}: {p.type}"
                + ("" if p.required else " (optional)")
                for p in t.parameters
            )
            parts.append(f"- **{t.name}**({params}): {t.description}")
        return "\n".join(parts)

    @classmethod
    def _extract_tool_calls_from_text(
        cls, text: str,
    ) -> list[tuple[str, dict]]:
        """Extract tool call blocks from LLM response text."""
        pattern = rf"```{cls._TOOL_CALL_TAG}\s*\n(.*?)\n```"
        calls: list[tuple[str, dict]] = []
        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                data = json.loads(match.group(1))
                name = data.get("tool", "")
                args = data.get("arguments", {})
                if name:
                    calls.append((name, args))
            except json.JSONDecodeError:
                continue
        return calls

    @staticmethod
    def _parse_json_response(content: str) -> dict:
        """Parse JSON from LLM response, handling markdown fences and
        surrounding prose."""
        if not content:
            return {}

        content = content.strip()

        # Try stripping markdown fences
        fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", content, re.DOTALL)
        if fence_match:
            content = fence_match.group(1).strip()
        elif content.startswith("```"):
            # Handle case where fence doesn't have json label
            lines = content.split("\n", 1)
            if len(lines) > 1:
                content = lines[1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Fallback: find the outermost { ... } in the text (handles
        # LLM responses that include prose before/after the JSON object).
        start = content.find("{")
        if start != -1:
            depth = 0
            in_string = False
            escape = False
            end = -1
            for i in range(start, len(content)):
                ch = content[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end != -1:
                candidate = content[start:end + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

        logger.warning("Failed to parse JSON from response")
        logger.debug("Raw content: %.500s", content)
        return {}

    @staticmethod
    def extract_json_array(text: str) -> Any:
        """Best-effort extraction of a JSON array from text.

        Handles:
        - Raw JSON (no fences)
        - JSON wrapped in markdown fences
        - Commentary/prose before/after the JSON block

        This is useful for LLM responses that should be arrays but may
        contain extra text.
        """
        if not text:
            return {}

        text = text.strip()

        # Lenient decoder: allows unescaped control characters (tabs,
        # newlines, etc.) inside JSON strings – a common LLM artefact.
        _decoder = json.JSONDecoder(strict=False)

        def _try_parse(s: str) -> Any:
            return _decoder.decode(s)

        # 1. Try stripping markdown fences first
        fence = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if fence:
            candidate = fence.group(1).strip()
            try:
                return _try_parse(candidate)
            except (json.JSONDecodeError, ValueError):
                pass

        # 2. Find the outermost [ ... ] bracket pair
        start = text.find("[")
        if start != -1:
            depth = 0
            in_string = False
            escape = False
            end = -1
            for i in range(start, len(text)):
                ch = text[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end != -1:
                candidate = text[start:end + 1]
                try:
                    return _try_parse(candidate)
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning(
                        "Bracket-matched text (%d chars) failed to parse: %s",
                        len(candidate), exc,
                    )

        # 3. Last resort — try parsing the whole text
        try:
            return _try_parse(text)
        except (json.JSONDecodeError, ValueError):
            return {}

    @staticmethod
    def extract_partial_json_array(text: str) -> list:
        """Extract complete JSON objects from a potentially truncated array.

        When the LLM response is cut off by output-token limits the
        JSON array ``[{...}, {...]`` is never closed.
        ``extract_json_array`` fails in that case.  This method recovers
        every *complete* top-level ``{...}`` object found inside the
        (possibly truncated) array.

        Returns a list of parsed dicts — may be empty.
        """
        if not text:
            return []

        text = text.strip()

        # Strip markdown fences
        fence = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()

        # Try a normal parse first (fast path)
        try:
            result = json.JSONDecoder(strict=False).decode(text)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

        # Locate the opening bracket
        arr_start = text.find("[")
        if arr_start == -1:
            return []

        # Scan for individual complete { ... } objects
        objects: list = []
        pos = arr_start + 1

        while pos < len(text):
            # Skip whitespace / commas between objects
            while pos < len(text) and text[pos] in " \t\n\r,":
                pos += 1

            if pos >= len(text) or text[pos] == "]":
                break

            if text[pos] != "{":
                break

            # Find the matching closing brace
            obj_start = pos
            depth = 0
            in_string = False
            escape = False
            obj_end = -1

            for i in range(obj_start, len(text)):
                ch = text[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        obj_end = i
                        break

            if obj_end == -1:
                # Incomplete object — response was truncated mid-object
                break

            obj_text = text[obj_start : obj_end + 1]
            try:
                obj = json.JSONDecoder(strict=False).decode(obj_text)
                objects.append(obj)
            except (json.JSONDecodeError, ValueError):
                pass

            pos = obj_end + 1

        return objects
