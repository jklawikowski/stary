"""Base interface for inference clients.

This module defines the InferenceClient protocol that all inference
backends must implement. Agents depend ONLY on this interface.

To add a new inference backend:
1. Create a new module in stary/inference/ (e.g., openai.py)
2. Implement a class that satisfies the InferenceClient protocol
3. Register it in the factory (stary/inference/factory.py)

The interface is intentionally minimal to make implementation easy
while covering all agent needs.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable


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
        """Send a chat completion request.

        Args:
            system: System prompt defining the assistant's behavior
            user: User message/prompt
            temperature: Sampling temperature (0.0-1.0), lower = more deterministic
            timeout: Request timeout in seconds

        Returns:
            The assistant's response as a string
        """
        ...

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        timeout: float = 300.0,
    ) -> dict:
        """Send a chat completion request and parse JSON response.

        A JSON-enforcement suffix is appended to the system prompt.

        Args:
            system: System prompt (should instruct LLM to return JSON)
            user: User message/prompt
            temperature: Sampling temperature (0.0-1.0)
            timeout: Request timeout in seconds

        Returns:
            Parsed JSON dict from the response, or empty dict on parse failure
        """
        ...


class BaseInferenceClient(ABC):
    """Abstract base class with shared utilities for inference clients.

    Concrete implementations can inherit from this to get common
    JSON parsing logic. They only need to implement `chat()`.
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
        """Send a chat completion request and parse JSON response.

        Appends a JSON-enforcement suffix to the system prompt so the
        model is more likely to return raw JSON.
        """
        _JSON_ENFORCE = (
            "\n\nYou MUST respond with ONLY a valid JSON object — no markdown "
            "fences, no commentary, no prose. Just the raw JSON."
        )
        content = self.chat(system + _JSON_ENFORCE, user, temperature, timeout)
        return self._parse_json_response(content)

    @staticmethod
    def _parse_json_response(content: str) -> dict:
        """Parse JSON from LLM response, handling markdown fences."""
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
        except json.JSONDecodeError as exc:
            print(f"[InferenceClient] Failed to parse JSON: {exc}")
            print(f"[InferenceClient] Raw content: {content[:500]}...")
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
                    print(
                        f"[extract_json_array] Bracket-matched text "
                        f"({len(candidate)} chars) failed to parse: {exc}"
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
