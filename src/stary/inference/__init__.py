"""Inference module for LLM completions.

This module provides an abstraction layer for LLM inference, allowing
the inference backend to be swapped without modifying agent code.

Usage:
    from stary.inference import get_inference_client, ToolDefinition, ToolParam

    client = get_inference_client()
    response = client.chat(system_prompt, user_prompt)

    # Tool-calling:
    tools = [ToolDefinition(name="read_file", ...)]
    response = client.chat_with_tools(system, user, tools)

The backend is selected via the INFERENCE_BACKEND environment variable:
    - "copilot" (default): GitHub Copilot SDK
    - Future backends can be added without changing agent code

Agents should ONLY use:
    - get_inference_client() to obtain a client
    - InferenceClient protocol for type hints
    - ToolDefinition / ToolParam for declaring tools
"""

from stary.inference.base import (
    BaseInferenceClient,
    InferenceClient,
    ToolDefinition,
    ToolParam,
)
from stary.inference.factory import get_inference_client

__all__ = [
    "BaseInferenceClient",
    "InferenceClient",
    "ToolDefinition",
    "ToolParam",
    "get_inference_client",
]
