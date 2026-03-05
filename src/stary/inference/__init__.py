"""Inference module for LLM completions.

This module provides an abstraction layer for LLM inference, allowing
the inference backend to be swapped without modifying agent code.

Usage:
    from stary.inference import get_inference_client

    client = get_inference_client()
    response = client.chat(system_prompt, user_prompt)
    json_response = client.chat_json(system_prompt, user_prompt)

The backend is selected via the INFERENCE_BACKEND environment variable:
    - "copilot" (default): GitHub Copilot SDK
    - Future backends can be added without changing agent code

Agents should ONLY use:
    - get_inference_client() to obtain a client
    - InferenceClient protocol for type hints

This ensures agents are decoupled from the specific inference implementation.
"""

from stary.inference.base import BaseInferenceClient, InferenceClient
from stary.inference.factory import get_inference_client

__all__ = [
    "BaseInferenceClient",
    "InferenceClient",
    "get_inference_client",
]
