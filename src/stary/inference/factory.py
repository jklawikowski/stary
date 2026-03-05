"""Factory for creating inference clients.

This module provides a factory function that returns the appropriate
inference client based on configuration.

To add a new backend:
1. Create the implementation in stary/inference/<backend>.py
2. Add it to BACKENDS dict below
3. Users can switch via INFERENCE_BACKEND env var

The factory handles singleton management so clients are reused.
"""

import os
from typing import Optional

from stary.inference.base import InferenceClient

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

# Lazy imports to avoid loading unused backends
BACKENDS = {
    "copilot": "stary.inference.copilot:CopilotInferenceClient",
    # Future backends:
    # "openai": "stary.inference.openai:OpenAIInferenceClient",
    # "anthropic": "stary.inference.anthropic:AnthropicInferenceClient",
    # "local": "stary.inference.local:LocalInferenceClient",
}

DEFAULT_BACKEND = "copilot"


# ---------------------------------------------------------------------------
# Singleton storage
# ---------------------------------------------------------------------------
_client_instance: Optional[InferenceClient] = None
_current_backend: Optional[str] = None


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def get_inference_client(
    backend: str | None = None,
    force_new: bool = False,
    **kwargs,
) -> InferenceClient:
    """Get or create an inference client.

    This is the ONLY function agents should use to obtain an inference
    client. It handles:
    - Backend selection via environment variable or parameter
    - Singleton management (clients are reused by default)
    - Lazy loading of backend implementations

    Args:
        backend: Backend name (default: INFERENCE_BACKEND env var or "copilot")
        force_new: If True, create a new client instance instead of reusing
        **kwargs: Additional arguments passed to the backend constructor

    Returns:
        An InferenceClient instance

    Raises:
        ValueError: If the requested backend is not registered
        RuntimeError: If the backend fails to initialize

    Example:
        from stary.inference import get_inference_client

        client = get_inference_client()
        response = client.chat("You are helpful.", "Hello!")
    """
    global _client_instance, _current_backend

    # Determine which backend to use
    requested_backend = backend or os.environ.get("INFERENCE_BACKEND", DEFAULT_BACKEND)
    requested_backend = requested_backend.lower().strip()

    # Return cached instance if available and matching
    if (
        not force_new
        and _client_instance is not None
        and _current_backend == requested_backend
    ):
        return _client_instance

    # Validate backend
    if requested_backend not in BACKENDS:
        available = ", ".join(sorted(BACKENDS.keys()))
        raise ValueError(
            f"Unknown inference backend: {requested_backend!r}. "
            f"Available backends: {available}"
        )

    # Load and instantiate the backend
    module_path = BACKENDS[requested_backend]
    client_class = _import_class(module_path)

    print(f"[Inference] Initializing {requested_backend} backend...")
    try:
        instance = client_class(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize {requested_backend} inference backend: {exc}"
        ) from exc

    # Cache the instance
    _client_instance = instance
    _current_backend = requested_backend

    return instance


def _import_class(path: str):
    """Dynamically import a class from a module path.

    Args:
        path: Module path in format "module.path:ClassName"

    Returns:
        The class object
    """
    module_path, class_name = path.rsplit(":", 1)

    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def reset_client() -> None:
    """Reset the cached client instance.

    Useful for testing or when switching backends at runtime.
    """
    global _client_instance, _current_backend
    _client_instance = None
    _current_backend = None


def get_available_backends() -> list[str]:
    """Return list of available backend names."""
    return list(BACKENDS.keys())
