"""OpenTelemetry tracing for Stary.

Provides distributed tracing via OTLP export.  The OTel Collector's
``spanmetrics`` connector automatically derives request counters and
duration histograms from spans, so no manual metric definitions are
needed.

All setup is behind ``init_telemetry()`` which reads standard OTEL_*
env vars.  When ``OTEL_EXPORTER_OTLP_ENDPOINT`` is unset, everything
is a silent no-op (the tracer returns no-op spans).
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import socket

from opentelemetry import trace

logger = logging.getLogger(__name__)

tracer = trace.get_tracer("stary")

_initialized = False
_tracer_provider = None

# Patterns used to normalise high-cardinality route segments.
_ISSUE_KEY_RE = re.compile(r"/rest/api/2/issue/[A-Z][A-Z0-9_]+-\d+")
_COMMENT_SUFFIX_RE = re.compile(r"(/comment)/\d+")


def _normalise_route(endpoint: str) -> str:
    """Replace dynamic path segments with placeholders.

    >>> _normalise_route("/rest/api/2/issue/PROJ-123/comment")
    '/rest/api/2/issue/{issueKey}/comment'
    >>> _normalise_route("/rest/api/2/issue/PROJ-123")
    '/rest/api/2/issue/{issueKey}'
    >>> _normalise_route("/rest/api/2/search")
    '/rest/api/2/search'
    """
    result = _ISSUE_KEY_RE.sub("/rest/api/2/issue/{issueKey}", endpoint)
    result = _COMMENT_SUFFIX_RE.sub(r"\1/{commentId}", result)
    return result


def init_telemetry() -> None:
    """Initialize OTel TracerProvider with OTLP gRPC exporter.

    Safe to call multiple times — only the first call takes effect.
    """
    global _initialized, _tracer_provider

    if _initialized:
        return

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if not endpoint:
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT not set — telemetry disabled")
        _initialized = True
        return

    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource

        service_name = os.environ.get("OTEL_SERVICE_NAME", "stary")
        resource = Resource.create({
            "service.name": service_name,
            "service.instance.id": socket.gethostname(),
        })

        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        _tracer_provider = provider

        atexit.register(_flush_traces)

        logger.info("Telemetry initialized (endpoint=%s, service=%s)", endpoint, service_name)
    except Exception:
        logger.warning("Failed to initialize telemetry — continuing without it", exc_info=True)

    _initialized = True


def _flush_traces() -> None:
    """Flush pending spans on process exit."""
    if _tracer_provider is not None:
        try:
            _tracer_provider.force_flush(timeout_millis=5_000)
        except Exception:
            logger.debug("Failed to flush traces on exit", exc_info=True)
