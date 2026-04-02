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

# Jenkins: /job/<name>/<number> → /job/{jobName}/{buildNumber}
_JENKINS_BUILD_NUM_RE = re.compile(r"(/job/[^/]+)/(\d+)")
_JENKINS_SUFFIX_RE = re.compile(
    r"/(api/json|consoleText|testReport/api/json|testReport)$"
)

# GitHub: replace owner/repo names and PR/issue numbers with placeholders
_GH_REPOS_RE = re.compile(r"/repos/[^/]+/[^/]+")
_GH_PULLS_NUM_RE = re.compile(r"(/pulls?)/\d+")
_GH_ISSUES_NUM_RE = re.compile(r"(/issues)/\d+")
_GH_COMMENTS_NUM_RE = re.compile(r"(/comments)/\d+")
_GH_TREES_SHA_RE = re.compile(r"(/git/trees)/[^/]+")


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


def _normalise_jenkins_route(url: str) -> str:
    """Normalise a Jenkins URL into a low-cardinality route for span metrics.

    Strips the scheme + host, replaces build numbers with ``{buildNumber}``,
    and strips API suffixes so all requests to the same job/build collapse.

    >>> _normalise_jenkins_route(
    ...     "https://ci.example.com/job/pipe/42/api/json"
    ... )
    '/job/pipe/{buildNumber}'
    >>> _normalise_jenkins_route(
    ...     "https://ci.example.com/job/folder/job/pipe/99/consoleText"
    ... )
    '/job/folder/job/pipe/{buildNumber}'
    >>> _normalise_jenkins_route(
    ...     "https://ci.example.com/job/pipe/42/testReport/api/json"
    ... )
    '/job/pipe/{buildNumber}'
    """
    from urllib.parse import urlparse
    path = urlparse(url).path.rstrip("/")
    # Strip API suffixes first
    path = _JENKINS_SUFFIX_RE.sub("", path)
    # Replace build numbers
    path = _JENKINS_BUILD_NUM_RE.sub(r"\1/{buildNumber}", path)
    return path


def _normalise_github_route(endpoint: str) -> str:
    """Normalise a GitHub API endpoint into a low-cardinality route.

    >>> _normalise_github_route("/repos/owner/repo/contents/src/main.py")
    '/repos/{owner}/{repo}/contents/src/main.py'
    >>> _normalise_github_route("/repos/org/project/pulls/42")
    '/repos/{owner}/{repo}/pulls/{number}'
    >>> _normalise_github_route("/repos/org/project/git/trees/abc123")
    '/repos/{owner}/{repo}/git/trees/{ref}'
    >>> _normalise_github_route("/user")
    '/user'
    """
    result = _GH_REPOS_RE.sub("/repos/{owner}/{repo}", endpoint)
    result = _GH_PULLS_NUM_RE.sub(r"\1/{number}", result)
    result = _GH_ISSUES_NUM_RE.sub(r"\1/{number}", result)
    result = _GH_COMMENTS_NUM_RE.sub(r"\1/{commentId}", result)
    result = _GH_TREES_SHA_RE.sub(r"\1/{ref}", result)
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

        insecure = not endpoint.startswith("https://")
        exporter = OTLPSpanExporter(insecure=insecure)

        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))
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
