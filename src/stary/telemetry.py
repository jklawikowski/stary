"""OpenTelemetry metrics for Stary.

Provides Jira API call counting and duration tracking via OTLP export.
All setup is behind ``init_telemetry()`` which reads standard OTEL_*
env vars.  When ``OTEL_EXPORTER_OTLP_ENDPOINT`` is unset, everything
is a silent no-op.
"""

from __future__ import annotations

import atexit
import logging
import os
import uuid

logger = logging.getLogger(__name__)

_initialized = False
_jira_request_counter = None
_jira_request_duration = None
_meter_provider = None


def init_telemetry() -> None:
    """Initialize OTel MeterProvider with OTLP gRPC exporter.

    Safe to call multiple times — only the first call takes effect.
    """
    global _initialized, _jira_request_counter, _jira_request_duration, _meter_provider

    if _initialized:
        return

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if not endpoint:
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT not set — telemetry disabled")
        _initialized = True
        return

    try:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry import metrics

        service_name = os.environ.get("OTEL_SERVICE_NAME", "stary")
        resource = Resource.create({
            "service.name": service_name,
            "service.instance.id": f"{os.getpid()}-{uuid.uuid4().hex[:8]}",
        })

        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(),
            export_interval_millis=5_000,
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(meter_provider)
        _meter_provider = meter_provider

        atexit.register(_flush_metrics)

        meter = metrics.get_meter("stary")

        _jira_request_counter = meter.create_counter(
            name="stary.jira.requests",
            description="Total Jira API requests",
            unit="1",
        )
        _jira_request_duration = meter.create_histogram(
            name="stary.jira.request.duration",
            description="Jira API request duration",
            unit="s",
        )

        logger.info("Telemetry initialized (endpoint=%s, service=%s)", endpoint, service_name)
    except Exception:
        logger.warning("Failed to initialize telemetry — continuing without it", exc_info=True)

    _initialized = True


def _flush_metrics() -> None:
    """Flush pending metrics on process exit."""
    if _meter_provider is not None:
        try:
            _meter_provider.force_flush(timeout_millis=5_000)
        except Exception:
            logger.debug("Failed to flush metrics on exit", exc_info=True)


def record_jira_request(method: str, endpoint: str, status_code: int, duration: float) -> None:
    """Record a single Jira HTTP request (counter + histogram)."""
    if _jira_request_counter is None:
        return
    attrs = {
        "http.method": method,
        "http.route": endpoint,
        "http.status_code": str(status_code),
    }
    _jira_request_counter.add(1, attrs)
    _jira_request_duration.record(duration, attrs)
