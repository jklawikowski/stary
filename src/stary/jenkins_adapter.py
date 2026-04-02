"""Low-level Jenkins REST API adapter.

Provides a clean, reusable interface for Jenkins operations with:
- Centralized authentication (HTTP Basic)
- Retry logic for transient failures
- Host allowlist for SSRF prevention
- Consistent error handling

This module has NO business logic — it's a pure data-access layer.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from stary.telemetry import _normalise_jenkins_route, tracer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
JENKINS_ALLOWED_HOSTS = [
    h.strip()
    for h in os.environ.get("JENKINS_ALLOWED_HOSTS", "").split(",")
    if h.strip()
]
JENKINS_USERNAME = os.environ.get("JENKINS_USERNAME", "")
JENKINS_PASSWORD = os.environ.get("JENKINS_PASSWORD", "")

DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 0.5

# Max characters of console log to fetch (protection against huge logs).
_MAX_LOG_CHARS = 5_000_000  # ~5 MB


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class JenkinsBuild:
    """Core metadata for a Jenkins build."""

    url: str
    full_display_name: str = ""
    result: str = ""
    duration_ms: int = 0
    timestamp: int = 0
    parameters: dict = field(default_factory=dict)


@dataclass
class JenkinsTestCase:
    """A single test case from a Jenkins test report."""

    class_name: str = ""
    name: str = ""
    status: str = ""
    duration: float = 0.0
    error_message: str = ""


@dataclass
class JenkinsTestReport:
    """Aggregated test report from a Jenkins build."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    cases: list[JenkinsTestCase] = field(default_factory=list)


class JenkinsAdapter:
    """Low-level Jenkins REST API operations.

    Supports multiple Jenkins hosts with a shared credential pair.
    All URLs are validated against an allowlist before any request
    is made (SSRF prevention).
    """

    def __init__(
        self,
        allowed_hosts: list[str] | None = None,
        username: str | None = None,
        password: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    ):
        self.allowed_hosts = [
            h.lower() for h in (allowed_hosts or JENKINS_ALLOWED_HOSTS)
        ]
        self.username = username or JENKINS_USERNAME
        self.password = password or JENKINS_PASSWORD
        self.timeout = timeout
        self._session = self._create_session(max_retries, backoff_factor)

    # ------------------------------------------------------------------
    # Session setup
    # ------------------------------------------------------------------

    def _create_session(
        self,
        max_retries: int,
        backoff_factor: float,
    ) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    # ------------------------------------------------------------------
    # URL validation
    # ------------------------------------------------------------------

    def _validate_url(self, url: str) -> str:
        """Validate *url* against the host allowlist.

        Returns the normalised URL.  Raises ``ValueError`` when the
        host is not in the allowlist.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"URL scheme must be http or https, got: {url!r}")
        host = parsed.hostname
        if not host:
            raise ValueError(f"Cannot parse host from URL: {url!r}")
        if host.lower() not in self.allowed_hosts:
            raise ValueError(
                f"Host '{host}' is not in JENKINS_ALLOWED_HOSTS. "
                f"Allowed: {', '.join(self.allowed_hosts)}"
            )
        return url.rstrip("/")

    # ------------------------------------------------------------------
    # Core HTTP
    # ------------------------------------------------------------------

    @tracer.start_as_current_span("jenkins.request")
    def _request(
        self,
        url: str,
        params: dict | None = None,
    ) -> requests.Response:
        """Execute a GET request to a validated Jenkins URL."""
        url = self._validate_url(url)

        span = trace.get_current_span()
        span.set_attribute("http.method", "GET")
        span.set_attribute("http.route", _normalise_jenkins_route(url))
        span.set_attribute("http.url", url)

        try:
            resp = self._session.get(
                url,
                params=params,
                auth=(self.username, self.password) if self.username else None,
                timeout=self.timeout,
            )
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            raise

        span.set_attribute("http.status_code", resp.status_code)
        if resp.status_code >= 500:
            span.set_status(StatusCode.ERROR, f"HTTP {resp.status_code}")
        if not resp.ok:
            logger.warning(
                "Jenkins request failed: GET %s — HTTP %d: %.500s",
                url, resp.status_code, resp.text,
            )
        resp.raise_for_status()
        return resp

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def normalise_build_url(url: str) -> str:
        """Normalise a Jenkins URL to point at a build root.

        Strips query strings, fragment identifiers, and trailing slashes.
        If the URL contains path segments after the build number
        (e.g. ``/console``, ``/artifact/...``), they are removed.

        Examples::

            >>> JenkinsAdapter.normalise_build_url(
            ...     "https://ci.example.com/job/pipe/42/console"
            ... )
            'https://ci.example.com/job/pipe/42'
            >>> JenkinsAdapter.normalise_build_url(
            ...     "https://ci.example.com/job/pipe/42/"
            ... )
            'https://ci.example.com/job/pipe/42'
        """
        parsed = urlparse(url)
        # Rebuild without query / fragment
        clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
        # Walk path segments backwards to find the build number
        parts = clean.split("/")
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].isdigit():
                return "/".join(parts[: i + 1])
        # No numeric segment found — return cleaned URL as-is
        return clean

    # ------------------------------------------------------------------
    # Build info
    # ------------------------------------------------------------------

    def get_build_info(self, build_url: str) -> JenkinsBuild:
        """Fetch build metadata via the Jenkins JSON API.

        Args:
            build_url: URL of a Jenkins build
                (e.g. ``https://ci.example.com/job/pipe/42``).
        """
        build_url = self.normalise_build_url(build_url)
        api_url = f"{build_url}/api/json"
        resp = self._request(api_url)
        data = resp.json()

        # Extract build parameters from actions
        parameters: dict = {}
        for action in data.get("actions", []):
            if action and action.get("_class", "").endswith("ParametersAction"):
                for p in action.get("parameters", []):
                    name = p.get("name", "")
                    if name:
                        parameters[name] = p.get("value", "")

        return JenkinsBuild(
            url=data.get("url", build_url),
            full_display_name=data.get("fullDisplayName", ""),
            result=data.get("result", "") or "",
            duration_ms=data.get("duration", 0),
            timestamp=data.get("timestamp", 0),
            parameters=parameters,
        )

    # ------------------------------------------------------------------
    # Console log
    # ------------------------------------------------------------------

    def get_console_log(
        self,
        build_url: str,
        tail_lines: int = 500,
    ) -> str:
        """Fetch console output for a build, returning the last *tail_lines*.

        Args:
            build_url: URL of a Jenkins build.
            tail_lines: Number of lines to return from the end of the log.
                Use 0 to return the full log (capped at ~5 MB).
        """
        build_url = self.normalise_build_url(build_url)
        log_url = f"{build_url}/consoleText"
        resp = self._request(log_url)
        text = resp.text[:_MAX_LOG_CHARS]

        if tail_lines <= 0:
            return text

        lines = text.splitlines()
        if len(lines) <= tail_lines:
            return text

        kept = lines[-tail_lines:]
        return (
            f"[... {len(lines) - tail_lines} earlier lines omitted ...]\n"
            + "\n".join(kept)
        )

    def search_console_log(
        self,
        build_url: str,
        pattern: str,
        context_lines: int = 5,
    ) -> str:
        """Search console output for *pattern* and return matching lines.

        Performs a case-insensitive search.  Returns up to 50 match
        groups, each with *context_lines* of surrounding context.

        Args:
            build_url: URL of a Jenkins build.
            pattern: Regex or plain-text pattern to search for.
            context_lines: Lines of context above and below each match.
        """
        build_url = self.normalise_build_url(build_url)
        log_url = f"{build_url}/consoleText"
        resp = self._request(log_url)
        text = resp.text[:_MAX_LOG_CHARS]
        lines = text.splitlines()

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # Treat as a plain substring search
            regex = re.compile(re.escape(pattern), re.IGNORECASE)

        match_indices: list[int] = []
        for i, line in enumerate(lines):
            if regex.search(line):
                match_indices.append(i)

        if not match_indices:
            return f"No matches for '{pattern}' in {len(lines)} lines of console output."

        # Merge overlapping context windows
        segments: list[str] = []
        used: set[int] = set()
        for idx in match_indices[:50]:
            start = max(0, idx - context_lines)
            end = min(len(lines), idx + context_lines + 1)
            block_lines: list[str] = []
            for j in range(start, end):
                if j not in used:
                    prefix = ">>> " if j == idx else "    "
                    block_lines.append(f"{j + 1:>6}: {prefix}{lines[j]}")
                    used.add(j)
            if block_lines:
                segments.append("\n".join(block_lines))

        header = (
            f"Found {len(match_indices)} match(es) for '{pattern}' "
            f"in {len(lines)} lines:"
        )
        return header + "\n\n" + "\n---\n".join(segments)

    # ------------------------------------------------------------------
    # Test report
    # ------------------------------------------------------------------

    def get_test_report(self, build_url: str) -> JenkinsTestReport | None:
        """Fetch the JUnit test report for a build.

        Returns ``None`` if the build has no test report (HTTP 404).
        """
        build_url = self.normalise_build_url(build_url)
        report_url = f"{build_url}/testReport/api/json"
        try:
            resp = self._request(report_url)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise

        data = resp.json()
        total = data.get("totalCount", 0)
        failed = data.get("failCount", 0)
        skipped = data.get("skipCount", 0)
        passed = total - failed - skipped

        cases: list[JenkinsTestCase] = []
        for suite in data.get("suites", []):
            for case in suite.get("cases", []):
                status = case.get("status", "")
                # Only include failed/errored cases in detail to save tokens
                if status in ("FAILED", "REGRESSION", "ERROR"):
                    cases.append(
                        JenkinsTestCase(
                            class_name=case.get("className", ""),
                            name=case.get("name", ""),
                            status=status,
                            duration=case.get("duration", 0.0),
                            error_message=(
                                case.get("errorDetails", "")
                                or case.get("errorStackTrace", "")
                                or ""
                            )[:2000],
                        )
                    )

        return JenkinsTestReport(
            total=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            cases=cases,
        )
