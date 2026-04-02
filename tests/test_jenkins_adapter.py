"""Tests for stary.jenkins_adapter – Jenkins REST API operations."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from stary.jenkins_adapter import (
    JenkinsAdapter,
    JenkinsBuild,
    JenkinsTestCase,
    JenkinsTestReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALLOWED_HOSTS = ["ci.example.com", "jenkins-ops.habana-labs.com"]


def _make_adapter(**kwargs) -> JenkinsAdapter:
    defaults = {
        "allowed_hosts": ALLOWED_HOSTS,
        "username": "bot",
        "password": "secret",
    }
    defaults.update(kwargs)
    return JenkinsAdapter(**defaults)


def _mock_response(json_data=None, text="", status_code=200, ok=True):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.ok = ok
    resp.text = text
    resp.json.return_value = json_data or {}
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------

class TestValidateUrl:
    def test_allowed_host_passes(self):
        adapter = _make_adapter()
        result = adapter._validate_url("https://ci.example.com/job/pipe/42")
        assert result == "https://ci.example.com/job/pipe/42"

    def test_trailing_slash_stripped(self):
        adapter = _make_adapter()
        result = adapter._validate_url("https://ci.example.com/job/pipe/42/")
        assert result == "https://ci.example.com/job/pipe/42"

    def test_disallowed_host_raises(self):
        adapter = _make_adapter()
        with pytest.raises(ValueError, match="not in JENKINS_ALLOWED_HOSTS"):
            adapter._validate_url("https://evil.com/job/pipe/42")

    def test_non_http_scheme_raises(self):
        adapter = _make_adapter()
        with pytest.raises(ValueError, match="http or https"):
            adapter._validate_url("ftp://ci.example.com/job/pipe/42")

    def test_empty_host_raises(self):
        adapter = _make_adapter()
        with pytest.raises(ValueError):
            adapter._validate_url("not-a-url")

    def test_case_insensitive_host(self):
        adapter = _make_adapter()
        result = adapter._validate_url("https://CI.EXAMPLE.COM/job/pipe/42")
        assert "CI.EXAMPLE.COM" in result


# ---------------------------------------------------------------------------
# normalise_build_url
# ---------------------------------------------------------------------------

class TestNormaliseBuildUrl:
    def test_strips_console_suffix(self):
        url = "https://ci.example.com/job/pipe/42/console"
        assert JenkinsAdapter.normalise_build_url(url) == (
            "https://ci.example.com/job/pipe/42"
        )

    def test_strips_artifact_suffix(self):
        url = "https://ci.example.com/job/pipe/42/artifact/output.log"
        assert JenkinsAdapter.normalise_build_url(url) == (
            "https://ci.example.com/job/pipe/42"
        )

    def test_strips_trailing_slash(self):
        url = "https://ci.example.com/job/pipe/42/"
        assert JenkinsAdapter.normalise_build_url(url) == (
            "https://ci.example.com/job/pipe/42"
        )

    def test_strips_query_and_fragment(self):
        url = "https://ci.example.com/job/pipe/42?foo=bar#section"
        assert JenkinsAdapter.normalise_build_url(url) == (
            "https://ci.example.com/job/pipe/42"
        )

    def test_nested_job(self):
        url = "https://ci.example.com/job/folder/job/pipe/99/consoleText"
        assert JenkinsAdapter.normalise_build_url(url) == (
            "https://ci.example.com/job/folder/job/pipe/99"
        )

    def test_no_build_number(self):
        url = "https://ci.example.com/job/pipe"
        assert JenkinsAdapter.normalise_build_url(url) == (
            "https://ci.example.com/job/pipe"
        )

    def test_real_world_url(self):
        url = (
            "https://qa-jenkins-ctrl03.habana-labs.com/"
            "job/qa_jobs/job/Custom_QA_Dispatcher/2929/"
        )
        assert JenkinsAdapter.normalise_build_url(url) == (
            "https://qa-jenkins-ctrl03.habana-labs.com/"
            "job/qa_jobs/job/Custom_QA_Dispatcher/2929"
        )


# ---------------------------------------------------------------------------
# get_build_info
# ---------------------------------------------------------------------------

class TestGetBuildInfo:
    def test_returns_build_dataclass(self):
        adapter = _make_adapter()
        api_data = {
            "url": "https://ci.example.com/job/pipe/42/",
            "fullDisplayName": "pipe #42",
            "result": "SUCCESS",
            "duration": 120000,
            "timestamp": 1700000000000,
            "actions": [
                {
                    "_class": "hudson.model.ParametersAction",
                    "parameters": [
                        {"name": "BRANCH", "value": "main"},
                        {"name": "DEBUG", "value": "true"},
                    ],
                },
                {},  # empty action — should be skipped
            ],
        }
        with patch.object(adapter, "_request", return_value=_mock_response(json_data=api_data)):
            build = adapter.get_build_info("https://ci.example.com/job/pipe/42")

        assert isinstance(build, JenkinsBuild)
        assert build.result == "SUCCESS"
        assert build.full_display_name == "pipe #42"
        assert build.duration_ms == 120000
        assert build.parameters == {"BRANCH": "main", "DEBUG": "true"}

    def test_missing_result_defaults_empty(self):
        adapter = _make_adapter()
        api_data = {"url": "url", "fullDisplayName": "b", "actions": []}
        with patch.object(adapter, "_request", return_value=_mock_response(json_data=api_data)):
            build = adapter.get_build_info("https://ci.example.com/job/pipe/1")
        assert build.result == ""


# ---------------------------------------------------------------------------
# get_console_log
# ---------------------------------------------------------------------------

class TestGetConsoleLog:
    def test_returns_tail(self):
        adapter = _make_adapter()
        log = "\n".join(f"line {i}" for i in range(1000))
        with patch.object(adapter, "_request", return_value=_mock_response(text=log)):
            result = adapter.get_console_log(
                "https://ci.example.com/job/pipe/42", tail_lines=5,
            )
        assert "line 999" in result
        assert "line 995" in result
        assert "line 994" not in result
        assert "earlier lines omitted" in result

    def test_returns_full_when_short(self):
        adapter = _make_adapter()
        log = "line 1\nline 2\nline 3"
        with patch.object(adapter, "_request", return_value=_mock_response(text=log)):
            result = adapter.get_console_log(
                "https://ci.example.com/job/pipe/42", tail_lines=100,
            )
        assert result == log
        assert "omitted" not in result

    def test_tail_zero_returns_full(self):
        adapter = _make_adapter()
        log = "\n".join(f"line {i}" for i in range(10))
        with patch.object(adapter, "_request", return_value=_mock_response(text=log)):
            result = adapter.get_console_log(
                "https://ci.example.com/job/pipe/42", tail_lines=0,
            )
        assert result == log


# ---------------------------------------------------------------------------
# search_console_log
# ---------------------------------------------------------------------------

class TestSearchConsoleLog:
    def test_finds_matches_with_context(self):
        adapter = _make_adapter()
        lines = [f"line {i}" for i in range(20)]
        lines[10] = "ERROR: something broke"
        log = "\n".join(lines)
        with patch.object(adapter, "_request", return_value=_mock_response(text=log)):
            result = adapter.search_console_log(
                "https://ci.example.com/job/pipe/42",
                "ERROR",
                context_lines=2,
            )
        assert "1 match" in result
        assert "ERROR: something broke" in result
        # Context lines should be present
        assert "line 8" in result
        assert "line 12" in result

    def test_no_matches(self):
        adapter = _make_adapter()
        log = "all good\nno problems\neverything fine"
        with patch.object(adapter, "_request", return_value=_mock_response(text=log)):
            result = adapter.search_console_log(
                "https://ci.example.com/job/pipe/42", "FATAL",
            )
        assert "No matches" in result

    def test_invalid_regex_falls_back_to_literal(self):
        adapter = _make_adapter()
        log = "line with [bracket"
        with patch.object(adapter, "_request", return_value=_mock_response(text=log)):
            result = adapter.search_console_log(
                "https://ci.example.com/job/pipe/42", "[bracket",
            )
        assert "1 match" in result


# ---------------------------------------------------------------------------
# get_test_report
# ---------------------------------------------------------------------------

class TestGetTestReport:
    def test_returns_report(self):
        adapter = _make_adapter()
        report_data = {
            "totalCount": 100,
            "failCount": 3,
            "skipCount": 5,
            "suites": [
                {
                    "cases": [
                        {
                            "className": "com.example.TestA",
                            "name": "testFoo",
                            "status": "FAILED",
                            "duration": 1.5,
                            "errorDetails": "assertion failed",
                        },
                        {
                            "className": "com.example.TestA",
                            "name": "testBar",
                            "status": "PASSED",
                            "duration": 0.1,
                        },
                    ],
                },
            ],
        }
        with patch.object(adapter, "_request", return_value=_mock_response(json_data=report_data)):
            report = adapter.get_test_report("https://ci.example.com/job/pipe/42")

        assert isinstance(report, JenkinsTestReport)
        assert report.total == 100
        assert report.failed == 3
        assert report.skipped == 5
        assert report.passed == 92
        # Only failed cases included
        assert len(report.cases) == 1
        assert report.cases[0].name == "testFoo"
        assert report.cases[0].error_message == "assertion failed"

    def test_returns_none_on_404(self):
        adapter = _make_adapter()
        http_error = requests.HTTPError()
        http_error.response = _mock_response(status_code=404, ok=False)
        http_error.response.status_code = 404

        def raise_404(url, params=None):
            raise http_error

        with patch.object(adapter, "_request", side_effect=raise_404):
            result = adapter.get_test_report("https://ci.example.com/job/pipe/42")
        assert result is None

    def test_reraises_non_404(self):
        adapter = _make_adapter()
        http_error = requests.HTTPError()
        http_error.response = _mock_response(status_code=500, ok=False)
        http_error.response.status_code = 500

        def raise_500(url, params=None):
            raise http_error

        with patch.object(adapter, "_request", side_effect=raise_500):
            with pytest.raises(requests.HTTPError):
                adapter.get_test_report("https://ci.example.com/job/pipe/42")
