"""Tests for make_jenkins_tools – tool factory for Jenkins integration."""

from unittest.mock import MagicMock

import pytest

from stary.agents.tools import make_jenkins_tools
from stary.jenkins_adapter import JenkinsBuild, JenkinsTestCase, JenkinsTestReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_adapter():
    return MagicMock()


def _tool_by_name(tools, name):
    for t in tools:
        if t.name == name:
            return t
    raise KeyError(f"No tool named '{name}'")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestMakeJenkinsTools:
    def test_returns_five_tools(self):
        tools = make_jenkins_tools(_make_mock_adapter())
        assert len(tools) == 5

    def test_tool_names(self):
        tools = make_jenkins_tools(_make_mock_adapter())
        names = {t.name for t in tools}
        assert names == {
            "fetch_jenkins_build",
            "fetch_jenkins_log",
            "search_jenkins_log",
            "search_jenkins_log_xpu_errors",
            "fetch_jenkins_test_report",
        }

    def test_tools_have_openai_schema(self):
        tools = make_jenkins_tools(_make_mock_adapter())
        for tool in tools:
            schema = tool.to_openai_schema()
            assert schema["type"] == "function"
            assert "name" in schema["function"]


# ---------------------------------------------------------------------------
# fetch_jenkins_build
# ---------------------------------------------------------------------------

class TestFetchJenkinsBuild:
    def test_formats_build_info(self):
        adapter = _make_mock_adapter()
        adapter.get_build_info.return_value = JenkinsBuild(
            url="https://ci.example.com/job/pipe/42/",
            full_display_name="pipe #42",
            result="FAILURE",
            duration_ms=65000,
            timestamp=1700000000000,
            parameters={"BRANCH": "main"},
        )
        tool = _tool_by_name(make_jenkins_tools(adapter), "fetch_jenkins_build")
        result = tool.handler(url="https://ci.example.com/job/pipe/42")

        assert "pipe #42" in result
        assert "FAILURE" in result
        assert "65s" in result
        assert "BRANCH = main" in result

    def test_returns_error_on_exception(self):
        adapter = _make_mock_adapter()
        adapter.get_build_info.side_effect = RuntimeError("connection refused")
        tool = _tool_by_name(make_jenkins_tools(adapter), "fetch_jenkins_build")
        result = tool.handler(url="https://ci.example.com/job/pipe/42")

        assert "Error" in result
        assert "connection refused" in result


# ---------------------------------------------------------------------------
# fetch_jenkins_log
# ---------------------------------------------------------------------------

class TestFetchJenkinsLog:
    def test_returns_log(self):
        adapter = _make_mock_adapter()
        adapter.get_console_log.return_value = "line 1\nline 2\nline 3"
        tool = _tool_by_name(make_jenkins_tools(adapter), "fetch_jenkins_log")
        result = tool.handler(url="https://ci.example.com/job/pipe/42")

        assert "line 1" in result
        adapter.get_console_log.assert_called_once_with(
            "https://ci.example.com/job/pipe/42", tail_lines=500,
        )

    def test_passes_custom_tail_lines(self):
        adapter = _make_mock_adapter()
        adapter.get_console_log.return_value = "data"
        tool = _tool_by_name(make_jenkins_tools(adapter), "fetch_jenkins_log")
        tool.handler(url="https://ci.example.com/job/pipe/42", tail_lines=100)

        adapter.get_console_log.assert_called_once_with(
            "https://ci.example.com/job/pipe/42", tail_lines=100,
        )

    def test_returns_error_on_exception(self):
        adapter = _make_mock_adapter()
        adapter.get_console_log.side_effect = ValueError("not allowed")
        tool = _tool_by_name(make_jenkins_tools(adapter), "fetch_jenkins_log")
        result = tool.handler(url="https://evil.com/job/pipe/42")

        assert "Error" in result


# ---------------------------------------------------------------------------
# search_jenkins_log
# ---------------------------------------------------------------------------

class TestSearchJenkinsLog:
    def test_returns_search_results(self):
        adapter = _make_mock_adapter()
        adapter.search_console_log.return_value = "Found 2 match(es)"
        tool = _tool_by_name(make_jenkins_tools(adapter), "search_jenkins_log")
        result = tool.handler(
            url="https://ci.example.com/job/pipe/42", pattern="error",
        )

        assert "2 match" in result
        adapter.search_console_log.assert_called_once_with(
            "https://ci.example.com/job/pipe/42", "error",
        )

    def test_returns_error_on_exception(self):
        adapter = _make_mock_adapter()
        adapter.search_console_log.side_effect = ConnectionError("timeout")
        tool = _tool_by_name(make_jenkins_tools(adapter), "search_jenkins_log")
        result = tool.handler(url="url", pattern="x")

        assert "Error" in result


# ---------------------------------------------------------------------------
# search_jenkins_log_xpu_errors
# ---------------------------------------------------------------------------


class TestSearchJenkinsLogXpuErrors:
    def test_finds_matching_patterns(self):
        adapter = _make_mock_adapter()

        # Return matches for OOM, empty for others
        def side_effect(url, pattern):
            if "OutOfMemory" in pattern:
                return (
                    "Found 1 match(es)\n"
                    "line 42: torch.OutOfMemoryError: XPU out of memory"
                )
            return "Found 0 match(es)"

        adapter.search_console_log.side_effect = side_effect
        tool = _tool_by_name(
            make_jenkins_tools(adapter), "search_jenkins_log_xpu_errors",
        )
        result = tool.handler(url="https://ci.example.com/job/pipe/42")

        assert "XPU-related errors" in result
        assert "OutOfMemory" in result

    def test_returns_no_patterns_message(self):
        adapter = _make_mock_adapter()
        adapter.search_console_log.return_value = "Found 0 match(es)"
        tool = _tool_by_name(
            make_jenkins_tools(adapter), "search_jenkins_log_xpu_errors",
        )
        result = tool.handler(url="https://ci.example.com/job/pipe/42")

        assert "No common XPU error patterns found" in result

    def test_handles_exceptions_gracefully(self):
        adapter = _make_mock_adapter()
        adapter.search_console_log.side_effect = ConnectionError("timeout")
        tool = _tool_by_name(
            make_jenkins_tools(adapter), "search_jenkins_log_xpu_errors",
        )
        result = tool.handler(url="https://ci.example.com/job/pipe/42")

        assert "Error" in result or "XPU-related errors" in result


# ---------------------------------------------------------------------------
# fetch_jenkins_test_report
# ---------------------------------------------------------------------------

class TestFetchJenkinsTestReport:
    def test_formats_report(self):
        adapter = _make_mock_adapter()
        adapter.get_test_report.return_value = JenkinsTestReport(
            total=100, passed=95, failed=3, skipped=2,
            cases=[
                JenkinsTestCase(
                    class_name="com.example.TestA",
                    name="testFoo",
                    status="FAILED",
                    duration=1.5,
                    error_message="expected 1 but got 2",
                ),
            ],
        )
        tool = _tool_by_name(make_jenkins_tools(adapter), "fetch_jenkins_test_report")
        result = tool.handler(url="https://ci.example.com/job/pipe/42")

        assert "100" in result
        assert "Failed: 3" in result
        assert "testFoo" in result
        assert "expected 1 but got 2" in result

    def test_returns_no_report_message(self):
        adapter = _make_mock_adapter()
        adapter.get_test_report.return_value = None
        tool = _tool_by_name(make_jenkins_tools(adapter), "fetch_jenkins_test_report")
        result = tool.handler(url="https://ci.example.com/job/pipe/42")

        assert "No test report" in result

    def test_returns_error_on_exception(self):
        adapter = _make_mock_adapter()
        adapter.get_test_report.side_effect = RuntimeError("boom")
        tool = _tool_by_name(make_jenkins_tools(adapter), "fetch_jenkins_test_report")
        result = tool.handler(url="url")

        assert "Error" in result
