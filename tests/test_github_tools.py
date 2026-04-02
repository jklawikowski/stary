"""Tests for GitHub read tools – URL parser and tool factory for TaskReader."""

from unittest.mock import MagicMock

import pytest

from stary.agents.tools import (
    _parse_github_file_url,
    make_github_read_tools,
)


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
# _parse_github_file_url
# ---------------------------------------------------------------------------

class TestParseGitHubFileUrl:
    def test_basic_blob_url(self):
        url = "https://github.com/org/repo/blob/main/src/module.py"
        result = _parse_github_file_url(url)
        assert result["owner"] == "org"
        assert result["repo"] == "repo"
        assert result["ref"] == "main"
        assert result["path"] == "src/module.py"
        assert result["kind"] == "blob"
        assert result["start_line"] is None
        assert result["end_line"] is None

    def test_blob_url_with_single_line(self):
        url = "https://github.com/habana-internal/event_tests_plugin/blob/master_next/event_tests_core/interface_base_test.py#L460"
        result = _parse_github_file_url(url)
        assert result["owner"] == "habana-internal"
        assert result["repo"] == "event_tests_plugin"
        assert result["ref"] == "master_next"
        assert result["path"] == "event_tests_core/interface_base_test.py"
        assert result["start_line"] == 460
        assert result["end_line"] == 460

    def test_blob_url_with_line_range(self):
        url = "https://github.com/org/repo/blob/dev/src/app.py#L10-L25"
        result = _parse_github_file_url(url)
        assert result["start_line"] == 10
        assert result["end_line"] == 25
        assert result["path"] == "src/app.py"
        assert result["ref"] == "dev"

    def test_blob_url_with_line_range_no_L_prefix_on_end(self):
        url = "https://github.com/org/repo/blob/main/file.py#L10-25"
        result = _parse_github_file_url(url)
        assert result["start_line"] == 10
        assert result["end_line"] == 25

    def test_tree_url(self):
        url = "https://github.com/org/repo/tree/main/src/tests"
        result = _parse_github_file_url(url)
        assert result["kind"] == "tree"
        assert result["path"] == "src/tests"
        assert result["ref"] == "main"

    def test_tree_url_root(self):
        url = "https://github.com/org/repo/tree/main"
        result = _parse_github_file_url(url)
        assert result["ref"] == "main"
        assert result["path"] == ""

    def test_strips_whitespace(self):
        url = "  https://github.com/org/repo/blob/main/file.py  "
        result = _parse_github_file_url(url)
        assert result["owner"] == "org"
        assert result["path"] == "file.py"

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Not a valid GitHub file URL"):
            _parse_github_file_url("https://github.com/org/repo")

    def test_not_github_raises(self):
        with pytest.raises(ValueError, match="Not a valid GitHub file URL"):
            _parse_github_file_url("https://gitlab.com/org/repo/blob/main/f.py")

    def test_deep_nested_path(self):
        url = "https://github.com/org/repo/blob/master_next/tests/gdn_tests/CB_tests/test_serving_benchmark_golden.py"
        result = _parse_github_file_url(url)
        assert result["path"] == "tests/gdn_tests/CB_tests/test_serving_benchmark_golden.py"
        assert result["ref"] == "master_next"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestMakeGitHubReadTools:
    def test_returns_two_tools(self):
        tools = make_github_read_tools(_make_mock_adapter())
        assert len(tools) == 2

    def test_tool_names(self):
        tools = make_github_read_tools(_make_mock_adapter())
        names = {t.name for t in tools}
        assert names == {"fetch_github_file", "list_github_directory"}

    def test_tools_have_openai_schema(self):
        tools = make_github_read_tools(_make_mock_adapter())
        for tool in tools:
            schema = tool.to_openai_schema()
            assert schema["type"] == "function"
            assert "name" in schema["function"]


# ---------------------------------------------------------------------------
# fetch_github_file handler
# ---------------------------------------------------------------------------

class TestFetchGitHubFile:
    def test_fetches_full_file(self):
        adapter = _make_mock_adapter()
        adapter.get_file_contents.return_value = "line1\nline2\nline3\n"
        tool = _tool_by_name(make_github_read_tools(adapter), "fetch_github_file")

        result = tool.handler(url="https://github.com/org/repo/blob/main/src/app.py")
        adapter.get_file_contents.assert_called_once_with("org", "repo", "src/app.py", ref="main")
        assert "line1" in result
        assert "line2" in result

    def test_applies_line_range(self):
        adapter = _make_mock_adapter()
        content = "\n".join(f"line {i}" for i in range(1, 101))
        adapter.get_file_contents.return_value = content
        tool = _tool_by_name(make_github_read_tools(adapter), "fetch_github_file")

        result = tool.handler(url="https://github.com/org/repo/blob/main/file.py#L50-L55")
        # Should include the targeted lines
        assert "line 50" in result
        assert "line 55" in result
        # Should NOT include lines far away
        assert "line 1:" not in result
        assert "line 99:" not in result

    def test_truncates_large_files(self):
        adapter = _make_mock_adapter()
        adapter.get_file_contents.return_value = "x" * 60_000
        tool = _tool_by_name(make_github_read_tools(adapter), "fetch_github_file")

        result = tool.handler(url="https://github.com/org/repo/blob/main/big.py")
        assert len(result) <= 60_000
        assert "truncated" in result

    def test_handles_adapter_error(self):
        adapter = _make_mock_adapter()
        adapter.get_file_contents.side_effect = RuntimeError("API error")
        tool = _tool_by_name(make_github_read_tools(adapter), "fetch_github_file")

        result = tool.handler(url="https://github.com/org/repo/blob/main/file.py")
        assert "Error" in result
        assert "API error" in result

    def test_handles_invalid_url(self):
        adapter = _make_mock_adapter()
        tool = _tool_by_name(make_github_read_tools(adapter), "fetch_github_file")
        result = tool.handler(url="https://github.com/org/repo")
        assert "Error" in result


# ---------------------------------------------------------------------------
# list_github_directory handler
# ---------------------------------------------------------------------------

class TestListGitHubDirectory:
    def test_lists_full_repo(self):
        adapter = _make_mock_adapter()
        adapter.get_repo_tree.return_value = [
            "src/app.py",
            "src/utils.py",
            "README.md",
        ]
        tool = _tool_by_name(make_github_read_tools(adapter), "list_github_directory")

        result = tool.handler(url="https://github.com/org/repo/tree/main")
        assert "src/app.py" in result
        assert "README.md" in result

    def test_filters_to_subdirectory(self):
        adapter = _make_mock_adapter()
        adapter.get_repo_tree.return_value = [
            "src/app.py",
            "src/utils.py",
            "tests/test_app.py",
            "README.md",
        ]
        tool = _tool_by_name(make_github_read_tools(adapter), "list_github_directory")

        result = tool.handler(url="https://github.com/org/repo/tree/main/src")
        assert "app.py" in result
        assert "utils.py" in result
        assert "test_app" not in result
        assert "README" not in result

    def test_filters_ignored_dirs(self):
        adapter = _make_mock_adapter()
        adapter.get_repo_tree.return_value = [
            "src/app.py",
            "__pycache__/foo.pyc",
            "node_modules/pkg/index.js",
        ]
        tool = _tool_by_name(make_github_read_tools(adapter), "list_github_directory")

        result = tool.handler(url="https://github.com/org/repo/tree/main")
        assert "src/app.py" in result
        assert "__pycache__" not in result
        assert "node_modules" not in result

    def test_handles_adapter_error(self):
        adapter = _make_mock_adapter()
        adapter.get_repo_tree.side_effect = RuntimeError("tree error")
        tool = _tool_by_name(make_github_read_tools(adapter), "list_github_directory")

        result = tool.handler(url="https://github.com/org/repo/tree/main")
        assert "Error" in result
        assert "tree error" in result

    def test_empty_subdirectory(self):
        adapter = _make_mock_adapter()
        adapter.get_repo_tree.return_value = ["src/app.py"]
        tool = _tool_by_name(make_github_read_tools(adapter), "list_github_directory")

        result = tool.handler(url="https://github.com/org/repo/tree/main/nonexistent")
        assert "No files found" in result
