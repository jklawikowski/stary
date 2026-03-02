"""Tests for the Sensor WIP comment formatting and posting."""

from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

from stary.sensor import Sensor


class TestFormatWipComment:
    """Unit tests for Sensor._format_wip_comment."""

    def _make_sensor(self) -> Sensor:
        return Sensor(
            jira_base_url="https://jira.example.com",
            jira_token="fake-token",
        )

    def test_comment_without_url(self):
        s = self._make_sensor()
        comment = s._format_wip_comment(dagster_run_url=None)
        assert "Pipeline has been triggered and is currently in progress." in comment
        assert "View live pipeline status" not in comment
        # Should not contain broken Jira markup
        assert "[|]" not in comment
        assert "[View live pipeline status|]" not in comment

    def test_comment_with_url(self):
        s = self._make_sensor()
        url = "https://dagster.example.com/runs/abc123"
        comment = s._format_wip_comment(dagster_run_url=url)
        assert "Pipeline has been triggered and is currently in progress." in comment
        assert f"[View live pipeline status|{url}]" in comment

    def test_comment_with_empty_string_url(self):
        """Empty string should be treated like None."""
        s = self._make_sensor()
        comment = s._format_wip_comment(dagster_run_url="")
        assert "View live pipeline status" not in comment

    def test_wip_marker_present(self):
        s = self._make_sensor()
        comment = s._format_wip_comment()
        assert s.wip_marker in comment


class TestMarkAsWip:
    """Integration-style tests (with mocked HTTP) for Sensor.mark_as_wip."""

    @patch("stary.sensor.requests.post")
    def test_posts_comment_with_dagster_url(self, mock_post: MagicMock):
        mock_post.return_value = MagicMock(status_code=201)
        mock_post.return_value.raise_for_status = MagicMock()

        s = Sensor(
            jira_base_url="https://jira.example.com",
            jira_token="fake-token",
        )
        dagster_url = "https://dagster.example.com/runs/run-42"
        s.mark_as_wip("PROJ-1", dagster_run_url=dagster_url)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert dagster_url in body["body"]
        assert "View live pipeline status" in body["body"]

    @patch("stary.sensor.requests.post")
    def test_posts_comment_without_dagster_url(self, mock_post: MagicMock):
        mock_post.return_value = MagicMock(status_code=201)
        mock_post.return_value.raise_for_status = MagicMock()

        s = Sensor(
            jira_base_url="https://jira.example.com",
            jira_token="fake-token",
        )
        s.mark_as_wip("PROJ-1")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "View live pipeline status" not in body["body"]
        assert "Pipeline has been triggered" in body["body"]

    @patch("stary.sensor.requests.post")
    def test_backward_compatible_no_url_arg(self, mock_post: MagicMock):
        """Calling mark_as_wip without dagster_run_url should work (backward compat)."""
        mock_post.return_value = MagicMock(status_code=201)
        mock_post.return_value.raise_for_status = MagicMock()

        s = Sensor(
            jira_base_url="https://jira.example.com",
            jira_token="fake-token",
        )
        # Should not raise
        s.mark_as_wip("PROJ-1")
        mock_post.assert_called_once()
