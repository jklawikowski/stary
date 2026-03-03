"""Tests for TicketStatusMarker WIP comment formatting and posting."""

from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

from stary.jira_adapter import JiraAdapter
from stary.ticket_status import StatusMarkerConfig, TicketStatusMarker


class TestFormatWipComment:
    """Unit tests for TicketStatusMarker.format_wip_comment."""

    def _make_status_marker(self) -> TicketStatusMarker:
        # Create a mock jira client
        mock_jira = MagicMock()
        return TicketStatusMarker(mock_jira)

    def test_comment_without_url(self):
        marker = self._make_status_marker()
        comment = marker.format_wip_comment(dagster_run_url=None)
        assert "Pipeline has been triggered and is currently in progress." in comment
        assert "View live pipeline status" not in comment
        # Should not contain broken Jira markup
        assert "[|]" not in comment
        assert "[View live pipeline status|]" not in comment

    def test_comment_with_url(self):
        marker = self._make_status_marker()
        url = "https://dagster.example.com/runs/abc123"
        comment = marker.format_wip_comment(dagster_run_url=url)
        assert "Pipeline has been triggered and is currently in progress." in comment
        assert f"[View live pipeline status|{url}]" in comment

    def test_comment_with_empty_string_url(self):
        """Empty string should be treated like None."""
        marker = self._make_status_marker()
        comment = marker.format_wip_comment(dagster_run_url="")
        assert "View live pipeline status" not in comment

    def test_wip_marker_present(self):
        marker = self._make_status_marker()
        comment = marker.format_wip_comment()
        assert marker.config.wip_marker in comment


class TestMarkWip:
    """Integration-style tests (with mocked HTTP) for TicketStatusMarker.mark_wip."""

    @patch("stary.jira_adapter.requests.Session.request")
    def test_posts_comment_with_dagster_url(self, mock_request: MagicMock):
        mock_response = MagicMock(status_code=201, ok=True)
        mock_response.json.return_value = {"id": "123", "body": "test"}
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        jira = JiraAdapter(
            base_url="https://jira.example.com",
            token="fake-token",
        )
        marker = TicketStatusMarker(jira)
        dagster_url = "https://dagster.example.com/runs/run-42"
        marker.mark_wip("PROJ-1", dagster_run_url=dagster_url)

        mock_request.assert_called_once()
        call_kwargs = mock_request.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert dagster_url in body["body"]
        assert "View live pipeline status" in body["body"]

    @patch("stary.jira_adapter.requests.Session.request")
    def test_posts_comment_without_dagster_url(self, mock_request: MagicMock):
        mock_response = MagicMock(status_code=201, ok=True)
        mock_response.json.return_value = {"id": "123", "body": "test"}
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        jira = JiraAdapter(
            base_url="https://jira.example.com",
            token="fake-token",
        )
        marker = TicketStatusMarker(jira)
        marker.mark_wip("PROJ-1")

        mock_request.assert_called_once()
        call_kwargs = mock_request.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "View live pipeline status" not in body["body"]
        assert "Pipeline has been triggered" in body["body"]

    @patch("stary.jira_adapter.requests.Session.request")
    def test_mark_wip_no_url_arg(self, mock_request: MagicMock):
        """Calling mark_wip without dagster_run_url should work."""
        mock_response = MagicMock(status_code=201, ok=True)
        mock_response.json.return_value = {"id": "123", "body": "test"}
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        jira = JiraAdapter(
            base_url="https://jira.example.com",
            token="fake-token",
        )
        marker = TicketStatusMarker(jira)
        # Should not raise
        marker.mark_wip("PROJ-1")
        mock_request.assert_called_once()
