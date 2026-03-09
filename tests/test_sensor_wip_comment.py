"""Tests for TicketStatusMarker WIP comment formatting and posting."""

from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

from stary.jira_adapter import JiraAdapter
from stary.ticket_status import StatusMarkerConfig, TicketStatusMarker


class TestFormatWipComment:
    """Unit tests for TicketStatusMarker.format_wip_comment."""

    def _make_status_marker(self, **kwargs) -> TicketStatusMarker:
        # Create a mock jira client
        mock_jira = MagicMock()
        config = StatusMarkerConfig(**kwargs) if kwargs else None
        return TicketStatusMarker(mock_jira, config=config)

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

    def test_comment_mentions_triggering_user(self):
        """When mention_user is provided, the comment should mention that user."""
        marker = self._make_status_marker()
        comment = marker.format_wip_comment(mention_user="alice")
        assert "[~alice]" in comment
        assert "[~sys_qaplatformbot]" not in comment

    def test_comment_mentions_different_users(self):
        """Different triggering users produce comments mentioning the respective user."""
        marker = self._make_status_marker()
        for user in ("alice", "bob", "charlie"):
            comment = marker.format_wip_comment(mention_user=user)
            assert f"[~{user}]" in comment
            assert "[~sys_qaplatformbot]" not in comment

    def test_comment_falls_back_to_bot_when_mention_user_none(self):
        """When mention_user is None, the default bot account is used."""
        marker = self._make_status_marker()
        comment = marker.format_wip_comment(mention_user=None)
        assert "[~sys_qaplatformbot]" in comment

    def test_comment_falls_back_to_bot_when_mention_user_empty(self):
        """When mention_user is an empty string, the default bot account is used."""
        marker = self._make_status_marker()
        comment = marker.format_wip_comment(mention_user="")
        assert "[~sys_qaplatformbot]" in comment

    def test_config_level_mention_user(self):
        """mention_user set on StatusMarkerConfig is used when not overridden."""
        marker = self._make_status_marker(mention_user="config_user")
        comment = marker.format_wip_comment()
        assert "[~config_user]" in comment
        assert "[~sys_qaplatformbot]" not in comment

    def test_call_level_mention_user_overrides_config(self):
        """mention_user passed at call time overrides the config-level value."""
        marker = self._make_status_marker(mention_user="config_user")
        comment = marker.format_wip_comment(mention_user="call_user")
        assert "[~call_user]" in comment
        assert "[~config_user]" not in comment


class TestFormatDoneComment:
    """Unit tests for done comment author mention."""

    def _make_status_marker(self) -> TicketStatusMarker:
        mock_jira = MagicMock()
        return TicketStatusMarker(mock_jira)

    def test_done_comment_mentions_triggering_user(self):
        marker = self._make_status_marker()
        comment = marker.format_done_comment("http://pr", "APPROVED", mention_user="alice")
        assert "[~alice]" in comment
        assert "[~sys_qaplatformbot]" not in comment

    def test_done_comment_falls_back_to_bot_when_no_user(self):
        marker = self._make_status_marker()
        comment = marker.format_done_comment("http://pr", "APPROVED")
        assert "[~sys_qaplatformbot]" in comment


class TestFormatFailedComment:
    """Unit tests for failed comment author mention."""

    def _make_status_marker(self) -> TicketStatusMarker:
        mock_jira = MagicMock()
        return TicketStatusMarker(mock_jira)

    def test_failed_comment_mentions_triggering_user(self):
        marker = self._make_status_marker()
        comment = marker.format_failed_comment("read_ticket", "timeout", mention_user="bob")
        assert "[~bob]" in comment
        assert "[~sys_qaplatformbot]" not in comment

    def test_failed_comment_falls_back_to_bot_when_no_user(self):
        marker = self._make_status_marker()
        comment = marker.format_failed_comment("read_ticket", "timeout")
        assert "[~sys_qaplatformbot]" in comment


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

    @patch("stary.jira_adapter.requests.Session.request")
    def test_mark_wip_with_mention_user(self, mock_request: MagicMock):
        """mark_wip with mention_user posts a comment mentioning that user."""
        mock_response = MagicMock(status_code=201, ok=True)
        mock_response.json.return_value = {"id": "123", "body": "test"}
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        jira = JiraAdapter(
            base_url="https://jira.example.com",
            token="fake-token",
        )
        marker = TicketStatusMarker(jira)
        marker.mark_wip("PROJ-1", mention_user="alice")

        mock_request.assert_called_once()
        call_kwargs = mock_request.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "[~alice]" in body["body"]
        assert "[~sys_qaplatformbot]" not in body["body"]

    @patch("stary.jira_adapter.requests.Session.request")
    def test_mark_wip_without_mention_user_uses_bot(self, mock_request: MagicMock):
        """mark_wip without mention_user defaults to the bot account."""
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
        assert "[~sys_qaplatformbot]" in body["body"]
