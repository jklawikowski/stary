"""Tests for TicketStatusMarker WIP comment formatting and posting."""

from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

from stary.jira_adapter import JiraAdapter, JiraComment
from stary.sensor import DEFAULT_BOT_ACCOUNT, TriggerDetector, TriggeredTicket
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


class TestAuthorMentionInComments:
    """Tests verifying the comment mentions the trigger author, not the bot."""

    def _make_status_marker(self) -> TicketStatusMarker:
        mock_jira = MagicMock()
        return TicketStatusMarker(mock_jira)

    # -- WIP comment author mention ------------------------------------

    def test_wip_comment_mentions_trigger_author(self):
        marker = self._make_status_marker()
        comment = marker.format_wip_comment(trigger_author="alice")
        assert comment.startswith("[~alice] stary:wip")
        assert "[~sys_qaplatformbot]" not in comment

    def test_wip_comment_different_authors(self):
        marker = self._make_status_marker()
        for author in ("alice", "bob", "charlie"):
            comment = marker.format_wip_comment(trigger_author=author)
            assert f"[~{author}]" in comment

    def test_wip_comment_fallback_when_author_none(self):
        marker = self._make_status_marker()
        comment = marker.format_wip_comment(trigger_author=None)
        assert comment.startswith("[~sys_qaplatformbot] stary:wip")

    def test_wip_comment_fallback_when_author_empty(self):
        marker = self._make_status_marker()
        comment = marker.format_wip_comment(trigger_author="")
        assert comment.startswith("[~sys_qaplatformbot] stary:wip")

    # -- Done comment author mention -----------------------------------

    def test_done_comment_mentions_trigger_author(self):
        marker = self._make_status_marker()
        comment = marker.format_done_comment(
            pr_url="https://github.com/org/repo/pull/1",
            status="APPROVED",
            trigger_author="alice",
        )
        assert "[~alice] stary:done" in comment
        assert "[~sys_qaplatformbot]" not in comment

    def test_done_comment_fallback_when_author_none(self):
        marker = self._make_status_marker()
        comment = marker.format_done_comment(
            pr_url="https://github.com/org/repo/pull/1",
            status="APPROVED",
            trigger_author=None,
        )
        assert "[~sys_qaplatformbot] stary:done" in comment

    # -- Failed comment author mention ---------------------------------

    def test_failed_comment_mentions_trigger_author(self):
        marker = self._make_status_marker()
        comment = marker.format_failed_comment(
            failed_step="agent2_implementer",
            error_message="Something went wrong",
            trigger_author="bob",
        )
        assert "[~bob] stary:failed" in comment
        assert "[~sys_qaplatformbot]" not in comment

    def test_failed_comment_fallback_when_author_none(self):
        marker = self._make_status_marker()
        comment = marker.format_failed_comment(
            failed_step="agent2_implementer",
            error_message="Something went wrong",
            trigger_author=None,
        )
        assert "[~sys_qaplatformbot] stary:failed" in comment


class TestMarkMethodsPassAuthor:
    """Verify mark_wip / mark_done / mark_failed forward trigger_author."""

    def _make_marker_with_mock(self):
        mock_jira = MagicMock()
        marker = TicketStatusMarker(mock_jira)
        return marker, mock_jira

    def test_mark_wip_passes_author_to_comment(self):
        marker, mock_jira = self._make_marker_with_mock()
        marker.mark_wip("PROJ-1", trigger_author="alice")
        body = mock_jira.add_comment.call_args[0][1]
        assert body.startswith("[~alice] stary:wip")

    def test_mark_done_passes_author_to_comment(self):
        marker, mock_jira = self._make_marker_with_mock()
        marker.mark_done(
            "PROJ-1",
            pr_url="https://github.com/org/repo/pull/1",
            status="APPROVED",
            trigger_author="alice",
        )
        body = mock_jira.add_comment.call_args[0][1]
        assert "[~alice] stary:done" in body

    def test_mark_failed_passes_author_to_comment(self):
        marker, mock_jira = self._make_marker_with_mock()
        marker.mark_failed(
            "PROJ-1",
            failed_step="agent2",
            error_message="err",
            trigger_author="alice",
        )
        body = mock_jira.add_comment.call_args[0][1]
        assert "[~alice] stary:failed" in body


class TestTriggeredTicketAuthor:
    """Verify TriggeredTicket stores and exposes trigger_author."""

    def test_default_author_is_bot(self):
        t = TriggeredTicket(key="X-1", url="http://x", auto_merge=True)
        assert t.trigger_author == DEFAULT_BOT_ACCOUNT

    def test_custom_author(self):
        t = TriggeredTicket(
            key="X-1", url="http://x", auto_merge=True, trigger_author="alice"
        )
        assert t.trigger_author == "alice"

    def test_to_dict_includes_author(self):
        t = TriggeredTicket(
            key="X-1", url="http://x", auto_merge=True, trigger_author="alice"
        )
        d = t.to_dict()
        assert d["trigger_author"] == "alice"


class TestTriggerDetectorAuthor:
    """Verify TriggerDetector.parse_trigger_type returns correct author."""

    def _make_detector(self) -> TriggerDetector:
        mock_jira = MagicMock()
        return TriggerDetector(mock_jira)

    def test_do_it_trigger_returns_comment_author(self):
        detector = self._make_detector()
        comments = [
            JiraComment(id="1", body="[~sys_qaplatformbot] do it", author="alice"),
        ]
        trigger_type, _, author = detector.parse_trigger_type(comments)
        assert trigger_type == "do_it"
        assert author == "alice"

    def test_pr_only_trigger_returns_comment_author(self):
        detector = self._make_detector()
        comments = [
            JiraComment(id="1", body="[~sys_qaplatformbot] pull request", author="bob"),
        ]
        trigger_type, _, author = detector.parse_trigger_type(comments)
        assert trigger_type == "pr_only"
        assert author == "bob"

    def test_multiple_trigger_comments_returns_last_author(self):
        detector = self._make_detector()
        comments = [
            JiraComment(id="1", body="[~sys_qaplatformbot] do it", author="alice"),
            JiraComment(id="2", body="[~sys_qaplatformbot] do it", author="bob"),
        ]
        _, _, author = detector.parse_trigger_type(comments)
        assert author == "bob"

    def test_no_trigger_returns_bot_account(self):
        detector = self._make_detector()
        comments = [
            JiraComment(id="1", body="just a regular comment", author="alice"),
        ]
        trigger_type, _, author = detector.parse_trigger_type(comments)
        assert trigger_type is None
        assert author == DEFAULT_BOT_ACCOUNT


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
    def test_mark_wip_with_trigger_author(self, mock_request: MagicMock):
        """mark_wip with trigger_author mentions that author in the comment."""
        mock_response = MagicMock(status_code=201, ok=True)
        mock_response.json.return_value = {"id": "123", "body": "test"}
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        jira = JiraAdapter(
            base_url="https://jira.example.com",
            token="fake-token",
        )
        marker = TicketStatusMarker(jira)
        marker.mark_wip("PROJ-1", trigger_author="alice")

        call_kwargs = mock_request.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["body"].startswith("[~alice] stary:wip")
        assert "[~sys_qaplatformbot]" not in body["body"]
