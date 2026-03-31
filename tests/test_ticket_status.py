"""Tests for TicketStatusMarker _build_marker and mention_user support."""

from unittest.mock import MagicMock

from stary.ticket_status import StatusMarkerConfig, TicketStatusMarker


def _make_marker(config: StatusMarkerConfig | None = None) -> TicketStatusMarker:
    """Create a TicketStatusMarker with a mock Jira client."""
    mock_jira = MagicMock()
    return TicketStatusMarker(mock_jira, config=config)


# ------------------------------------------------------------------
# _build_marker
# ------------------------------------------------------------------


class TestBuildMarker:
    def test_no_mention_user_returns_original(self):
        marker = _make_marker()
        base = "[~sys_qaplatformbot] stary:wip"
        assert marker._build_marker(base, "") == base

    def test_replaces_mention_with_custom_user(self):
        marker = _make_marker()
        result = marker._build_marker(
            "[~sys_qaplatformbot] stary:wip", "john.doe"
        )
        assert result == "[~john.doe] stary:wip"

    def test_preserves_stary_suffix(self):
        marker = _make_marker()
        for suffix in ("stary:wip", "stary:done", "stary:failed"):
            base = f"[~sys_qaplatformbot] {suffix}"
            result = marker._build_marker(base, "alice")
            assert suffix in result


# ------------------------------------------------------------------
# format_wip_comment
# ------------------------------------------------------------------


class TestFormatWipComment:
    def test_default_mentions_bot(self):
        marker = _make_marker()
        comment = marker.format_wip_comment()
        assert "[~sys_qaplatformbot]" in comment
        assert "stary:wip" in comment

    def test_custom_mention_user(self):
        marker = _make_marker()
        comment = marker.format_wip_comment(mention_user="alice")
        first_line = comment.split("\n")[0]
        assert "[~alice]" in first_line
        assert "stary:wip" in first_line

    def test_includes_dagster_run_url(self):
        marker = _make_marker()
        url = "https://dagster.example.com/runs/abc123"
        comment = marker.format_wip_comment(
            dagster_run_url=url, mention_user="bob",
        )
        assert url in comment
        assert "[~bob]" in comment


# ------------------------------------------------------------------
# format_done_comment
# ------------------------------------------------------------------


class TestFormatDoneComment:
    def test_default_mentions_bot(self):
        marker = _make_marker()
        comment = marker.format_done_comment(
            pr_url="https://github.com/org/repo/pull/1",
            status="APPROVED",
        )
        assert "[~sys_qaplatformbot]" in comment
        assert "stary:done" in comment

    def test_custom_mention_user(self):
        marker = _make_marker()
        comment = marker.format_done_comment(
            pr_url="https://github.com/org/repo/pull/1",
            status="APPROVED",
            mention_user="charlie",
        )
        first_line = comment.split("\n")[0]
        assert "[~charlie]" in first_line
        assert "stary:done" in first_line


# ------------------------------------------------------------------
# format_failed_comment
# ------------------------------------------------------------------


class TestFormatFailedComment:
    def test_default_mentions_bot(self):
        marker = _make_marker()
        comment = marker.format_failed_comment(
            failed_step="code_gen",
            error_message="Something went wrong",
        )
        assert "[~sys_qaplatformbot]" in comment
        assert "stary:failed" in comment

    def test_custom_mention_user(self):
        marker = _make_marker()
        comment = marker.format_failed_comment(
            failed_step="code_gen",
            error_message="Something went wrong",
            mention_user="dave",
        )
        first_line = comment.split("\n")[0]
        assert "[~dave]" in first_line
        assert "stary:failed" in first_line
