"""Tests for TriggerDetector split-JQL sensor refactoring.

Verifies that the 3-query approach minimises Jira API calls:
- do_it / pr_only triggers need zero comment fetches
- only retry candidates trigger a get_comments call
"""

from unittest.mock import MagicMock

from stary.jira_adapter import JiraComment, JiraIssue
from stary.sensor import TriggerDetector


def _make_detector():
    jira = MagicMock()
    jira.base_url = "https://jira.example.com"
    jira.build_browse_url.side_effect = lambda k: f"https://jira.example.com/browse/{k}"
    return TriggerDetector(jira), jira


# ---------------------------------------------------------------------------
# do_it triggers
# ---------------------------------------------------------------------------


class TestDoItTrigger:
    def test_returns_ticket_with_auto_merge(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [JiraIssue(key="PROJ-1")],  # do_it
            [],                          # pr_only
            [],                          # retry
        ]

        tickets = detector.poll()

        assert len(tickets) == 1
        assert tickets[0].key == "PROJ-1"
        assert tickets[0].auto_merge is True

    def test_no_comment_fetch(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [JiraIssue(key="PROJ-1")],
            [],
            [],
        ]

        detector.poll()

        jira.get_comments.assert_not_called()

    def test_has_priority_over_pr_only(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [JiraIssue(key="PROJ-1")],   # do_it
            [JiraIssue(key="PROJ-1")],   # pr_only (same ticket)
            [],                           # retry
        ]

        tickets = detector.poll()

        assert len(tickets) == 1
        assert tickets[0].auto_merge is True  # do_it wins


# ---------------------------------------------------------------------------
# pr_only triggers
# ---------------------------------------------------------------------------


class TestPrOnlyTrigger:
    def test_returns_ticket_without_auto_merge(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [],                          # do_it
            [JiraIssue(key="PROJ-2")],   # pr_only
            [],                          # retry
        ]

        tickets = detector.poll()

        assert len(tickets) == 1
        assert tickets[0].key == "PROJ-2"
        assert tickets[0].auto_merge is False

    def test_no_comment_fetch(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [],
            [JiraIssue(key="PROJ-2")],
            [],
        ]

        detector.poll()

        jira.get_comments.assert_not_called()


# ---------------------------------------------------------------------------
# retry triggers
# ---------------------------------------------------------------------------


class TestRetryTrigger:
    def test_valid_retry_fetches_comments(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [],                          # do_it
            [],                          # pr_only
            [JiraIssue(key="PROJ-3")],   # retry
        ]
        jira.get_comments.return_value = [
            JiraComment(id="1", body="[~sys_qaplatformbot] stary:failed"),
            JiraComment(id="2", body="[~sys_qaplatformbot] retry"),
        ]

        tickets = detector.poll()

        assert len(tickets) == 1
        assert tickets[0].key == "PROJ-3"
        assert tickets[0].auto_merge is True
        assert tickets[0].retry_count == 1
        jira.get_comments.assert_called_once_with("PROJ-3")

    def test_invalid_retry_is_skipped(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [],
            [],
            [JiraIssue(key="PROJ-3")],
        ]
        # retry older than failed → invalid
        jira.get_comments.return_value = [
            JiraComment(id="1", body="[~sys_qaplatformbot] retry"),
            JiraComment(id="2", body="[~sys_qaplatformbot] stary:failed"),
        ]

        tickets = detector.poll()

        assert len(tickets) == 0
        jira.get_comments.assert_called_once()


# ---------------------------------------------------------------------------
# API call count
# ---------------------------------------------------------------------------


class TestApiCallCount:
    def test_no_candidates_three_searches_zero_comments(self):
        detector, jira = _make_detector()
        jira.search_issues.return_value = []

        detector.poll()

        assert jira.search_issues.call_count == 3
        jira.get_comments.assert_not_called()

    def test_mixed_triggers_minimal_comment_fetches(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [JiraIssue(key="A-1"), JiraIssue(key="A-2")],  # do_it
            [JiraIssue(key="B-1")],                          # pr_only
            [JiraIssue(key="C-1")],                          # retry
        ]
        jira.get_comments.return_value = [
            JiraComment(id="1", body="[~sys_qaplatformbot] stary:failed"),
            JiraComment(id="2", body="[~sys_qaplatformbot] retry"),
        ]

        tickets = detector.poll()

        # 3 search calls, 1 comment fetch (only for retry candidate)
        assert jira.search_issues.call_count == 3
        assert jira.get_comments.call_count == 1
        assert len(tickets) == 4


# ---------------------------------------------------------------------------
# JQL content
# ---------------------------------------------------------------------------


class TestJqlQueries:
    def test_do_it_jql_excludes_failed_marker(self):
        detector, _ = _make_detector()
        jql = detector._build_do_it_jql()
        assert "do it" in jql
        assert "stary:failed" in jql

    def test_pr_only_jql_excludes_failed_marker(self):
        detector, _ = _make_detector()
        jql = detector._build_pr_only_jql()
        assert "pull request" in jql
        assert "stary:failed" in jql

    def test_retry_jql_requires_failed_marker(self):
        detector, _ = _make_detector()
        jql = detector._build_retry_jql()
        assert "retry" in jql
        assert "stary:failed" in jql

    def test_all_queries_filter_by_updated_within_7_days(self):
        detector, _ = _make_detector()
        for jql in (
            detector._build_do_it_jql(),
            detector._build_pr_only_jql(),
            detector._build_retry_jql(),
        ):
            assert "updated >= -7d" in jql