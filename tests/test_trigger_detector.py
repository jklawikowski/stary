"""Tests for TriggerDetector split-JQL sensor refactoring.

Verifies that the 3-query approach minimises Jira API calls:
- do_it / pr_only triggers fetch comments only for matched tickets
- retry candidates trigger a get_comments call
"""

from unittest.mock import MagicMock

from stary.jira_adapter import JiraComment, JiraIssue
from stary.sensor import TriggerConfig, TriggerDetector


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
        jira.get_comments.return_value = [
            JiraComment(id="1", body="[~sys_qaplatformbot] do it", author="alice"),
        ]

        tickets = detector.poll()

        assert len(tickets) == 1
        assert tickets[0].key == "PROJ-1"
        assert tickets[0].auto_merge is True

    def test_comment_fetch_for_author(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [JiraIssue(key="PROJ-1")],
            [],
            [],
        ]
        jira.get_comments.return_value = [
            JiraComment(id="1", body="[~sys_qaplatformbot] do it", author="alice"),
        ]

        detector.poll()

        jira.get_comments.assert_called_once_with("PROJ-1")

    def test_has_priority_over_pr_only(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [JiraIssue(key="PROJ-1")],   # do_it
            [JiraIssue(key="PROJ-1")],   # pr_only (same ticket)
            [],                           # retry
        ]
        jira.get_comments.return_value = [
            JiraComment(id="1", body="[~sys_qaplatformbot] do it", author="alice"),
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
        jira.get_comments.return_value = [
            JiraComment(
                id="1", body="[~sys_qaplatformbot] pull request", author="bob",
            ),
        ]

        tickets = detector.poll()

        assert len(tickets) == 1
        assert tickets[0].key == "PROJ-2"
        assert tickets[0].auto_merge is False

    def test_comment_fetch_for_author(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [],
            [JiraIssue(key="PROJ-2")],
            [],
        ]
        jira.get_comments.return_value = [
            JiraComment(
                id="1", body="[~sys_qaplatformbot] pull request", author="bob",
            ),
        ]

        detector.poll()

        jira.get_comments.assert_called_once_with("PROJ-2")


# ---------------------------------------------------------------------------
# retry triggers
# ---------------------------------------------------------------------------


class TestRetryTrigger:
    def test_valid_retry_after_failure_fetches_comments(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [],                          # do_it
            [],                          # pr_only
            [JiraIssue(key="PROJ-3")],   # retry
        ]
        jira.get_comments.return_value = [
            JiraComment(
                id="1", body="[~sys_qaplatformbot] stary:failed", author="bot",
            ),
            JiraComment(
                id="2", body="[~sys_qaplatformbot] retry", author="carol",
            ),
        ]

        tickets = detector.poll()

        assert len(tickets) == 1
        assert tickets[0].key == "PROJ-3"
        assert tickets[0].auto_merge is True
        assert tickets[0].retry_count == 1
        jira.get_comments.assert_called_once_with("PROJ-3")

    def test_valid_retry_after_done_fetches_comments(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [],                          # do_it
            [],                          # pr_only
            [JiraIssue(key="PROJ-3")],   # retry
        ]
        jira.get_comments.return_value = [
            JiraComment(
                id="1", body="[~sys_qaplatformbot] do it", author="alice",
            ),
            JiraComment(
                id="2", body="[~sys_qaplatformbot] stary:done", author="bot",
            ),
            JiraComment(
                id="3", body="[~sys_qaplatformbot] retry", author="carol",
            ),
        ]

        tickets = detector.poll()

        assert len(tickets) == 1
        assert tickets[0].key == "PROJ-3"
        assert tickets[0].retry_count == 1
        jira.get_comments.assert_called_once_with("PROJ-3")

    def test_retry_older_than_done_is_skipped(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [],
            [],
            [JiraIssue(key="PROJ-3")],
        ]
        # retry older than done → invalid (pipeline already completed after retry)
        jira.get_comments.return_value = [
            JiraComment(
                id="1", body="[~sys_qaplatformbot] retry", author="carol",
            ),
            JiraComment(
                id="2", body="[~sys_qaplatformbot] stary:done", author="bot",
            ),
        ]

        tickets = detector.poll()

        assert len(tickets) == 0
        jira.get_comments.assert_called_once()

    def test_invalid_retry_is_skipped(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [],
            [],
            [JiraIssue(key="PROJ-3")],
        ]
        # retry older than failed → invalid
        jira.get_comments.return_value = [
            JiraComment(
                id="1", body="[~sys_qaplatformbot] retry", author="carol",
            ),
            JiraComment(
                id="2", body="[~sys_qaplatformbot] stary:failed", author="bot",
            ),
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

    def test_mixed_triggers_comment_fetches(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [JiraIssue(key="A-1"), JiraIssue(key="A-2")],  # do_it
            [JiraIssue(key="B-1")],                          # pr_only
            [JiraIssue(key="C-1")],                          # retry
        ]
        jira.get_comments.return_value = [
            JiraComment(
                id="1", body="[~sys_qaplatformbot] stary:failed", author="bot",
            ),
            JiraComment(
                id="2", body="[~sys_qaplatformbot] retry", author="carol",
            ),
        ]

        tickets = detector.poll()

        # 3 search calls, comment fetches for each matched ticket
        assert jira.search_issues.call_count == 3
        # do_it(2) + pr_only(1) + retry(1) = 4 comment fetches
        assert jira.get_comments.call_count == 4
        assert len(tickets) == 4


# ---------------------------------------------------------------------------
# JQL content
# ---------------------------------------------------------------------------


class TestJqlQueries:
    def test_do_it_jql_contains_trigger(self):
        detector, _ = _make_detector()
        jql = detector._build_do_it_jql()
        assert "do it" in jql

    def test_pr_only_jql_contains_trigger(self):
        detector, _ = _make_detector()
        jql = detector._build_pr_only_jql()
        assert "pull request" in jql

    def test_retry_jql_requires_terminal_marker(self):
        detector, _ = _make_detector()
        jql = detector._build_retry_jql()
        assert "retry" in jql
        assert "stary:failed" in jql
        assert "stary:done" in jql

    def test_do_it_jql_has_no_NOT_comment(self):
        """NOT comment ~ is broken on Jira Server; cursor handles dedup."""
        detector, _ = _make_detector()
        jql = detector._build_do_it_jql()
        assert "NOT comment" not in jql

    def test_pr_only_jql_has_no_NOT_comment(self):
        detector, _ = _make_detector()
        jql = detector._build_pr_only_jql()
        assert "NOT comment" not in jql

    def test_scheduled_jql_has_no_NOT_comment(self):
        detector, _ = _make_detector()
        jql = detector._build_scheduled_jql(["alice"])
        assert "NOT comment" not in jql
        assert "alice" in jql

    def test_all_queries_filter_by_updated_span(self):
        detector, _ = _make_detector()
        for jql in (
            detector._build_do_it_jql(),
            detector._build_pr_only_jql(),
            detector._build_retry_jql(),
        ):
            assert "updated >= -1d" in jql

    def test_custom_query_span_days(self):
        config = TriggerConfig(query_span_days=3)
        detector, _ = _make_detector()
        detector.config = config
        for jql in (
            detector._build_do_it_jql(),
            detector._build_pr_only_jql(),
            detector._build_retry_jql(),
        ):
            assert "updated >= -3d" in jql

    def test_single_label_filter(self):
        config = TriggerConfig(jira_labels=["gdn_qa"])
        detector, _ = _make_detector()
        detector.config = config
        for jql in (
            detector._build_do_it_jql(),
            detector._build_pr_only_jql(),
            detector._build_retry_jql(),
        ):
            assert 'labels in ("gdn_qa")' in jql

    def test_multiple_labels_filter(self):
        config = TriggerConfig(jira_labels=["gdn_qa", "stary"])
        detector, _ = _make_detector()
        detector.config = config
        for jql in (
            detector._build_do_it_jql(),
            detector._build_pr_only_jql(),
            detector._build_retry_jql(),
        ):
            assert 'labels in ("gdn_qa", "stary")' in jql

    def test_no_label_filter_when_empty(self):
        detector, _ = _make_detector()
        for jql in (
            detector._build_do_it_jql(),
            detector._build_pr_only_jql(),
            detector._build_retry_jql(),
        ):
            assert "labels" not in jql


# ---------------------------------------------------------------------------
# poll_comment_triggers (unified method)
# ---------------------------------------------------------------------------


class TestPollCommentTriggers:
    def test_returns_deduplicated_candidates(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [JiraIssue(key="PROJ-3")],   # retry
            [JiraIssue(key="PROJ-1")],   # do_it
            [JiraIssue(key="PROJ-1"), JiraIssue(key="PROJ-2")],  # pr_only (PROJ-1 dup)
        ]

        candidates = detector.poll_comment_triggers()

        keys = [c[0] for c in candidates]
        assert keys == ["PROJ-3", "PROJ-1", "PROJ-2"]

    def test_retry_has_priority_over_do_it_and_pr_only(self):
        """If same ticket in retry and pr_only JQL, hint should be retry_candidate."""
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [JiraIssue(key="PROJ-1")],   # retry
            [JiraIssue(key="PROJ-1")],   # do_it
            [JiraIssue(key="PROJ-1")],   # pr_only
        ]

        candidates = detector.poll_comment_triggers()

        assert len(candidates) == 1
        assert candidates[0] == ("PROJ-1", "https://jira.example.com/browse/PROJ-1", "retry_candidate")

    def test_assigns_correct_hints(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [JiraIssue(key="C-1")],
            [JiraIssue(key="A-1")],
            [JiraIssue(key="B-1")],
        ]

        candidates = detector.poll_comment_triggers()

        hints = {c[0]: c[2] for c in candidates}
        assert hints["C-1"] == "retry_candidate"
        assert hints["A-1"] == "do_it"
        assert hints["B-1"] == "pr_only"

    def test_three_jql_calls(self):
        detector, jira = _make_detector()
        jira.search_issues.return_value = []

        detector.poll_comment_triggers()

        assert jira.search_issues.call_count == 3

    def test_no_comment_fetches(self):
        """poll_comment_triggers doesn't fetch comments — that's the sensor's job."""
        detector, jira = _make_detector()
        jira.search_issues.return_value = []

        detector.poll_comment_triggers()

        jira.get_comments.assert_not_called()

    def test_empty_result(self):
        detector, jira = _make_detector()
        jira.search_issues.return_value = []

        candidates = detector.poll_comment_triggers()

        assert candidates == []


# ---------------------------------------------------------------------------
# trigger_author extraction
# ---------------------------------------------------------------------------


class TestTriggerAuthorExtraction:
    def test_do_it_extracts_trigger_author(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [JiraIssue(key="PROJ-1")],
            [],
            [],
        ]
        jira.get_comments.return_value = [
            JiraComment(
                id="1", body="[~sys_qaplatformbot] do it", author="alice",
            ),
        ]

        tickets = detector.poll()

        assert len(tickets) == 1
        assert tickets[0].trigger_author == "alice"

    def test_pr_only_extracts_trigger_author(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [],
            [JiraIssue(key="PROJ-2")],
            [],
        ]
        jira.get_comments.return_value = [
            JiraComment(
                id="1",
                body="[~sys_qaplatformbot] pull request",
                author="bob",
            ),
        ]

        tickets = detector.poll()

        assert len(tickets) == 1
        assert tickets[0].trigger_author == "bob"

    def test_retry_extracts_trigger_author(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [],
            [],
            [JiraIssue(key="PROJ-3")],
        ]
        jira.get_comments.return_value = [
            JiraComment(
                id="1", body="[~sys_qaplatformbot] stary:failed", author="bot",
            ),
            JiraComment(
                id="2", body="[~sys_qaplatformbot] retry", author="carol",
            ),
        ]

        tickets = detector.poll()

        assert len(tickets) == 1
        assert tickets[0].trigger_author == "carol"

    def test_trigger_author_is_last_matching_comment_author(self):
        """When multiple trigger comments exist, the LAST one's author is used."""
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [JiraIssue(key="PROJ-1")],
            [],
            [],
        ]
        jira.get_comments.return_value = [
            JiraComment(
                id="1", body="[~sys_qaplatformbot] do it", author="alice",
            ),
            JiraComment(
                id="2", body="some unrelated comment", author="dave",
            ),
            JiraComment(
                id="3", body="[~sys_qaplatformbot] do it", author="eve",
            ),
        ]

        tickets = detector.poll()

        assert len(tickets) == 1
        assert tickets[0].trigger_author == "eve"

    def test_trigger_author_empty_when_no_match(self):
        """If no matching comment is found, trigger_author is empty string."""
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [JiraIssue(key="PROJ-1")],
            [],
            [],
        ]
        jira.get_comments.return_value = []

        tickets = detector.poll()

        assert len(tickets) == 1
        assert tickets[0].trigger_author == ""

    def test_to_dict_includes_trigger_author(self):
        detector, jira = _make_detector()
        jira.search_issues.side_effect = [
            [JiraIssue(key="PROJ-1")],
            [],
            [],
        ]
        jira.get_comments.return_value = [
            JiraComment(
                id="1", body="[~sys_qaplatformbot] do it", author="alice",
            ),
        ]

        tickets = detector.poll()

        d = tickets[0].to_dict()
        assert d["trigger_author"] == "alice"
