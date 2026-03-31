"""Tests for the lifecycle agent (Agent #4)."""

from unittest.mock import MagicMock

from stary.agents.lifecycle import LifecycleAgent, LifecycleConfig


class TestLifecycleAgent:
    def test_transitions_ticket_on_approved_and_merged(self):
        jira = MagicMock()
        jira.transition_issue.return_value = True
        agent = LifecycleAgent(jira=jira)

        result = agent.run(
            ticket_key="PROJ-1",
            pr_urls=["https://github.com/org/repo/pull/1"],
            all_approved=True,
            merged=True,
        )

        assert result["transitioned"] is True
        assert result["transition_status"] == "Done"
        jira.transition_issue.assert_called_once_with("PROJ-1", "Done")

    def test_skips_transition_when_not_approved(self):
        jira = MagicMock()
        agent = LifecycleAgent(jira=jira)

        result = agent.run(
            ticket_key="PROJ-2",
            pr_urls=["https://github.com/org/repo/pull/2"],
            all_approved=False,
            merged=True,
        )

        assert result["transitioned"] is False
        assert result["transition_status"] == ""
        jira.transition_issue.assert_not_called()

    def test_skips_transition_when_not_merged(self):
        jira = MagicMock()
        agent = LifecycleAgent(jira=jira)

        result = agent.run(
            ticket_key="PROJ-3",
            pr_urls=["https://github.com/org/repo/pull/3"],
            all_approved=True,
            merged=False,
        )

        assert result["transitioned"] is False
        assert result["transition_status"] == ""
        jira.transition_issue.assert_not_called()

    def test_skips_transition_when_disabled(self):
        jira = MagicMock()
        config = LifecycleConfig(transition_on_approval=False)
        agent = LifecycleAgent(jira=jira, config=config)

        result = agent.run(
            ticket_key="PROJ-4",
            pr_urls=["https://github.com/org/repo/pull/4"],
            all_approved=True,
            merged=True,
        )

        assert result["transitioned"] is False
        assert result["transition_status"] == ""
        jira.transition_issue.assert_not_called()

    def test_handles_transition_failure_gracefully(self):
        jira = MagicMock()
        jira.transition_issue.side_effect = RuntimeError("API error")
        agent = LifecycleAgent(jira=jira)

        result = agent.run(
            ticket_key="PROJ-5",
            pr_urls=["https://github.com/org/repo/pull/5"],
            all_approved=True,
            merged=True,
        )

        assert result["transitioned"] is False
        assert "error" in result["transition_status"]

    def test_transition_unavailable(self):
        jira = MagicMock()
        jira.transition_issue.return_value = False
        agent = LifecycleAgent(jira=jira)

        result = agent.run(
            ticket_key="PROJ-6",
            pr_urls=[],
            all_approved=True,
            merged=True,
        )

        assert result["transitioned"] is False
        assert result["transition_status"] == "transition_unavailable"


class TestCreateRegressionTicket:
    def test_posts_regression_comment(self):
        jira = MagicMock()
        agent = LifecycleAgent(jira=jira)

        result = agent.create_regression_ticket(
            original_ticket_key="PROJ-10",
            pr_url="https://github.com/org/repo/pull/10",
            failure_details="Test suite failed: 3 tests broken",
        )

        assert result == "PROJ-10"
        jira.add_comment.assert_called_once()
        call_args = jira.add_comment.call_args
        assert call_args[0][0] == "PROJ-10"
        body = call_args[0][1]
        assert "regression" in body
        assert "https://github.com/org/repo/pull/10" in body
        assert "Test suite failed: 3 tests broken" in body

    def test_handles_comment_failure(self):
        jira = MagicMock()
        jira.add_comment.side_effect = RuntimeError("API error")
        agent = LifecycleAgent(jira=jira)

        result = agent.create_regression_ticket(
            original_ticket_key="PROJ-11",
            pr_url="https://github.com/org/repo/pull/11",
            failure_details="Build failed",
        )

        assert result is None
