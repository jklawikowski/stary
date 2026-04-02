"""Stary – Software Task Agentic Resolution sYstem."""

from stary.github_adapter import GitHubAdapter, PRFile, PullRequest, RepoFile
from stary.jenkins_adapter import (
    JenkinsAdapter,
    JenkinsBuild,
    JenkinsTestCase,
    JenkinsTestReport,
)
from stary.jira_adapter import JiraAdapter, JiraComment, JiraIssue
from stary.sensor import TriggerConfig, TriggerDetector, TriggeredTicket
from stary.ticket_status import StatusMarkerConfig, TicketStatusMarker

__all__ = [
    # GitHub adapter
    "GitHubAdapter",
    "PullRequest",
    "PRFile",
    "RepoFile",
    # Jenkins adapter
    "JenkinsAdapter",
    "JenkinsBuild",
    "JenkinsTestCase",
    "JenkinsTestReport",
    # Jira adapter
    "JiraAdapter",
    "JiraComment",
    "JiraIssue",
    # Trigger detection
    "TriggerDetector",
    "TriggerConfig",
    "TriggeredTicket",
    # Status markers
    "TicketStatusMarker",
    "StatusMarkerConfig",
]
