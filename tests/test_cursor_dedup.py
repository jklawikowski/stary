"""Tests for cursor-based deduplication in Dagster sensors.

Verifies that _load_cursor / _save_cursor helpers and sensor-level
filtering prevent duplicate ticket processing across evaluation cycles.

Updated for the unified stary_comment_sensor / stary_users_sensor
architecture.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from dagster import build_sensor_context

from stary.dagster.defs.sensors import (
    CURSOR_RETENTION_DAYS,
    _load_cursor,
    _save_cursor,
)


# ---------------------------------------------------------------------------
# _load_cursor
# ---------------------------------------------------------------------------


class TestLoadCursor:
    def test_returns_empty_dict_on_none(self):
        assert _load_cursor(None) == {}

    def test_returns_empty_dict_on_empty_string(self):
        assert _load_cursor("") == {}

    def test_returns_empty_dict_on_invalid_json(self):
        assert _load_cursor("not-json") == {}

    def test_deserializes_valid_json(self):
        now = datetime.now(timezone.utc).isoformat()
        raw = json.dumps({"PROJ-1": now})
        result = _load_cursor(raw)
        assert result == {"PROJ-1": now}

    def test_prunes_entries_older_than_retention(self):
        now = datetime.now(timezone.utc).isoformat()
        old = (datetime.now(timezone.utc) - timedelta(days=CURSOR_RETENTION_DAYS + 1)).isoformat()
        raw = json.dumps({"PROJ-1": now, "PROJ-OLD": old})

        result = _load_cursor(raw)

        assert "PROJ-1" in result
        assert "PROJ-OLD" not in result

    def test_keeps_entries_within_retention(self):
        recent = (datetime.now(timezone.utc) - timedelta(days=CURSOR_RETENTION_DAYS - 1)).isoformat()
        raw = json.dumps({"PROJ-1": recent})

        result = _load_cursor(raw)
        assert "PROJ-1" in result


# ---------------------------------------------------------------------------
# _save_cursor
# ---------------------------------------------------------------------------


class TestSaveCursor:
    def test_serializes_to_json(self):
        data = {"PROJ-1": "2026-03-24T10:00:00+00:00"}
        result = _save_cursor(data)
        assert json.loads(result) == data

    def test_roundtrip(self):
        now = datetime.now(timezone.utc).isoformat()
        data = {"PROJ-1": now, "PROJ-2": now}
        assert _load_cursor(_save_cursor(data)) == data


# ---------------------------------------------------------------------------
# Sensor-level deduplication (stary_users_sensor)
# ---------------------------------------------------------------------------


class TestScheduledSensorCursor:
    """Verify stary_users_sensor skips already-processed tickets."""

    @patch.dict("os.environ", {
        "STARY_SCHEDULE_USERS": "alice",
        "JIRA_BASE_URL": "https://jira.example.com",
        "JIRA_TOKEN": "fake-token",
    })
    @patch("stary.dagster.defs.sensors.JiraAdapter")
    @patch("stary.dagster.defs.sensors.TriggerDetector")
    @patch("stary.dagster.defs.sensors.TicketStateValidator")
    def test_skips_ticket_already_in_cursor(self, MockValidator, MockDetector, MockJira):
        MockDetector.return_value.poll_scheduled_candidates.return_value = [
            ("PROJ-1", "https://jira.example.com/browse/PROJ-1"),
        ]

        now = datetime.now(timezone.utc).isoformat()
        cursor = json.dumps({"PROJ-1": now})

        from stary.dagster.defs.sensors import stary_users_sensor
        context = build_sensor_context(cursor=cursor)
        result = stary_users_sensor.evaluate_tick(context)

        assert len(result.run_requests) == 0
        assert context.cursor is not None

    @patch.dict("os.environ", {
        "STARY_SCHEDULE_USERS": "alice",
        "JIRA_BASE_URL": "https://jira.example.com",
        "JIRA_TOKEN": "fake-token",
    })
    @patch("stary.dagster.defs.sensors.JiraAdapter")
    @patch("stary.dagster.defs.sensors.TriggerDetector")
    @patch("stary.dagster.defs.sensors.TicketStateValidator")
    def test_processes_idle_ticket_not_in_cursor(self, MockValidator, MockDetector, MockJira):
        MockDetector.return_value.poll_scheduled_candidates.return_value = [
            ("PROJ-2", "https://jira.example.com/browse/PROJ-2"),
        ]
        MockJira.return_value.get_comments.return_value = []
        MockValidator.return_value.resolve_scheduled.return_value = True

        from stary.dagster.defs.sensors import stary_users_sensor
        context = build_sensor_context(cursor=None)
        result = stary_users_sensor.evaluate_tick(context)

        assert len(result.run_requests) == 1
        assert result.run_requests[0].tags["ticket_key"] == "PROJ-2"
        saved = json.loads(context.cursor)
        assert "PROJ-2" in saved

    @patch.dict("os.environ", {
        "STARY_SCHEDULE_USERS": "alice",
        "JIRA_BASE_URL": "https://jira.example.com",
        "JIRA_TOKEN": "fake-token",
    })
    @patch("stary.dagster.defs.sensors.JiraAdapter")
    @patch("stary.dagster.defs.sensors.TriggerDetector")
    @patch("stary.dagster.defs.sensors.TicketStateValidator")
    def test_rejects_done_ticket(self, MockValidator, MockDetector, MockJira):
        """Scheduled sensor rejects tickets already handled by stary."""
        MockDetector.return_value.poll_scheduled_candidates.return_value = [
            ("PROJ-3", "https://jira.example.com/browse/PROJ-3"),
        ]
        MockJira.return_value.get_comments.return_value = [
            MagicMock(body="[~sys_qaplatformbot] stary:done"),
        ]
        MockValidator.return_value.resolve_scheduled.return_value = False

        from stary.dagster.defs.sensors import stary_users_sensor
        context = build_sensor_context(cursor=None)
        result = stary_users_sensor.evaluate_tick(context)

        assert len(result.run_requests) == 0
        # Still added to cursor so we don't re-check
        saved = json.loads(context.cursor)
        assert "PROJ-3" in saved


# ---------------------------------------------------------------------------
# Sensor-level deduplication (stary_comment_sensor)
# ---------------------------------------------------------------------------


class TestCommentSensorCursor:
    @patch.dict("os.environ", {
        "JIRA_BASE_URL": "https://jira.example.com",
        "JIRA_TOKEN": "fake-token",
    })
    @patch("stary.dagster.defs.sensors.JiraAdapter")
    @patch("stary.dagster.defs.sensors.TriggerDetector")
    @patch("stary.dagster.defs.sensors.TicketStateValidator")
    def test_skips_ticket_in_cursor(self, MockValidator, MockDetector, MockJira):
        MockDetector.return_value.poll_comment_triggers.return_value = [
            ("PROJ-1", "https://jira.example.com/browse/PROJ-1", "do_it"),
        ]

        now = datetime.now(timezone.utc).isoformat()

        from stary.dagster.defs.sensors import stary_comment_sensor
        context = build_sensor_context(cursor=json.dumps({"PROJ-1": now}))
        result = stary_comment_sensor.evaluate_tick(context)

        assert len(result.run_requests) == 0

    @patch.dict("os.environ", {
        "JIRA_BASE_URL": "https://jira.example.com",
        "JIRA_TOKEN": "fake-token",
    })
    @patch("stary.dagster.defs.sensors.JiraAdapter")
    @patch("stary.dagster.defs.sensors.TriggerDetector")
    @patch("stary.dagster.defs.sensors.TicketStateValidator")
    def test_processes_new_do_it_ticket(self, MockValidator, MockDetector, MockJira):
        MockDetector.return_value.poll_comment_triggers.return_value = [
            ("PROJ-3", "https://jira.example.com/browse/PROJ-3", "do_it"),
        ]
        MockJira.return_value.get_comments.return_value = []
        MockValidator.return_value.resolve_trigger.return_value = ("do_it", 0, True, "")

        from stary.dagster.defs.sensors import stary_comment_sensor
        context = build_sensor_context(cursor=None)
        result = stary_comment_sensor.evaluate_tick(context)

        assert len(result.run_requests) == 1
        saved = json.loads(context.cursor)
        assert "PROJ-3" in saved


# ---------------------------------------------------------------------------
# Retry sensor: allows same ticket with different retry_count (shared cursor)
# ---------------------------------------------------------------------------


class TestRetryCursorInCommentSensor:
    @patch.dict("os.environ", {
        "JIRA_BASE_URL": "https://jira.example.com",
        "JIRA_TOKEN": "fake-token",
    })
    @patch("stary.dagster.defs.sensors.JiraAdapter")
    @patch("stary.dagster.defs.sensors.TriggerDetector")
    @patch("stary.dagster.defs.sensors.TicketStateValidator")
    def test_allows_same_ticket_new_retry_count(self, MockValidator, MockDetector, MockJira):
        MockDetector.return_value.poll_comment_triggers.return_value = [
            ("PROJ-1", "https://jira.example.com/browse/PROJ-1", "retry_candidate"),
        ]
        MockJira.return_value.get_comments.return_value = []
        MockValidator.return_value.resolve_trigger.return_value = ("retry", 2, True, "")

        now = datetime.now(timezone.utc).isoformat()

        from stary.dagster.defs.sensors import stary_comment_sensor
        context = build_sensor_context(cursor=json.dumps({"PROJ-1-1": now}))
        result = stary_comment_sensor.evaluate_tick(context)

        assert len(result.run_requests) == 1
        saved = json.loads(context.cursor)
        assert "PROJ-1-2" in saved

    @patch.dict("os.environ", {
        "JIRA_BASE_URL": "https://jira.example.com",
        "JIRA_TOKEN": "fake-token",
    })
    @patch("stary.dagster.defs.sensors.JiraAdapter")
    @patch("stary.dagster.defs.sensors.TriggerDetector")
    @patch("stary.dagster.defs.sensors.TicketStateValidator")
    def test_skips_same_retry_count(self, MockValidator, MockDetector, MockJira):
        MockDetector.return_value.poll_comment_triggers.return_value = [
            ("PROJ-1", "https://jira.example.com/browse/PROJ-1", "retry_candidate"),
        ]
        MockJira.return_value.get_comments.return_value = []
        MockValidator.return_value.resolve_trigger.return_value = ("retry", 1, True, "")

        now = datetime.now(timezone.utc).isoformat()

        from stary.dagster.defs.sensors import stary_comment_sensor
        context = build_sensor_context(cursor=json.dumps({"PROJ-1-1": now}))
        result = stary_comment_sensor.evaluate_tick(context)

        assert len(result.run_requests) == 0


# ---------------------------------------------------------------------------
# Cross-trigger dedup (ticket in both do_it and scheduled)
# ---------------------------------------------------------------------------


class TestCrossTriggerDedup:
    """Verify that a ticket handled by comment sensor is skipped by scheduled."""

    @patch.dict("os.environ", {
        "STARY_SCHEDULE_USERS": "alice",
        "JIRA_BASE_URL": "https://jira.example.com",
        "JIRA_TOKEN": "fake-token",
    })
    @patch("stary.dagster.defs.sensors.JiraAdapter")
    @patch("stary.dagster.defs.sensors.TriggerDetector")
    @patch("stary.dagster.defs.sensors.TicketStateValidator")
    def test_scheduled_rejects_ticket_with_stary_markers(self, MockValidator, MockDetector, MockJira):
        """Even if cursor is empty, scheduled sensor checks comments."""
        MockDetector.return_value.poll_scheduled_candidates.return_value = [
            ("PROJ-1", "https://jira.example.com/browse/PROJ-1"),
        ]
        MockJira.return_value.get_comments.return_value = [
            MagicMock(body="[~sys_qaplatformbot] stary:wip"),
            MagicMock(body="[~sys_qaplatformbot] stary:done"),
        ]
        MockValidator.return_value.resolve_scheduled.return_value = False

        from stary.dagster.defs.sensors import stary_users_sensor
        context = build_sensor_context(cursor=None)
        result = stary_users_sensor.evaluate_tick(context)

        assert len(result.run_requests) == 0
