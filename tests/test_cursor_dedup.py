"""Tests for cursor-based deduplication in Dagster sensors/schedules.

Verifies that _load_cursor / _save_cursor helpers and sensor-level
filtering prevent duplicate ticket processing across evaluation cycles.
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
# Sensor-level deduplication (scheduled trigger)
# ---------------------------------------------------------------------------


class TestScheduledTriggerCursor:
    """Verify jira_users_trigger skips already-processed tickets."""

    @patch.dict("os.environ", {
        "STARY_SCHEDULE_USERS": "alice",
        "JIRA_BASE_URL": "https://jira.example.com",
        "JIRA_TOKEN": "fake-token",
    })
    @patch("stary.dagster.defs.sensors.JiraAdapter")
    @patch("stary.dagster.defs.sensors.TriggerDetector")
    def test_skips_ticket_already_in_cursor(self, MockDetector, MockJira):
        from stary.sensor import TriggeredTicket

        ticket = TriggeredTicket(key="PROJ-1", url="https://jira.example.com/browse/PROJ-1", auto_merge=False)
        MockDetector.return_value.poll_scheduled.return_value = [ticket]

        now = datetime.now(timezone.utc).isoformat()
        cursor = json.dumps({"PROJ-1": now})

        from stary.dagster.defs.sensors import jira_users_trigger
        context = build_sensor_context(cursor=cursor)
        result = jira_users_trigger.evaluate_tick(context)

        assert len(result.run_requests) == 0
        assert context.cursor is not None

    @patch.dict("os.environ", {
        "STARY_SCHEDULE_USERS": "alice",
        "JIRA_BASE_URL": "https://jira.example.com",
        "JIRA_TOKEN": "fake-token",
    })
    @patch("stary.dagster.defs.sensors.JiraAdapter")
    @patch("stary.dagster.defs.sensors.TriggerDetector")
    def test_processes_ticket_not_in_cursor(self, MockDetector, MockJira):
        from stary.sensor import TriggeredTicket

        ticket = TriggeredTicket(key="PROJ-2", url="https://jira.example.com/browse/PROJ-2", auto_merge=False)
        MockDetector.return_value.poll_scheduled.return_value = [ticket]

        from stary.dagster.defs.sensors import jira_users_trigger
        context = build_sensor_context(cursor=None)
        result = jira_users_trigger.evaluate_tick(context)

        assert len(result.run_requests) == 1
        assert result.run_requests[0].tags["ticket_key"] == "PROJ-2"
        saved = json.loads(context.cursor)
        assert "PROJ-2" in saved


# ---------------------------------------------------------------------------
# Sensor-level deduplication (do_it sensor)
# ---------------------------------------------------------------------------


class TestDoItSensorCursor:
    @patch.dict("os.environ", {
        "JIRA_BASE_URL": "https://jira.example.com",
        "JIRA_TOKEN": "fake-token",
    })
    @patch("stary.dagster.defs.sensors.JiraAdapter")
    @patch("stary.dagster.defs.sensors.TriggerDetector")
    def test_skips_ticket_in_cursor(self, MockDetector, MockJira):
        from stary.sensor import TriggeredTicket

        ticket = TriggeredTicket(key="PROJ-1", url="https://jira.example.com/browse/PROJ-1", auto_merge=True)
        MockDetector.return_value.poll_do_it.return_value = [ticket]

        now = datetime.now(timezone.utc).isoformat()

        from stary.dagster.defs.sensors import jira_do_it_sensor
        context = build_sensor_context(cursor=json.dumps({"PROJ-1": now}))
        result = jira_do_it_sensor.evaluate_tick(context)

        assert len(result.run_requests) == 0

    @patch.dict("os.environ", {
        "JIRA_BASE_URL": "https://jira.example.com",
        "JIRA_TOKEN": "fake-token",
    })
    @patch("stary.dagster.defs.sensors.JiraAdapter")
    @patch("stary.dagster.defs.sensors.TriggerDetector")
    def test_processes_new_ticket(self, MockDetector, MockJira):
        from stary.sensor import TriggeredTicket

        ticket = TriggeredTicket(key="PROJ-3", url="https://jira.example.com/browse/PROJ-3", auto_merge=True)
        MockDetector.return_value.poll_do_it.return_value = [ticket]

        from stary.dagster.defs.sensors import jira_do_it_sensor
        context = build_sensor_context(cursor=None)
        result = jira_do_it_sensor.evaluate_tick(context)

        assert len(result.run_requests) == 1
        saved = json.loads(context.cursor)
        assert "PROJ-3" in saved


# ---------------------------------------------------------------------------
# Retry sensor: allows same ticket with different retry_count
# ---------------------------------------------------------------------------


class TestRetrySensorCursor:
    @patch.dict("os.environ", {
        "JIRA_BASE_URL": "https://jira.example.com",
        "JIRA_TOKEN": "fake-token",
    })
    @patch("stary.dagster.defs.sensors.JiraAdapter")
    @patch("stary.dagster.defs.sensors.TriggerDetector")
    def test_allows_same_ticket_new_retry_count(self, MockDetector, MockJira):
        from stary.sensor import TriggeredTicket

        # Ticket was already retried at count 1
        ticket = TriggeredTicket(key="PROJ-1", url="https://jira.example.com/browse/PROJ-1", auto_merge=True, retry_count=2)
        MockDetector.return_value.poll_retry.return_value = [ticket]

        now = datetime.now(timezone.utc).isoformat()

        from stary.dagster.defs.sensors import jira_retry_sensor
        context = build_sensor_context(cursor=json.dumps({"PROJ-1-1": now}))
        result = jira_retry_sensor.evaluate_tick(context)

        assert len(result.run_requests) == 1  # retry_count=2 is new, should process
        saved = json.loads(context.cursor)
        assert "PROJ-1-2" in saved

    @patch.dict("os.environ", {
        "JIRA_BASE_URL": "https://jira.example.com",
        "JIRA_TOKEN": "fake-token",
    })
    @patch("stary.dagster.defs.sensors.JiraAdapter")
    @patch("stary.dagster.defs.sensors.TriggerDetector")
    def test_skips_same_retry_count(self, MockDetector, MockJira):
        from stary.sensor import TriggeredTicket

        ticket = TriggeredTicket(key="PROJ-1", url="https://jira.example.com/browse/PROJ-1", auto_merge=True, retry_count=1)
        MockDetector.return_value.poll_retry.return_value = [ticket]

        now = datetime.now(timezone.utc).isoformat()

        from stary.dagster.defs.sensors import jira_retry_sensor
        context = build_sensor_context(cursor=json.dumps({"PROJ-1-1": now}))
        result = jira_retry_sensor.evaluate_tick(context)

        assert len(result.run_requests) == 0
