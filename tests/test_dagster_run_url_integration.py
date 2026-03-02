"""End-to-end tests verifying the Dagster run URL flows correctly from
the ops layer through to the Jira WIP comment."""

import os
from unittest import mock
from unittest.mock import MagicMock, patch

from stary.config import build_dagster_run_url, get_dagster_base_url


class TestDagsterRunUrlFlow:
    """Verify the full construction path: env var → base URL → run URL → comment."""

    def test_full_flow_with_env_var(self):
        with mock.patch.dict(os.environ, {"DAGSTER_BASE_URL": "https://dagster.corp.com/"}):
            base = get_dagster_base_url()
            assert base == "https://dagster.corp.com"
            url = build_dagster_run_url(base, "run-xyz-789")
            assert url == "https://dagster.corp.com/runs/run-xyz-789"

    def test_full_flow_without_env_var(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            base = get_dagster_base_url()
            assert base is None
            url = build_dagster_run_url(base, "run-xyz-789")
            assert url is None

    @patch("stary.sensor.requests.post")
    def test_wip_comment_includes_url_when_configured(self, mock_post: MagicMock):
        """Simulate the mark_ticket_wip op logic."""
        mock_post.return_value = MagicMock(status_code=201)
        mock_post.return_value.raise_for_status = MagicMock()

        from stary.sensor import Sensor

        with mock.patch.dict(os.environ, {"DAGSTER_BASE_URL": "https://dagster.corp.com"}):
            dagster_base_url = get_dagster_base_url()
            run_id = "abc-def-123"
            dagster_run_url = build_dagster_run_url(dagster_base_url, run_id)

            s = Sensor(jira_base_url="https://jira.example.com", jira_token="tok")
            s.mark_as_wip("TEST-1", dagster_run_url=dagster_run_url)

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "https://dagster.corp.com/runs/abc-def-123" in body["body"]

    @patch("stary.sensor.requests.post")
    def test_wip_comment_omits_url_when_not_configured(self, mock_post: MagicMock):
        mock_post.return_value = MagicMock(status_code=201)
        mock_post.return_value.raise_for_status = MagicMock()

        from stary.sensor import Sensor

        with mock.patch.dict(os.environ, {}, clear=True):
            dagster_base_url = get_dagster_base_url()
            dagster_run_url = build_dagster_run_url(dagster_base_url, "some-run")
            assert dagster_run_url is None

            s = Sensor(jira_base_url="https://jira.example.com", jira_token="tok")
            s.mark_as_wip("TEST-2", dagster_run_url=dagster_run_url)

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "View live pipeline status" not in body["body"]
