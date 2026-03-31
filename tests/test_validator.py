"""Tests for PostMergeValidator."""

from unittest.mock import MagicMock

from stary.agents.validator import (
    PostMergeValidator,
    ValidationConfig,
    ValidationResult,
)


class TestPostMergeValidator:
    def test_disabled_returns_skipped(self):
        config = ValidationConfig(enabled=False)
        validator = PostMergeValidator(config=config)
        result = validator.run("owner", "repo", "abc123")
        assert result.passed is True
        assert result.skipped is True

    def test_no_check_runs_passes(self):
        github = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"check_runs": []}
        github._get.return_value = resp
        config = ValidationConfig(enabled=True)
        validator = PostMergeValidator(github=github, config=config)
        result = validator.run("owner", "repo", "abc123")
        assert result.passed is True

    def test_all_checks_pass(self):
        github = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {
            "check_runs": [
                {
                    "name": "test",
                    "status": "completed",
                    "conclusion": "success",
                },
                {
                    "name": "lint",
                    "status": "completed",
                    "conclusion": "success",
                },
            ]
        }
        github._get.return_value = resp
        config = ValidationConfig(enabled=True)
        validator = PostMergeValidator(github=github, config=config)
        result = validator.run("owner", "repo", "abc123")
        assert result.passed is True
        assert result.checks_passed == 2

    def test_failed_checks(self):
        github = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {
            "check_runs": [
                {
                    "name": "test",
                    "status": "completed",
                    "conclusion": "failure",
                },
                {
                    "name": "lint",
                    "status": "completed",
                    "conclusion": "success",
                },
            ]
        }
        github._get.return_value = resp
        config = ValidationConfig(enabled=True)
        validator = PostMergeValidator(github=github, config=config)
        result = validator.run("owner", "repo", "abc123")
        assert result.passed is False
        assert result.checks_failed == 1
        assert "test" in result.details

    def test_validation_result_to_dict(self):
        r = ValidationResult(passed=True, checks_found=2)
        d = r.to_dict()
        assert d["passed"] is True
        assert d["checks_found"] == 2
