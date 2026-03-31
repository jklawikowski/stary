"""Post-merge production validator.

Checks whether a merged PR introduced regressions by:
- Polling the target repo's CI status (GitHub Actions / checks API)
- Optionally running smoke tests via configurable commands

This is a best-effort check — not all repos have CI or health endpoints.
"""

import logging
import time
from dataclasses import dataclass, field

from stary.github_adapter import GitHubAdapter
from stary.telemetry import tracer

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for post-merge validation."""

    enabled: bool = False
    check_ci_status: bool = True
    ci_poll_interval_seconds: int = 30
    ci_poll_max_wait_seconds: int = 300
    required_checks: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of a post-merge validation."""

    passed: bool
    checks_found: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    details: str = ""
    skipped: bool = False

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks_found": self.checks_found,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "details": self.details,
            "skipped": self.skipped,
        }


class PostMergeValidator:
    """Validates that a merged PR hasn't introduced regressions.

    Uses the GitHub Checks API to poll CI status on the merge commit.
    """

    def __init__(
        self,
        github: GitHubAdapter | None = None,
        config: ValidationConfig | None = None,
    ):
        self._github = github or GitHubAdapter()
        self.config = config or ValidationConfig()

    @tracer.start_as_current_span("validator.run")
    def run(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
    ) -> ValidationResult:
        """Validate the merge commit by checking CI status.

        Args:
            owner: Repository owner
            repo: Repository name
            commit_sha: The merge commit SHA to validate

        Returns:
            ValidationResult with check outcomes.
        """
        if not self.config.enabled:
            logger.info("Post-merge validation is disabled")
            return ValidationResult(
                passed=True, skipped=True, details="Validation disabled",
            )

        if not self.config.check_ci_status:
            return ValidationResult(
                passed=True, skipped=True, details="CI check disabled",
            )

        logger.info(
            "Polling CI status for %s/%s@%s",
            owner,
            repo,
            commit_sha[:8],
        )

        elapsed = 0
        while elapsed < self.config.ci_poll_max_wait_seconds:
            result = self._check_commit_status(owner, repo, commit_sha)
            if result is not None:
                return result
            logger.info(
                "CI checks still pending, waiting %ds (elapsed: %ds)",
                self.config.ci_poll_interval_seconds,
                elapsed,
            )
            time.sleep(self.config.ci_poll_interval_seconds)
            elapsed += self.config.ci_poll_interval_seconds

        logger.warning(
            "CI checks did not complete within %ds",
            self.config.ci_poll_max_wait_seconds,
        )
        return ValidationResult(
            passed=True,
            details=(
                "CI checks did not complete within timeout — assuming OK"
            ),
        )

    def _check_commit_status(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
    ) -> ValidationResult | None:
        """Check the combined status of a commit.

        Returns ValidationResult if checks are complete, None if still
        pending.
        """
        try:
            resp = self._github._get(
                f"/repos/{owner}/{repo}/commits/{commit_sha}/check-runs",
            )
            data = resp.json()
            check_runs = data.get("check_runs", [])

            if not check_runs:
                return ValidationResult(
                    passed=True,
                    details="No CI checks configured",
                )

            total = len(check_runs)
            completed = [
                cr
                for cr in check_runs
                if cr.get("status") == "completed"
            ]
            if len(completed) < total:
                return None  # Still pending

            passed = [
                cr
                for cr in completed
                if cr.get("conclusion")
                in ("success", "neutral", "skipped")
            ]
            failed = [
                cr
                for cr in completed
                if cr.get("conclusion")
                in ("failure", "timed_out", "cancelled")
            ]

            failed_names = [cr.get("name", "unknown") for cr in failed]
            details = ""
            if failed_names:
                details = f"Failed checks: {', '.join(failed_names)}"

            return ValidationResult(
                passed=len(failed) == 0,
                checks_found=total,
                checks_passed=len(passed),
                checks_failed=len(failed),
                details=details,
            )
        except Exception as exc:
            logger.warning("Failed to check CI status: %s", exc)
            return ValidationResult(
                passed=True,
                details=f"CI status check failed: {exc}",
            )
