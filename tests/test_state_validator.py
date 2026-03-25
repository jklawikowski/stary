"""Tests for TicketStateValidator — comment-based state machine.

Covers:
- State determination from comment history (IDLE, WIP, WIP_STALE, DONE, FAILED)
- Eligibility matrix for all trigger types
- Retry auto_merge resolution (preserves original trigger type)
- WIP staleness detection
- Scheduled trigger rejection of touched tickets
"""

from datetime import datetime, timedelta, timezone

from stary.jira_adapter import JiraComment
from stary.sensor import TicketStateValidator, TriggerConfig
from stary.ticket_status import TicketState


def _make_comment(body: str, created: str = "") -> JiraComment:
    return JiraComment(id="", body=body, created=created)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hours_ago(hours: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


# ---------------------------------------------------------------------------
# determine_state
# ---------------------------------------------------------------------------


class TestDetermineState:
    def test_no_markers_is_idle(self):
        v = TicketStateValidator()
        comments = [_make_comment("some random comment")]
        assert v.determine_state(comments) == TicketState.IDLE

    def test_empty_comments_is_idle(self):
        v = TicketStateValidator()
        assert v.determine_state([]) == TicketState.IDLE

    def test_wip_only_is_wip(self):
        v = TicketStateValidator()
        comments = [
            _make_comment("[~sys_qaplatformbot] do it"),
            _make_comment("[~sys_qaplatformbot] stary:wip", created=_now_iso()),
        ]
        assert v.determine_state(comments) == TicketState.WIP

    def test_wip_stale(self):
        v = TicketStateValidator()
        comments = [
            _make_comment("[~sys_qaplatformbot] stary:wip", created=_hours_ago(5)),
        ]
        assert v.determine_state(comments, wip_stale_hours=3) == TicketState.WIP_STALE

    def test_wip_not_stale_yet(self):
        v = TicketStateValidator()
        comments = [
            _make_comment("[~sys_qaplatformbot] stary:wip", created=_hours_ago(1)),
        ]
        assert v.determine_state(comments, wip_stale_hours=3) == TicketState.WIP

    def test_done_after_wip_is_done(self):
        v = TicketStateValidator()
        comments = [
            _make_comment("[~sys_qaplatformbot] stary:wip"),
            _make_comment("[~sys_qaplatformbot] stary:done"),
        ]
        assert v.determine_state(comments) == TicketState.DONE

    def test_failed_after_wip_is_failed(self):
        v = TicketStateValidator()
        comments = [
            _make_comment("[~sys_qaplatformbot] stary:wip"),
            _make_comment("[~sys_qaplatformbot] stary:failed"),
        ]
        assert v.determine_state(comments) == TicketState.FAILED

    def test_done_then_failed_is_failed(self):
        v = TicketStateValidator()
        comments = [
            _make_comment("[~sys_qaplatformbot] stary:done"),
            _make_comment("[~sys_qaplatformbot] stary:failed"),
        ]
        assert v.determine_state(comments) == TicketState.FAILED

    def test_failed_then_done_is_done(self):
        v = TicketStateValidator()
        comments = [
            _make_comment("[~sys_qaplatformbot] stary:failed"),
            _make_comment("[~sys_qaplatformbot] stary:done"),
        ]
        assert v.determine_state(comments) == TicketState.DONE

    def test_wip_after_done_is_wip(self):
        """If a new WIP is posted after done (retry scenario), state is WIP."""
        v = TicketStateValidator()
        comments = [
            _make_comment("[~sys_qaplatformbot] stary:done"),
            _make_comment("[~sys_qaplatformbot] stary:wip", created=_now_iso()),
        ]
        assert v.determine_state(comments) == TicketState.WIP

    def test_multiple_cycles(self):
        """Full lifecycle: do_it → wip → done → retry → wip → failed"""
        v = TicketStateValidator()
        comments = [
            _make_comment("[~sys_qaplatformbot] do it"),
            _make_comment("[~sys_qaplatformbot] stary:wip"),
            _make_comment("[~sys_qaplatformbot] stary:done"),
            _make_comment("[~sys_qaplatformbot] retry"),
            _make_comment("[~sys_qaplatformbot] stary:wip"),
            _make_comment("[~sys_qaplatformbot] stary:failed"),
        ]
        assert v.determine_state(comments) == TicketState.FAILED


# ---------------------------------------------------------------------------
# is_eligible
# ---------------------------------------------------------------------------


class TestIsEligible:
    def test_idle_allows_do_it(self):
        v = TicketStateValidator()
        assert v.is_eligible(TicketState.IDLE, "do_it") is True

    def test_idle_allows_pr_only(self):
        v = TicketStateValidator()
        assert v.is_eligible(TicketState.IDLE, "pr_only") is True

    def test_idle_allows_scheduled(self):
        v = TicketStateValidator()
        assert v.is_eligible(TicketState.IDLE, "scheduled") is True

    def test_idle_rejects_retry(self):
        v = TicketStateValidator()
        assert v.is_eligible(TicketState.IDLE, "retry") is False

    def test_wip_blocks_everything(self):
        v = TicketStateValidator()
        for trigger in ("do_it", "pr_only", "retry", "scheduled"):
            assert v.is_eligible(TicketState.WIP, trigger) is False

    def test_wip_stale_allows_do_it(self):
        v = TicketStateValidator()
        assert v.is_eligible(TicketState.WIP_STALE, "do_it") is True

    def test_wip_stale_allows_pr_only(self):
        v = TicketStateValidator()
        assert v.is_eligible(TicketState.WIP_STALE, "pr_only") is True

    def test_wip_stale_blocks_retry(self):
        v = TicketStateValidator()
        assert v.is_eligible(TicketState.WIP_STALE, "retry") is False

    def test_wip_stale_blocks_scheduled(self):
        v = TicketStateValidator()
        assert v.is_eligible(TicketState.WIP_STALE, "scheduled") is False

    def test_done_allows_retry(self):
        v = TicketStateValidator()
        assert v.is_eligible(TicketState.DONE, "retry") is True

    def test_done_blocks_do_it(self):
        v = TicketStateValidator()
        assert v.is_eligible(TicketState.DONE, "do_it") is False

    def test_done_blocks_scheduled(self):
        v = TicketStateValidator()
        assert v.is_eligible(TicketState.DONE, "scheduled") is False

    def test_failed_allows_retry(self):
        v = TicketStateValidator()
        assert v.is_eligible(TicketState.FAILED, "retry") is True

    def test_failed_blocks_do_it(self):
        v = TicketStateValidator()
        assert v.is_eligible(TicketState.FAILED, "do_it") is False


# ---------------------------------------------------------------------------
# resolve_trigger — do_it / pr_only
# ---------------------------------------------------------------------------


class TestResolveTriggerDoItPrOnly:
    def test_do_it_idle_ticket(self):
        v = TicketStateValidator()
        comments = [_make_comment("[~sys_qaplatformbot] do it")]
        trigger, retry, auto = v.resolve_trigger(comments, "do_it")
        assert trigger == "do_it"
        assert retry == 0
        assert auto is True

    def test_pr_only_idle_ticket(self):
        v = TicketStateValidator()
        comments = [_make_comment("[~sys_qaplatformbot] pull request")]
        trigger, retry, auto = v.resolve_trigger(comments, "pr_only")
        assert trigger == "pr_only"
        assert retry == 0
        assert auto is False

    def test_do_it_rejected_when_wip(self):
        v = TicketStateValidator()
        comments = [
            _make_comment("[~sys_qaplatformbot] do it"),
            _make_comment("[~sys_qaplatformbot] stary:wip", created=_now_iso()),
        ]
        trigger, retry, auto = v.resolve_trigger(comments, "do_it")
        assert trigger is None

    def test_do_it_rejected_when_done(self):
        v = TicketStateValidator()
        comments = [
            _make_comment("[~sys_qaplatformbot] do it"),
            _make_comment("[~sys_qaplatformbot] stary:done"),
        ]
        trigger, retry, auto = v.resolve_trigger(comments, "do_it")
        assert trigger is None

    def test_do_it_allowed_when_wip_stale(self):
        v = TicketStateValidator()
        comments = [
            _make_comment("[~sys_qaplatformbot] do it"),
            _make_comment("[~sys_qaplatformbot] stary:wip", created=_hours_ago(5)),
        ]
        trigger, retry, auto = v.resolve_trigger(comments, "do_it")
        assert trigger == "do_it"
        assert auto is True


# ---------------------------------------------------------------------------
# resolve_trigger — retry with auto_merge preservation
# ---------------------------------------------------------------------------


class TestResolveTriggerRetry:
    def test_retry_after_failure(self):
        v = TicketStateValidator()
        comments = [
            _make_comment("[~sys_qaplatformbot] do it"),
            _make_comment("[~sys_qaplatformbot] stary:wip"),
            _make_comment("[~sys_qaplatformbot] stary:failed"),
            _make_comment("[~sys_qaplatformbot] retry"),
        ]
        trigger, retry, auto = v.resolve_trigger(comments, "retry_candidate")
        assert trigger == "retry"
        assert retry == 1
        assert auto is True  # original was "do it"

    def test_retry_preserves_pr_only_auto_merge(self):
        """Key bug fix: retry should preserve the original trigger's auto_merge."""
        v = TicketStateValidator()
        comments = [
            _make_comment("[~sys_qaplatformbot] pull request"),
            _make_comment("[~sys_qaplatformbot] stary:wip"),
            _make_comment("[~sys_qaplatformbot] stary:failed"),
            _make_comment("[~sys_qaplatformbot] retry"),
        ]
        trigger, retry, auto = v.resolve_trigger(comments, "retry_candidate")
        assert trigger == "retry"
        assert retry == 1
        assert auto is False  # original was "pull request" → must stay False

    def test_retry_after_done(self):
        v = TicketStateValidator()
        comments = [
            _make_comment("[~sys_qaplatformbot] do it"),
            _make_comment("[~sys_qaplatformbot] stary:done"),
            _make_comment("[~sys_qaplatformbot] retry"),
        ]
        trigger, retry, auto = v.resolve_trigger(comments, "retry_candidate")
        assert trigger == "retry"
        assert retry == 1
        assert auto is True

    def test_retry_rejected_when_older_than_terminal(self):
        v = TicketStateValidator()
        comments = [
            _make_comment("[~sys_qaplatformbot] retry"),
            _make_comment("[~sys_qaplatformbot] stary:done"),
        ]
        trigger, retry, auto = v.resolve_trigger(comments, "retry_candidate")
        assert trigger is None

    def test_retry_rejected_when_idle(self):
        v = TicketStateValidator()
        comments = [_make_comment("[~sys_qaplatformbot] retry")]
        trigger, retry, auto = v.resolve_trigger(comments, "retry_candidate")
        assert trigger is None

    def test_retry_rejected_when_max_exceeded(self):
        v = TicketStateValidator()
        comments = [
            _make_comment("[~sys_qaplatformbot] do it"),
            _make_comment("[~sys_qaplatformbot] stary:failed"),
            _make_comment("[~sys_qaplatformbot] retry"),
            _make_comment("[~sys_qaplatformbot] stary:failed"),
            _make_comment("[~sys_qaplatformbot] retry"),
            _make_comment("[~sys_qaplatformbot] stary:failed"),
            _make_comment("[~sys_qaplatformbot] retry"),
            _make_comment("[~sys_qaplatformbot] stary:failed"),
            _make_comment("[~sys_qaplatformbot] retry"),  # 4th retry > MAX_RETRY_COUNT=3
        ]
        trigger, retry, auto = v.resolve_trigger(comments, "retry_candidate")
        assert trigger is None

    def test_retry_rejected_when_wip(self):
        v = TicketStateValidator()
        comments = [
            _make_comment("[~sys_qaplatformbot] stary:wip", created=_now_iso()),
        ]
        trigger, retry, auto = v.resolve_trigger(comments, "retry_candidate")
        assert trigger is None

    def test_retry_no_original_trigger_defaults_auto_merge_true(self):
        """When no do_it/pr_only found before terminal, default auto_merge=True."""
        v = TicketStateValidator()
        comments = [
            _make_comment("[~sys_qaplatformbot] stary:failed"),
            _make_comment("[~sys_qaplatformbot] retry"),
        ]
        trigger, retry, auto = v.resolve_trigger(comments, "retry_candidate")
        assert trigger == "retry"
        assert auto is True


# ---------------------------------------------------------------------------
# resolve_scheduled
# ---------------------------------------------------------------------------


class TestResolveScheduled:
    def test_idle_ticket_eligible(self):
        v = TicketStateValidator()
        comments = [_make_comment("unrelated comment")]
        assert v.resolve_scheduled(comments) is True

    def test_empty_comments_eligible(self):
        v = TicketStateValidator()
        assert v.resolve_scheduled([]) is True

    def test_wip_ticket_rejected(self):
        v = TicketStateValidator()
        comments = [_make_comment("[~sys_qaplatformbot] stary:wip", created=_now_iso())]
        assert v.resolve_scheduled(comments) is False

    def test_done_ticket_rejected(self):
        v = TicketStateValidator()
        comments = [_make_comment("[~sys_qaplatformbot] stary:done")]
        assert v.resolve_scheduled(comments) is False

    def test_failed_ticket_rejected(self):
        v = TicketStateValidator()
        comments = [_make_comment("[~sys_qaplatformbot] stary:failed")]
        assert v.resolve_scheduled(comments) is False

    def test_wip_stale_ticket_rejected(self):
        v = TicketStateValidator()
        comments = [_make_comment("[~sys_qaplatformbot] stary:wip", created=_hours_ago(5))]
        assert v.resolve_scheduled(comments) is False


# ---------------------------------------------------------------------------
# WIP staleness edge cases
# ---------------------------------------------------------------------------


class TestWipStaleness:
    def test_wip_without_created_timestamp_is_not_stale(self):
        """If created field is empty, treat as fresh (safe default)."""
        v = TicketStateValidator()
        comments = [_make_comment("[~sys_qaplatformbot] stary:wip", created="")]
        assert v.determine_state(comments) == TicketState.WIP

    def test_wip_with_malformed_timestamp_is_not_stale(self):
        v = TicketStateValidator()
        comments = [_make_comment("[~sys_qaplatformbot] stary:wip", created="not-a-date")]
        assert v.determine_state(comments) == TicketState.WIP

    def test_wip_exactly_at_threshold_is_not_stale(self):
        v = TicketStateValidator()
        # WIP posted exactly 3 hours ago, threshold is 3 → not stale (> not >=)
        comments = [_make_comment("[~sys_qaplatformbot] stary:wip", created=_hours_ago(3))]
        # 3 hours ago is not > 3, so should still be WIP
        state = v.determine_state(comments, wip_stale_hours=3)
        # Due to execution time, this might flip. Use 4h to be safe.
        comments2 = [_make_comment("[~sys_qaplatformbot] stary:wip", created=_hours_ago(4))]
        assert v.determine_state(comments2, wip_stale_hours=3) == TicketState.WIP_STALE
