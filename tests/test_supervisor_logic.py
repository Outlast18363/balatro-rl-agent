from datetime import datetime
from datetime import timedelta
from datetime import timezone

from training.runtime import isoformat
from training.supervisor import ProgressWatchdog


def test_watchdog_detects_step_stall_even_with_recent_status_updates():
    watchdog = ProgressWatchdog(stall_timeout=5.0)
    started = datetime(2026, 4, 9, 12, 0, tzinfo=timezone.utc)

    first = watchdog.observe(
        {
            "trainer_session_id": "trainer-0",
            "state": "running",
            "global_step": 3,
            "updated_at": isoformat(started),
        },
        now=started,
    )
    second = watchdog.observe(
        {
            "trainer_session_id": "trainer-0",
            "state": "running",
            "global_step": 3,
            "updated_at": isoformat(started + timedelta(seconds=4)),
        },
        now=started + timedelta(seconds=4),
    )
    third = watchdog.observe(
        {
            "trainer_session_id": "trainer-0",
            "state": "running",
            "global_step": 3,
            "updated_at": isoformat(started + timedelta(seconds=6)),
        },
        now=started + timedelta(seconds=6),
    )

    assert first.stalled is False
    assert second.stalled is False
    assert third.stalled is True
    assert "step_stale" in third.reasons


def test_watchdog_resets_when_trainer_session_changes():
    watchdog = ProgressWatchdog(stall_timeout=5.0)
    started = datetime(2026, 4, 9, 12, 0, tzinfo=timezone.utc)

    watchdog.observe(
        {
            "trainer_session_id": "trainer-0",
            "state": "running",
            "global_step": 2,
            "updated_at": isoformat(started),
        },
        now=started,
    )
    snapshot = watchdog.observe(
        {
            "trainer_session_id": "trainer-1",
            "state": "running",
            "global_step": 2,
            "updated_at": isoformat(started + timedelta(seconds=1)),
        },
        now=started + timedelta(seconds=1),
    )

    assert snapshot.stalled is False
    assert watchdog.current_session_id == "trainer-1"
