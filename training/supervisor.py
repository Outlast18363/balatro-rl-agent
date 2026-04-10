from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .runtime import RunPaths
from .runtime import TERMINAL_STATES
from .runtime import append_event
from .runtime import load_control
from .runtime import load_status
from .runtime import now_utc
from .runtime import parse_timestamp
from .runtime import read_pid
from .runtime import write_control
from .runtime import write_pid
from .slack_notify import send_notification


@dataclass
class HealthSnapshot:
    stalled: bool
    reasons: tuple[str, ...]
    status_age_seconds: float
    step_age_seconds: float
    global_step: int


@dataclass
class ProgressWatchdog:
    stall_timeout: float
    current_session_id: str | None = None
    last_step: int | None = None
    last_step_change_at: datetime | None = None
    last_updated_at: datetime | None = None
    _terminal_notified: set[tuple[str | None, str]] = field(default_factory=set)

    def reset(self, session_id: str | None = None) -> None:
        self.current_session_id = session_id
        self.last_step = None
        self.last_step_change_at = None
        self.last_updated_at = None

    def observe(self, status: dict[str, object], *, now: datetime) -> HealthSnapshot:
        session_id = status.get("trainer_session_id")
        if session_id != self.current_session_id:
            self.reset(session_id=str(session_id) if session_id is not None else None)

        updated_at = parse_timestamp(str(status.get("updated_at"))) or now
        global_step = int(status.get("global_step", 0) or 0)
        state = str(status.get("state", "unknown"))

        if self.last_updated_at is None or updated_at > self.last_updated_at:
            self.last_updated_at = updated_at

        if self.last_step is None or global_step != self.last_step:
            self.last_step = global_step
            self.last_step_change_at = now
        elif self.last_step_change_at is None:
            self.last_step_change_at = now

        if state in TERMINAL_STATES:
            return HealthSnapshot(
                stalled=False,
                reasons=(),
                status_age_seconds=max(0.0, (now - updated_at).total_seconds()),
                step_age_seconds=0.0,
                global_step=global_step,
            )

        status_age = max(0.0, (now - updated_at).total_seconds())
        step_age = max(0.0, (now - (self.last_step_change_at or now)).total_seconds())

        reasons: list[str] = []
        if status_age > self.stall_timeout:
            reasons.append("status_stale")
        if step_age > self.stall_timeout:
            reasons.append("step_stale")

        return HealthSnapshot(
            stalled=bool(reasons),
            reasons=tuple(reasons),
            status_age_seconds=status_age,
            step_age_seconds=step_age,
            global_step=global_step,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor a managed training run")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory under runs/")
    parser.add_argument("--poll-interval", type=float, default=0.25, help="Seconds between polls")
    parser.add_argument(
        "--stall-timeout",
        type=float,
        default=5.0,
        help="Seconds without progress before requesting restart",
    )
    parser.add_argument(
        "--webhook-url",
        default=None,
        help="Optional Slack incoming webhook for stall and terminal notifications",
    )
    parser.add_argument(
        "--exit-on-terminal",
        action="store_true",
        help="Exit once the trainer reaches a terminal state",
    )
    return parser.parse_args()


def is_process_alive(pid: int | None) -> bool:
    if pid in (None, 0):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _request_restart(
    paths: RunPaths,
    *,
    reason: str,
    trainer_session_id: str | None,
    global_step: int,
    webhook_url: str | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    if load_control(paths) is not None:
        return

    payload = {
        "trainer_session_id": trainer_session_id,
        "global_step": global_step,
    }
    if extra:
        payload.update(extra)

    write_control(paths, "restart", reason, **payload)
    append_event(paths, "restart_requested", reason=reason, **payload)
    send_notification(
        paths,
        text=f"training restart requested: {reason} at step {global_step}",
        trainer_session_id=trainer_session_id,
        global_step=global_step,
        webhook_url=webhook_url,
    )


def run_supervisor(args: argparse.Namespace) -> int:
    paths = RunPaths.from_path(args.run_dir)
    paths.ensure()
    write_pid(paths.supervisor_pid_path)

    watchdog = ProgressWatchdog(stall_timeout=args.stall_timeout)
    append_event(
        paths,
        "supervisor_started",
        poll_interval=args.poll_interval,
        stall_timeout=args.stall_timeout,
    )

    while True:
        status = load_status(paths)
        trainer_pid = read_pid(paths.trainer_pid_path)
        trainer_alive = is_process_alive(trainer_pid)
        now = now_utc()
        control = load_control(paths)

        if status is not None:
            snapshot = watchdog.observe(status, now=now)
            state = str(status.get("state", "unknown"))
            trainer_session_id = status.get("trainer_session_id")
            global_step = int(status.get("global_step", 0) or 0)

            if state in TERMINAL_STATES:
                if control and control.get("action") == "restart":
                    time.sleep(args.poll_interval)
                    continue

                notify_key = (str(trainer_session_id) if trainer_session_id is not None else None, state)
                if notify_key not in watchdog._terminal_notified:
                    append_event(
                        paths,
                        "supervisor_terminal_state_seen",
                        trainer_session_id=trainer_session_id,
                        state=state,
                        global_step=global_step,
                    )
                    watchdog._terminal_notified.add(notify_key)

                if args.exit_on_terminal:
                    return 0
            elif snapshot.stalled and trainer_alive:
                _request_restart(
                    paths,
                    reason="stalled",
                    trainer_session_id=str(trainer_session_id) if trainer_session_id is not None else None,
                    global_step=global_step,
                    webhook_url=args.webhook_url,
                    extra={
                        "status_age_seconds": snapshot.status_age_seconds,
                        "step_age_seconds": snapshot.step_age_seconds,
                        "reasons": list(snapshot.reasons),
                    },
                )
            elif trainer_pid is not None and not trainer_alive:
                _request_restart(
                    paths,
                    reason="trainer_process_missing",
                    trainer_session_id=str(trainer_session_id) if trainer_session_id is not None else None,
                    global_step=global_step,
                    webhook_url=args.webhook_url,
                )

        time.sleep(args.poll_interval)


def main() -> int:
    return run_supervisor(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
