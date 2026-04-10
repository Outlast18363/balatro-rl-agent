from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .runtime import RunPaths
from .runtime import append_event
from .runtime import clear_control
from .runtime import load_control
from .runtime import load_status
from .runtime import write_pid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a managed trainer and supervisor pair")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory under runs/")
    parser.add_argument(
        "--trainer-mode",
        choices=("finish", "fail", "stall"),
        default="finish",
        help="Initial dummy trainer mode",
    )
    parser.add_argument(
        "--restart-trainer-mode",
        choices=("finish", "fail", "stall"),
        default=None,
        help="Dummy trainer mode used after a restart request",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Final target step")
    parser.add_argument("--step-interval", type=float, default=0.2, help="Seconds per dummy step")
    parser.add_argument("--checkpoint-interval", type=int, default=2, help="Checkpoint cadence")
    parser.add_argument("--fail-at", type=int, default=3, help="Failure step for fail mode")
    parser.add_argument("--stall-after", type=int, default=2, help="Step after which stall mode hangs")
    parser.add_argument("--stall-seconds", type=float, default=60.0, help="Simulated hang duration")
    parser.add_argument(
        "--supervisor-poll-interval",
        type=float,
        default=0.25,
        help="Seconds between supervisor health checks",
    )
    parser.add_argument(
        "--launcher-poll-interval",
        type=float,
        default=0.1,
        help="Seconds between launcher control polls",
    )
    parser.add_argument(
        "--stall-timeout",
        type=float,
        default=5.0,
        help="Seconds without progress before supervisor requests restart",
    )
    parser.add_argument("--max-restarts", type=int, default=1, help="How many trainer restarts are allowed")
    parser.add_argument(
        "--webhook-url",
        default=None,
        help="Optional Slack incoming webhook used by trainer and supervisor",
    )
    return parser.parse_args()


def _spawn_supervisor(args: argparse.Namespace) -> subprocess.Popen[str]:
    command = [
        sys.executable,
        "-m",
        "training.supervisor",
        "--run-dir",
        str(args.run_dir),
        "--poll-interval",
        str(args.supervisor_poll_interval),
        "--stall-timeout",
        str(args.stall_timeout),
        "--exit-on-terminal",
    ]
    if args.webhook_url:
        command.extend(["--webhook-url", args.webhook_url])
    return subprocess.Popen(command, text=True)


def _spawn_trainer(args: argparse.Namespace, *, mode: str, session_id: str) -> subprocess.Popen[str]:
    command = [
        sys.executable,
        "-m",
        "training.trainer_main",
        "--run-dir",
        str(args.run_dir),
        "--mode",
        mode,
        "--max-steps",
        str(args.max_steps),
        "--step-interval",
        str(args.step_interval),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--fail-at",
        str(args.fail_at),
        "--stall-after",
        str(args.stall_after),
        "--stall-seconds",
        str(args.stall_seconds),
        "--session-id",
        session_id,
    ]
    if args.webhook_url:
        command.extend(["--webhook-url", args.webhook_url])
    return subprocess.Popen(command, text=True)


def _terminate_process(proc: subprocess.Popen[str], *, name: str, timeout: float = 3.0) -> None:
    if proc.poll() is not None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


def _wait_for_trainer_claim(paths: RunPaths, session_id: str, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = load_status(paths)
        if (
            status is not None
            and status.get("trainer_session_id") == session_id
            and status.get("state") == "running"
        ):
            return
        time.sleep(0.05)


def run_launcher(args: argparse.Namespace) -> int:
    paths = RunPaths.from_path(args.run_dir)
    paths.ensure()
    write_pid(paths.launcher_pid_path)
    clear_control(paths)
    append_event(paths, "launcher_started", trainer_mode=args.trainer_mode, max_restarts=args.max_restarts)

    supervisor = _spawn_supervisor(args)
    trainer_generation = 0
    trainer_mode = args.trainer_mode
    trainer = _spawn_trainer(args, mode=trainer_mode, session_id=f"trainer-{trainer_generation}")
    restarts = 0

    try:
        while True:
            control = load_control(paths)
            if control and control.get("action") == "restart":
                reason = str(control.get("reason", "unknown"))
                if restarts >= args.max_restarts:
                    append_event(paths, "launcher_restart_limit_reached", reason=reason, max_restarts=args.max_restarts)
                    clear_control(paths)
                    _terminate_process(trainer, name="trainer")
                    return 1

                append_event(
                    paths,
                    "launcher_restart_handled",
                    reason=reason,
                    restart_index=restarts + 1,
                    trainer_session_id=control.get("trainer_session_id"),
                )
                _terminate_process(trainer, name="trainer")

                restarts += 1
                trainer_generation += 1
                trainer_mode = args.restart_trainer_mode or trainer_mode
                next_session_id = f"trainer-{trainer_generation}"
                trainer = _spawn_trainer(
                    args,
                    mode=trainer_mode,
                    session_id=next_session_id,
                )
                _wait_for_trainer_claim(paths, next_session_id)
                clear_control(paths)

            trainer_returncode = trainer.poll()
            supervisor_returncode = supervisor.poll()

            if trainer_returncode is not None and supervisor_returncode is not None:
                return 0 if trainer_returncode == 0 and supervisor_returncode == 0 else trainer_returncode or supervisor_returncode

            if supervisor_returncode not in (None, 0) and trainer_returncode is None:
                append_event(paths, "launcher_detected_supervisor_exit", returncode=supervisor_returncode)
                _terminate_process(trainer, name="trainer")
                return supervisor_returncode

            time.sleep(args.launcher_poll_interval)
    finally:
        _terminate_process(trainer, name="trainer")
        _terminate_process(supervisor, name="supervisor")


def main() -> int:
    return run_launcher(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
