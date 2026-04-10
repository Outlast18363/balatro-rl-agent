from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

from .runtime import RunPaths
from .runtime import append_event
from .runtime import create_checkpoint
from .runtime import isoformat
from .runtime import latest_checkpoint
from .runtime import load_control
from .runtime import replace_status
from .runtime import write_pid
from .runtime import write_status
from .slack_notify import send_notification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal managed trainer harness")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory under runs/")
    parser.add_argument(
        "--mode",
        choices=("finish", "fail", "stall"),
        default="finish",
        help="How the dummy trainer should behave",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Final target step")
    parser.add_argument("--step-interval", type=float, default=0.2, help="Seconds per dummy step")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=2,
        help="Emit a checkpoint marker every N steps",
    )
    parser.add_argument("--fail-at", type=int, default=3, help="Step at which fail mode raises")
    parser.add_argument(
        "--stall-after",
        type=int,
        default=2,
        help="Step after which stall mode stops reporting progress",
    )
    parser.add_argument(
        "--stall-seconds",
        type=float,
        default=60.0,
        help="How long the simulated hang lasts before failing if not restarted",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Logical trainer generation id used to reject stale writes after restart",
    )
    parser.add_argument(
        "--webhook-url",
        default=None,
        help="Optional Slack incoming webhook for terminal notifications",
    )
    return parser.parse_args()


def _install_signal_handlers() -> None:
    def _raise_interrupt(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _raise_interrupt)
    signal.signal(signal.SIGINT, _raise_interrupt)


def run_trainer(args: argparse.Namespace) -> int:
    _install_signal_handlers()

    paths = RunPaths.from_path(args.run_dir)
    paths.ensure()
    write_pid(paths.trainer_pid_path)

    session_id = args.session_id or f"trainer-{int(time.time() * 1000)}"
    checkpoint = latest_checkpoint(paths)
    current_step = int(checkpoint["step"]) if checkpoint else 0
    last_checkpoint = str(checkpoint["path"]) if checkpoint else None
    last_checkpoint_step = int(checkpoint["step"]) if checkpoint else None

    replace_status(
        paths,
        {
            "state": "running",
            "pid": os.getpid(),
            "trainer_session_id": session_id,
            "started_at": isoformat(),
            "global_step": current_step,
            "last_checkpoint": last_checkpoint,
            "last_checkpoint_step": last_checkpoint_step,
            "resume_from": last_checkpoint,
            "message": "trainer bootstrapped",
            "error": None,
        },
    )
    append_event(
        paths,
        "trainer_started",
        trainer_session_id=session_id,
        mode=args.mode,
        resumed_from_step=current_step,
    )

    try:
        while current_step < args.max_steps:
            time.sleep(args.step_interval)
            current_step += 1

            if args.checkpoint_interval > 0 and current_step % args.checkpoint_interval == 0:
                checkpoint_path = create_checkpoint(
                    paths,
                    current_step,
                    payload={"trainer_session_id": session_id},
                )
                last_checkpoint = str(checkpoint_path)
                last_checkpoint_step = current_step
                append_event(
                    paths,
                    "checkpoint_saved",
                    trainer_session_id=session_id,
                    global_step=current_step,
                    checkpoint=last_checkpoint,
                )

            write_status(
                paths,
                owner_id=session_id,
                state="running",
                pid=os.getpid(),
                global_step=current_step,
                last_checkpoint=last_checkpoint,
                last_checkpoint_step=last_checkpoint_step,
                message="training",
                error=None,
            )

            if args.mode == "fail" and current_step >= args.fail_at:
                raise RuntimeError(f"simulated trainer failure at step {current_step}")

            if args.mode == "stall" and current_step >= args.stall_after:
                append_event(
                    paths,
                    "trainer_stalling",
                    trainer_session_id=session_id,
                    global_step=current_step,
                    stall_seconds=args.stall_seconds,
                )
                deadline = time.monotonic() + args.stall_seconds
                while time.monotonic() < deadline:
                    time.sleep(min(args.step_interval, 0.25))
                raise RuntimeError("simulated stall window elapsed without restart")

        write_status(
            paths,
            owner_id=session_id,
            state="finished",
            pid=os.getpid(),
            global_step=current_step,
            last_checkpoint=last_checkpoint,
            last_checkpoint_step=last_checkpoint_step,
            message="training finished",
            error=None,
        )
        append_event(paths, "trainer_finished", trainer_session_id=session_id, global_step=current_step)
        send_notification(
            paths,
            text=f"training finished at step {current_step}",
            webhook_url=args.webhook_url,
            trainer_session_id=session_id,
            final_state="finished",
        )
        return 0
    except KeyboardInterrupt:
        control = load_control(paths)
        if control and control.get("action") == "restart" and control.get("trainer_session_id") == session_id:
            append_event(paths, "trainer_restarting", trainer_session_id=session_id, global_step=current_step)
            return 130

        write_status(
            paths,
            owner_id=session_id,
            state="interrupted",
            pid=os.getpid(),
            global_step=current_step,
            last_checkpoint=last_checkpoint,
            last_checkpoint_step=last_checkpoint_step,
            message="training interrupted",
            error="terminated by signal",
        )
        append_event(paths, "trainer_interrupted", trainer_session_id=session_id, global_step=current_step)
        send_notification(
            paths,
            text=f"training interrupted at step {current_step}",
            webhook_url=args.webhook_url,
            trainer_session_id=session_id,
            final_state="interrupted",
        )
        return 130
    except Exception as exc:
        write_status(
            paths,
            owner_id=session_id,
            state="failed",
            pid=os.getpid(),
            global_step=current_step,
            last_checkpoint=last_checkpoint,
            last_checkpoint_step=last_checkpoint_step,
            message="training failed",
            error=str(exc),
        )
        append_event(
            paths,
            "trainer_failed",
            trainer_session_id=session_id,
            global_step=current_step,
            error=str(exc),
        )
        send_notification(
            paths,
            text=f"training failed at step {current_step}: {exc}",
            webhook_url=args.webhook_url,
            trainer_session_id=session_id,
            final_state="failed",
        )
        return 1


def main() -> int:
    return run_trainer(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
