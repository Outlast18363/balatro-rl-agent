import json
import subprocess
import sys
from pathlib import Path


def test_launcher_restarts_stalled_trainer_and_finishes(tmp_path):
    run_dir = tmp_path / "managed_run"
    repo_root = Path(__file__).resolve().parents[1]

    command = [
        sys.executable,
        "-m",
        "training.launcher",
        "--run-dir",
        str(run_dir),
        "--trainer-mode",
        "stall",
        "--restart-trainer-mode",
        "finish",
        "--max-steps",
        "4",
        "--step-interval",
        "0.05",
        "--checkpoint-interval",
        "2",
        "--stall-after",
        "2",
        "--stall-seconds",
        "5",
        "--supervisor-poll-interval",
        "0.05",
        "--launcher-poll-interval",
        "0.05",
        "--stall-timeout",
        "0.2",
        "--max-restarts",
        "1",
    ]

    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    event_names = [
        json.loads(line)["event"]
        for line in (run_dir / "events.ndjson").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert status["state"] == "finished"
    assert status["global_step"] == 4
    assert status["resume_from"] is not None
    assert "restart_requested" in event_names
    assert "launcher_restart_handled" in event_names
    assert "trainer_finished" in event_names
