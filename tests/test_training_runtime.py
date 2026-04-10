from training.runtime import RunPaths
from training.runtime import create_checkpoint
from training.runtime import latest_checkpoint
from training.runtime import load_events
from training.runtime import load_status
from training.runtime import write_status


def test_latest_checkpoint_picks_highest_step(tmp_path):
    paths = RunPaths.from_path(tmp_path / "managed_run")
    paths.ensure()

    create_checkpoint(paths, 2)
    create_checkpoint(paths, 12)
    create_checkpoint(paths, 7)

    checkpoint = latest_checkpoint(paths)

    assert checkpoint is not None
    assert checkpoint["step"] == 12
    assert checkpoint["path"].name == "step_00000012.ckpt.json"


def test_write_status_preserves_previous_fields_for_same_owner(tmp_path):
    paths = RunPaths.from_path(tmp_path / "managed_run")
    paths.ensure()

    write_status(paths, owner_id="trainer-0", state="running", global_step=1, message="boot")
    write_status(paths, owner_id="trainer-0", global_step=2)

    status = load_status(paths)

    assert status is not None
    assert status["trainer_session_id"] == "trainer-0"
    assert status["state"] == "running"
    assert status["global_step"] == 2
    assert status["message"] == "boot"


def test_write_status_ignores_stale_owner_updates(tmp_path):
    paths = RunPaths.from_path(tmp_path / "managed_run")
    paths.ensure()

    write_status(paths, owner_id="trainer-0", state="running", global_step=3)
    write_status(paths, owner_id="trainer-1", state="running", global_step=4)
    write_status(paths, owner_id="trainer-0", state="interrupted", global_step=3)

    status = load_status(paths)

    assert status is not None
    assert status["trainer_session_id"] == "trainer-0"
    assert status["state"] == "running"
    assert status["global_step"] == 3

    # New trainer claims ownership by replacing the status file.
    from training.runtime import replace_status

    replace_status(
        paths,
        {
            "trainer_session_id": "trainer-1",
            "state": "running",
            "global_step": 4,
        },
    )
    write_status(paths, owner_id="trainer-0", state="failed", global_step=3)

    status = load_status(paths)

    assert status is not None
    assert status["trainer_session_id"] == "trainer-1"
    assert status["state"] == "running"
    assert status["global_step"] == 4
