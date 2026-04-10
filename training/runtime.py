from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TERMINAL_STATES = frozenset({"finished", "failed", "interrupted"})


@dataclass(frozen=True)
class RunPaths:
    """Filesystem layout for one managed training run."""

    root: Path

    @classmethod
    def from_path(cls, run_dir: str | Path) -> "RunPaths":
        return cls(Path(run_dir).expanduser().resolve())

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    @property
    def control_path(self) -> Path:
        return self.root / "control.json"

    @property
    def events_path(self) -> Path:
        return self.root / "events.ndjson"

    @property
    def trainer_pid_path(self) -> Path:
        return self.root / "trainer.pid"

    @property
    def supervisor_pid_path(self) -> Path:
        return self.root / "supervisor.pid"

    @property
    def launcher_pid_path(self) -> Path:
        return self.root / "launcher.pid"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(ts: datetime | None = None) -> str:
    instant = ts or now_utc()
    return instant.astimezone(timezone.utc).isoformat()


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp_name = handle.name

    os.replace(temp_name, path)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_status(paths: RunPaths) -> dict[str, Any] | None:
    return read_json(paths.status_path)


def replace_status(paths: RunPaths, payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    data["updated_at"] = data.get("updated_at", isoformat())
    _atomic_write_json(paths.status_path, data)
    return data


def write_status(
    paths: RunPaths,
    *,
    owner_id: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    current = load_status(paths) or {}

    if owner_id is not None:
        current_owner = current.get("trainer_session_id")
        if current_owner not in (None, owner_id):
            return current
        fields = {"trainer_session_id": owner_id, **fields}

    merged = {**current, **fields, "updated_at": isoformat()}
    _atomic_write_json(paths.status_path, merged)
    return merged


def append_event(paths: RunPaths, event: str, **fields: Any) -> dict[str, Any]:
    record = {"timestamp": isoformat(), "event": event, **fields}
    paths.events_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")
    return record


def load_events(paths: RunPaths) -> list[dict[str, Any]]:
    if not paths.events_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in paths.events_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def write_pid(path: Path, pid: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{pid or os.getpid()}\n", encoding="utf-8")


def read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    return int(raw)


def _checkpoint_name(step: int) -> str:
    return f"step_{step:08d}.ckpt.json"


def create_checkpoint(
    paths: RunPaths,
    step: int,
    *,
    payload: dict[str, Any] | None = None,
) -> Path:
    record = {
        "step": int(step),
        "created_at": isoformat(),
    }
    if payload:
        record.update(payload)

    target = paths.checkpoints_dir / _checkpoint_name(int(step))
    _atomic_write_json(target, record)
    return target


def _checkpoint_step(path: Path) -> int:
    stem = path.name.removeprefix("step_").removesuffix(".ckpt.json")
    return int(stem)


def latest_checkpoint(paths: RunPaths) -> dict[str, Any] | None:
    candidates = sorted(paths.checkpoints_dir.glob("step_*.ckpt.json"))
    if not candidates:
        return None

    latest = max(candidates, key=_checkpoint_step)
    payload = read_json(latest) or {}
    return {
        "path": latest,
        "step": int(payload.get("step", _checkpoint_step(latest))),
        "payload": payload,
    }


def write_control(paths: RunPaths, action: str, reason: str, **fields: Any) -> dict[str, Any]:
    payload = {
        "action": action,
        "reason": reason,
        "requested_at": isoformat(),
        **fields,
    }
    _atomic_write_json(paths.control_path, payload)
    return payload


def load_control(paths: RunPaths) -> dict[str, Any] | None:
    return read_json(paths.control_path)


def clear_control(paths: RunPaths) -> None:
    if paths.control_path.exists():
        paths.control_path.unlink()
