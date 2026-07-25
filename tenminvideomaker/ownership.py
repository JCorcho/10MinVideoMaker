"""Single-controller process ownership for the standalone supervisor."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import BinaryIO


class OwnershipError(RuntimeError):
    """Raised when another supervisor already owns the pipeline."""


def legacy_supervisor_process_ids(*, exclude_pid: int | None = None) -> tuple[int, ...]:
    """Return exact legacy run_supervisor.py processes without inspecting other projects."""
    if os.name != "nt":
        return ()
    script = (
        "$items = Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -in @('python.exe','pythonw.exe') -and $_.CommandLine -like "
        "'*10MinVideoMaker*scripts*run_supervisor.py*' } | "
        "Select-Object -ExpandProperty ProcessId; "
        "@($items) | ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:
        raise OwnershipError(
            "Could not verify whether the legacy supervisor is running."
        )
    try:
        values = json.loads(completed.stdout.strip() or "[]")
    except json.JSONDecodeError as error:
        raise OwnershipError("Could not parse the legacy supervisor process check.") from error
    if isinstance(values, int):
        values = [values]
    excluded = os.getpid() if exclude_pid is None else exclude_pid
    return tuple(sorted(int(value) for value in values if int(value) != excluded))


class SupervisorInstanceLock:
    """Hold a cross-process lock for the lifetime of one controller."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle: BinaryIO | None = None

    def acquire(self) -> None:
        if self._handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0)
        if handle.read(1) == b"":
            handle.seek(0)
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - production is Windows.
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            handle.close()
            raise OwnershipError(
                "Another 10MinVideoMaker controller is already running."
            ) from error
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - production is Windows.
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None

    def __enter__(self) -> "SupervisorInstanceLock":
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()
