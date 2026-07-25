"""Launch the loopback human-review GUI and its single supervisor worker."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import signal
import sys
import threading
import time
import webbrowser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tenminvideomaker.comfy_http import ComfyHttpClient, ComfyHttpError
from tenminvideomaker.gui_app import create_gui_app
from tenminvideomaker.gui_service import SupervisorController
from tenminvideomaker.ownership import (
    OwnershipError,
    SupervisorInstanceLock,
    legacy_supervisor_process_ids,
)
from tenminvideomaker.state_store import PipelineState, PipelineStateStore
from tenminvideomaker.storage import StorageLayout, migrate_legacy_storage

from scripts.run_supervisor import build_supervisor


LOGGER = logging.getLogger("10MinVideoMaker.gui-launcher")
SAFE_TAKEOVER_STATES = frozenset(
    {
        PipelineState.IDLE,
        PipelineState.WAITING_FOR_GROK,
        PipelineState.AWAITING_REVIEW,
        PipelineState.ERROR,
    }
)


def _take_over_idle_legacy_supervisor(process_ids: tuple[int, ...]) -> None:
    if not process_ids:
        return
    legacy_database = PROJECT_ROOT / "runtime" / "pipeline.sqlite3"
    if not legacy_database.is_file():
        raise OwnershipError(
            "A legacy supervisor is running but its state database is missing."
        )
    snapshot = PipelineStateStore(legacy_database).snapshot()
    if snapshot.state not in SAFE_TAKEOVER_STATES:
        raise OwnershipError(
            f"Legacy supervisor is actively processing {snapshot.state.value}; "
            "let it finish or cancel it from the existing console before opening the GUI."
        )
    try:
        running, pending = ComfyHttpClient().queue_counts()
    except ComfyHttpError as error:
        raise OwnershipError(
            "Could not verify the ComfyUI queue before taking over the idle supervisor."
        ) from error
    if running or pending:
        raise OwnershipError(
            "ComfyUI has queued work, so the GUI will not stop the legacy supervisor."
        )
    for process_id in process_ids:
        os.kill(process_id, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not legacy_supervisor_process_ids():
            LOGGER.info("Stopped the idle legacy supervisor and preserved its durable state.")
            return
        time.sleep(0.25)
    raise OwnershipError("The idle legacy supervisor did not stop within ten seconds.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the local GUI in the default browser.",
    )
    parser.add_argument(
        "--no-take-over",
        action="store_true",
        help="Refuse to stop an idle legacy run_supervisor.py process.",
    )
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("The supervisor GUI may bind only to loopback.")
    if not 1 <= args.port <= 65535:
        raise SystemExit("Port must be from 1 to 65535.")

    logging.basicConfig(
        level=os.environ.get("TENMIN_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    other_processes = legacy_supervisor_process_ids()
    if other_processes:
        if args.no_take_over:
            raise OwnershipError(
                "The legacy supervisor is still running. Close it before opening the GUI."
            )
        _take_over_idle_legacy_supervisor(other_processes)

    storage = StorageLayout.configured()
    with SupervisorInstanceLock(storage.instance_lock_path):
        migration = migrate_legacy_storage(PROJECT_ROOT, layout=storage)
        LOGGER.info(
            "Persistent storage ready at %s (legacy migration=%s).",
            storage.root,
            not migration["already_migrated"],
        )
        supervisor = build_supervisor(
            allow_restart=True,
            require_human_review=True,
        )
        controller = SupervisorController(supervisor, storage)
        app = create_gui_app(controller, storage, PROJECT_ROOT)
        controller.start()
        if not args.no_browser:
            threading.Timer(
                1.0,
                lambda: webbrowser.open(f"http://{args.host}:{args.port}/"),
            ).start()
        try:
            import uvicorn

            uvicorn.run(
                app,
                host=args.host,
                port=args.port,
                log_level="info",
                access_log=False,
            )
        finally:
            controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
