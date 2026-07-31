"""Launch the supervisor GUI and its single supervisor worker."""

from __future__ import annotations

import argparse
import ipaddress
import logging
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time
import webbrowser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tenminvideomaker.comfy_http import ComfyHttpClient, ComfyHttpError
from tenminvideomaker.configuration import load_project_environment
from tenminvideomaker.gui_app import create_gui_app
from tenminvideomaker.gui_service import SupervisorController
from tenminvideomaker.ownership import (
    OwnershipError,
    SupervisorInstanceLock,
    legacy_supervisor_process_ids,
)
from tenminvideomaker.state_store import PipelineState, PipelineStateStore
from tenminvideomaker.storage import StorageLayout, migrate_legacy_storage

from scripts.setup_and_start import ensure_comfyui
from scripts.run_supervisor import build_supervisor, restart_comfyui


LOGGER = logging.getLogger("10MinVideoMaker.gui-launcher")
SAFE_TAKEOVER_STATES = frozenset(
    {
        PipelineState.IDLE,
        PipelineState.WAITING_FOR_GROK,
        PipelineState.AWAITING_REVIEW,
        PipelineState.ERROR,
    }
)
LAN_ENABLED_ENV = "TENMIN_GUI_LAN_ENABLED"
LAN_PASSWORD_ENV = "TENMIN_GUI_LAN_PASSWORD"
LAN_USERNAME = "10min"


def _enabled(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _private_ipv4_addresses() -> tuple[str, ...]:
    addresses: set[str] = set()
    try:
        candidates = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except socket.gaierror:
        return ()
    for item in candidates:
        address = item[4][0]
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue
        if parsed.is_private and not parsed.is_loopback:
            addresses.add(str(parsed))
    return tuple(sorted(addresses))


def _gui_binding(
    args: argparse.Namespace,
    environment: dict[str, str],
) -> tuple[str, str | None]:
    requested_lan = args.lan or (not args.host and _enabled(environment.get(LAN_ENABLED_ENV, "")))
    if args.host and args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("--host may only select a loopback address; use --lan for mobile access.")
    if not requested_lan:
        return args.host or "127.0.0.1", None
    password = environment.get(LAN_PASSWORD_ENV, "")
    if len(password) < 12:
        raise SystemExit(
            "LAN access needs a 12+ character password. Run the launcher, choose optional settings, "
            "then Configure mobile LAN access."
        )
    return "0.0.0.0", password


def _required_inputs(comfy: ComfyHttpClient, node_type: str) -> dict:
    document = comfy.object_info(node_type)
    node = document.get(node_type, {})
    required = node.get("input", {}).get("required", {})
    return required if isinstance(required, dict) else {}


def _artifact_kind_options(required: dict) -> set[str]:
    specification = required.get("artifact_kind")
    if (
        not isinstance(specification, (list, tuple))
        or not specification
        or not isinstance(specification[0], (list, tuple))
    ):
        return set()
    return {
        value
        for value in specification[0]
        if isinstance(value, str)
    }


def _current_node_contract_loaded(comfy: ComfyHttpClient) -> bool:
    expected_artifacts = {
        "stage1_handoff",
        "stage2_video",
        "stage2_audio",
    }
    save_frame = _required_inputs(comfy, "10MinVideoMaker_SaveSceneFrame")
    save_chunk = _required_inputs(comfy, "10MinVideoMaker_SaveChunkLatent")
    load_chunk = _required_inputs(comfy, "10MinVideoMaker_LoadChunkLatent")
    return (
        "revision" in save_frame
        and _artifact_kind_options(save_chunk) == expected_artifacts
        and _artifact_kind_options(load_chunk) == expected_artifacts
        and "expected_temporal_tokens" in load_chunk
    )


def _ensure_current_node_contract(comfy: ComfyHttpClient) -> None:
    if _current_node_contract_loaded(comfy):
        return
    running, pending = comfy.queue_counts()
    if running or pending:
        raise OwnershipError(
            "ComfyUI must reload the updated 10MinVideoMaker nodes, but its queue is busy. "
            "Let the active work finish and launch the GUI again."
        )
    LOGGER.info("Reloading ComfyUI once to activate current continuation artifact nodes.")
    if not restart_comfyui() or not _current_node_contract_loaded(comfy):
        raise OwnershipError(
            "ComfyUI restarted but did not expose the current continuation artifact contracts."
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


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--lan",
        action="store_true",
        help="Bind the GUI to private LAN interfaces after password configuration.",
    )
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
    parser.add_argument(
        "--hold-new-jobs-for-review",
        action="store_true",
        help=(
            "Hold new Gmail jobs for manual Approve & Queue instead of starting "
            "them automatically. Intended for testing and review sessions."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("Port must be from 1 to 65535.")

    logging.basicConfig(
        level=os.environ.get("TENMIN_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    environment = load_project_environment(PROJECT_ROOT)
    host, lan_password = _gui_binding(args, environment)
    ensure_comfyui(environment)
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
        _ensure_current_node_contract(ComfyHttpClient())
        supervisor = build_supervisor(
            allow_restart=True,
            require_human_review=args.hold_new_jobs_for_review,
        )
        LOGGER.info(
            "New Gmail jobs will %s.",
            "wait for manual review" if args.hold_new_jobs_for_review else "start automatically",
        )
        controller = SupervisorController(supervisor, storage)
        app = create_gui_app(controller, storage, PROJECT_ROOT, lan_password=lan_password)
        controller.start()
        if lan_password:
            addresses = _private_ipv4_addresses()
            address_text = ", ".join(f"http://{address}:{args.port}/" for address in addresses)
            LOGGER.warning(
                "LAN GUI active. Sign in as %s with the configured LAN password. Phone URL: %s",
                LAN_USERNAME,
                address_text or f"http://<this-PC-LAN-IP>:{args.port}/",
            )
        if not args.no_browser:
            threading.Timer(
                1.0,
                lambda: webbrowser.open(f"http://127.0.0.1:{args.port}/"),
            ).start()
        try:
            import uvicorn

            uvicorn.run(
                app,
                host=host,
                port=args.port,
                log_level="info",
                access_log=False,
            )
        finally:
            controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
