"""Run the 24/7 10MinVideoMaker supervisor."""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import logging
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PROJECT_ROOT.parents[1]
EASY_INSTALL_ROOT = COMFY_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tenminvideomaker.assembly import FfmpegAssembler, probe_video
from tenminvideomaker.assets import ComfyLoraAssetClient
from tenminvideomaker.comfy_http import ComfyHttpClient
from tenminvideomaker.configuration import load_project_environment
from tenminvideomaker.delivery import DiscordDeliverySettings
from tenminvideomaker.mail import GmailClient, GmailSettings
from tenminvideomaker.ownership import (
    OwnershipError,
    SupervisorInstanceLock,
    legacy_supervisor_process_ids,
)
from tenminvideomaker.state_store import PipelineStateStore
from tenminvideomaker.storage import StorageLayout, migrate_legacy_storage
from tenminvideomaker.supervisor import PipelineSupervisor, SupervisorSettings


def restart_comfyui() -> bool:
    storage = StorageLayout.configured()
    script = PROJECT_ROOT / "scripts" / "restart_comfyui.ps1"
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-EasyInstallRoot",
            str(EASY_INSTALL_ROOT),
            "-ProjectRuntimeRoot",
            str(storage.state_root),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    if completed.returncode != 0:
        logging.getLogger("10MinVideoMaker.supervisor").error(
            "Controlled ComfyUI restart failed: %s",
            completed.stderr.strip() or completed.stdout.strip(),
        )
        return False
    return True


def build_supervisor(
    *,
    allow_restart: bool,
    require_human_review: bool | None = None,
) -> PipelineSupervisor:
    storage = StorageLayout.configured()
    migrate_legacy_storage(PROJECT_ROOT, layout=storage)
    configured_environment = load_project_environment(
        PROJECT_ROOT,
        storage_layout=storage,
    )
    os.environ.update(
        {
            key: value
            for key, value in configured_environment.items()
            if key.startswith("TENMIN_")
        }
    )
    storage.ensure()
    comfy_url = os.environ.get("TENMIN_COMFY_URL", "http://127.0.0.1:8188")
    ffmpeg = os.environ.get("TENMIN_FFMPEG", "ffmpeg")
    ffprobe = os.environ.get("TENMIN_FFPROBE", "ffprobe")
    comfy = ComfyHttpClient(comfy_url)
    settings = SupervisorSettings.from_environment()
    if require_human_review is not None:
        settings = replace(
            settings,
            require_human_review=require_human_review,
        )
    return PipelineSupervisor(
        store=PipelineStateStore(storage.database_path),
        mail_client=GmailClient(GmailSettings.from_environment()),
        asset_manager=ComfyLoraAssetClient(comfy),
        comfy=comfy,
        assembler=FfmpegAssembler(
            storage.finals_root,
            ffmpeg_executable=ffmpeg,
        ),
        settings=settings,
        restart_comfy=restart_comfyui if allow_restart else None,
        video_probe=partial(probe_video, ffprobe_executable=ffprobe),
        delivery=DiscordDeliverySettings.from_environment(configured_environment),
        storage=storage,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one durable state-machine tick instead of the five-minute loop.",
    )
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help="Disable controlled ComfyUI restart on fatal server failures.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=os.environ.get("TENMIN_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    storage = StorageLayout.configured()
    other_processes = legacy_supervisor_process_ids()
    if other_processes:
        raise OwnershipError(
            "Another legacy 10MinVideoMaker supervisor is already running "
            f"(PID {', '.join(str(pid) for pid in other_processes)})."
        )
    with SupervisorInstanceLock(storage.instance_lock_path):
        supervisor = build_supervisor(allow_restart=not args.no_restart)
        if args.once:
            supervisor.tick()
        else:
            supervisor.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
