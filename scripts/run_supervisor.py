"""Run the 24/7 10MinVideoMaker supervisor."""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(COMFY_ROOT))

import folder_paths

from tenminvideomaker.assembly import FfmpegAssembler, probe_video
from tenminvideomaker.assets import LoraAssetManager
from tenminvideomaker.comfy_http import ComfyHttpClient
from tenminvideomaker.configuration import load_project_environment
from tenminvideomaker.mail import GmailClient, GmailSettings
from tenminvideomaker.state_store import PipelineStateStore
from tenminvideomaker.supervisor import PipelineSupervisor, SupervisorSettings


def restart_comfyui() -> bool:
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
            str(PROJECT_ROOT / "runtime"),
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


def build_supervisor(*, allow_restart: bool) -> PipelineSupervisor:
    configured_environment = load_project_environment(PROJECT_ROOT)
    os.environ.update(
        {
            key: value
            for key, value in configured_environment.items()
            if key.startswith("TENMIN_")
        }
    )
    runtime = PROJECT_ROOT / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    comfy_url = os.environ.get("TENMIN_COMFY_URL", "http://127.0.0.1:8188")
    ffmpeg = os.environ.get("TENMIN_FFMPEG", "ffmpeg")
    ffprobe = os.environ.get("TENMIN_FFPROBE", "ffprobe")
    return PipelineSupervisor(
        store=PipelineStateStore(runtime / "pipeline.sqlite3"),
        mail_client=GmailClient(GmailSettings.from_environment()),
        asset_manager=LoraAssetManager(
            folder_paths.get_folder_paths("loras"),
            runtime / "asset_manifest.json",
        ),
        comfy=ComfyHttpClient(comfy_url),
        assembler=FfmpegAssembler(ffmpeg_executable=ffmpeg),
        settings=SupervisorSettings.from_environment(),
        restart_comfy=restart_comfyui if allow_restart else None,
        video_probe=partial(probe_video, ffprobe_executable=ffprobe),
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
    supervisor = build_supervisor(allow_restart=not args.no_restart)
    if args.once:
        supervisor.tick()
    else:
        supervisor.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
