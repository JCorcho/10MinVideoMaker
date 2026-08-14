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
from tenminvideomaker.chunk_assembly import SceneChunkAssembler
from tenminvideomaker.comfy_http import ComfyHttpClient
from tenminvideomaker.configuration import load_project_environment
from tenminvideomaker.continuation_validation import require_auto_rollout_approval
from tenminvideomaker.delivery import DiscordDeliverySettings
from tenminvideomaker.mail import GmailClient, GmailSettings
from tenminvideomaker.ownership import (
    OwnershipError,
    SupervisorInstanceLock,
    legacy_supervisor_process_ids,
)
from tenminvideomaker.qc_backend import LlamaCppHttpBackend
from tenminvideomaker.qc_config import QualityControlSettings
from tenminvideomaker.qc_controller import Phase1QcController
from tenminvideomaker.qc_llama import LlamaCppProcess
from tenminvideomaker.state_store import PipelineStateStore
from tenminvideomaker.storage import StorageLayout, migrate_legacy_storage
from tenminvideomaker.supervisor import PipelineSupervisor, SupervisorSettings


def _require_auto_continuation_approval(
    supervisor: PipelineSupervisor,
    storage: StorageLayout,
) -> None:
    """Keep unvalidated continuation opt-in; fail closed before auto rollout."""
    if supervisor.settings.continuation_mode != "auto":
        return
    renderer = supervisor.continuation_renderer
    if renderer is None:
        raise RuntimeError("Automatic continuation requires D-drive continuation storage.")
    require_auto_rollout_approval(
        storage,
        implementation_sha256=renderer.implementation_sha256(),
        node_contracts_sha256=renderer.runtime_contract_sha256(),
    )


def _runtime_root_overlaps_disallowed_root(
    runtime_root: Path,
    disallowed_root: Path,
) -> bool:
    runtime_parts = tuple(part.casefold() for part in runtime_root.resolve().parts)
    disallowed_parts = tuple(part.casefold() for part in disallowed_root.resolve().parts)
    if not runtime_parts or not disallowed_parts:
        return False
    if len(runtime_parts) >= len(disallowed_parts):
        if runtime_parts[: len(disallowed_parts)] == disallowed_parts:
            return True
    if len(disallowed_parts) >= len(runtime_parts):
        if disallowed_parts[: len(runtime_parts)] == runtime_parts:
            return True
    return False


def _qc_owned_runtime_layout(
    qc_settings: QualityControlSettings,
    storage: StorageLayout,
) -> StorageLayout:
    candidate = (PROJECT_ROOT / "runtime" / "qc-owned").resolve()
    disallowed_roots: list[Path] = [storage.root]
    if qc_settings.llama_vendor_root is not None:
        disallowed_roots.append(qc_settings.llama_vendor_root)
    if qc_settings.llama_executable is not None:
        disallowed_roots.append(qc_settings.llama_executable)
        disallowed_roots.append(qc_settings.llama_executable.parent)
    if qc_settings.model_path is not None:
        disallowed_roots.append(qc_settings.model_path)
        disallowed_roots.append(qc_settings.model_path.parent)
    if qc_settings.projector_path is not None:
        disallowed_roots.append(qc_settings.projector_path)
        disallowed_roots.append(qc_settings.projector_path.parent)
    if any(
        _runtime_root_overlaps_disallowed_root(candidate, disallowed_root)
        for disallowed_root in disallowed_roots
    ):
        raise RuntimeError(
            "Refusing to place QC-owned runtime under persistent production storage or QC assets."
        )
    return StorageLayout(candidate)


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
    store = PipelineStateStore(storage.database_path)
    qc_settings = QualityControlSettings.from_environment(configured_environment)
    qc_controller = Phase1QcController(
        store=store,
        layout=storage,
        settings=qc_settings,
        backend_factory=lambda: LlamaCppHttpBackend(
            qc_settings,
            LlamaCppProcess(qc_settings, _qc_owned_runtime_layout(qc_settings, storage)),
        ),
        prompt_root=PROJECT_ROOT / "prompts",
        ffmpeg_command=ffmpeg,
        ffprobe_command=ffprobe,
    )
    supervisor = PipelineSupervisor(
        store=store,
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
        chunk_assembler=SceneChunkAssembler(
            ffmpeg_executable=ffmpeg,
            ffprobe_executable=ffprobe,
        ),
        qc_controller=qc_controller,
    )
    _require_auto_continuation_approval(supervisor, storage)
    return supervisor


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
