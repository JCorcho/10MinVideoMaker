"""Run a historical Phase-1 QC sidecar for an existing production job."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tenminvideomaker.assets import ComfyLoraAssetClient
from tenminvideomaker.comfy_http import ComfyHttpClient
from tenminvideomaker.qc_backend import LlamaCppHttpBackend
from tenminvideomaker.qc_config import QualityControlSettings
from tenminvideomaker.qc_contracts import QcCandidateState, QcTier
from tenminvideomaker.qc_controller import Phase1QcController
from tenminvideomaker.qc_llama import LlamaCppProcess
from tenminvideomaker.state_store import PipelineStateStore
from tenminvideomaker.storage import StorageLayout
from tenminvideomaker.supervisor import (
    FfmpegAssembler,
    PipelineSupervisor,
    SceneChunkAssembler,
    SupervisorSettings,
)


QC_ENV = {
    "TENMIN_QUALITY_CONTROL_ENABLED": "1",
    "TENMIN_QC_AUTO_ADVANCE_PASS": "0",
    "TENMIN_QC_LOOPBACK_PORT": "18081",
    "TENMIN_QC_LLAMA_EXECUTABLE": r"C:\Users\Elijah\.lmstudio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.28.2\llama-server.exe",
    "TENMIN_QC_LLAMA_VENDOR_ROOT": r"C:\Users\Elijah\.lmstudio\extensions\backends\vendor\win-llama-cuda12-vendor-v2",
    "TENMIN_QC_MODEL_PATH": r"D:\AI Shared Models\LM Studio\models\HauhauCS\Qwen3.6-27B-Uncensored-HauhauCS-Balanced\Qwen3.6-27B-Uncensored-HauhauCS-Balanced-IQ3_M.gguf",
    "TENMIN_QC_PROJECTOR_PATH": r"D:\AI Shared Models\LM Studio\models\HauhauCS\Qwen3.6-27B-Uncensored-HauhauCS-Balanced\mmproj-Qwen3.6-27B-Uncensored-HauhauCS-Balanced-f16.gguf",
    "TENMIN_QC_LLAMA_SHA256": "e25313077d8ed57a838c475ce2f3d31422881212caf2ddac2c18385e5e49ae69",
    "TENMIN_QC_MODEL_SHA256": "08c29d8693b04c58ea0c9f4d97e4e4c0f08099f87148c362b62c0f1f8ec7650c",
    "TENMIN_QC_PROJECTOR_SHA256": "81c5dfa3cd41367eb7b3f6e480def1a8cfaf50532ebae791931578b5c7a01437",
    "TENMIN_QC_GPU_UUID": "GPU-36264e58-dfbb-6e82-46f2-99e3b9ff6198",
    "TENMIN_QC_GPU_NAME": "NVIDIA GeForce RTX 4080 SUPER",
    "TENMIN_STORAGE_ROOT": r"D:\LTX_Supervisor_Storage",
    "TENMIN_COMFY_URL": "http://127.0.0.1:8188",
}


def _configure_environment() -> None:
    os.environ.update(QC_ENV)
    os.environ.setdefault("TENMIN_COMFY_URL", "http://127.0.0.1:8188")
    vendor = Path(os.environ["TENMIN_QC_LLAMA_VENDOR_ROOT"]).resolve()
    path_value = os.environ.get("PATH", "")
    if vendor.as_posix().casefold() not in {item.casefold() for item in path_value.split(os.pathsep)}:
        os.environ["PATH"] = f"{vendor}{os.pathsep}{path_value}"


def _qc_owned_runtime_layout(
    qc_settings: QualityControlSettings,
    layout: StorageLayout,
) -> StorageLayout:
    return StorageLayout((PROJECT_ROOT / "runtime" / "qc-owned").resolve())


def _build_stack(layout: StorageLayout, settings: QualityControlSettings):
    comfy = ComfyHttpClient(os.environ["TENMIN_COMFY_URL"])
    ffmpeg = os.environ.get("TENMIN_FFMPEG", "ffmpeg")
    ffprobe = os.environ.get("TENMIN_FFPROBE", "ffprobe")
    supervisor_settings = replace(
        SupervisorSettings.from_environment(),
        require_human_review=True,
    )
    controller = Phase1QcController(
        store=PipelineStateStore(layout.database_path),
        layout=layout,
        settings=settings,
        backend_factory=lambda: LlamaCppHttpBackend(
            settings,
            LlamaCppProcess(settings, _qc_owned_runtime_layout(settings, layout)),
        ),
        prompt_root=Path(__file__).resolve().parents[1] / "prompts",
        ffmpeg_command=ffmpeg,
        ffprobe_command=ffprobe,
    )
    mail_client = type("noop", (), {"send_request": lambda self, **_: ""})()
    supervisor = PipelineSupervisor(
        store=controller.store,
        mail_client=mail_client,
        asset_manager=ComfyLoraAssetClient(comfy),
        comfy=comfy,
        assembler=FfmpegAssembler(layout.finals_root, ffmpeg_executable=ffmpeg),
        settings=supervisor_settings,
        storage=layout,
        chunk_assembler=SceneChunkAssembler(ffmpeg_executable=ffmpeg, ffprobe_executable=ffprobe),
        qc_controller=controller,
        delivery=None,
    )
    return controller, supervisor


def _snapshot_line(snapshot) -> str:
    return f"state={snapshot.state},job_id={snapshot.job_id},active_scene_id={snapshot.active_scene_id}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_actionable_candidate(
    *,
    store: PipelineStateStore,
    job_id: str,
    scene_id: int,
):
    candidates = tuple(store.qc_candidates(job_id, scene_id))
    if not candidates:
        return None
    current = max(candidates, key=lambda item: (item.revision, item.created_at))
    seen: set[str] = set()
    while current.state == QcCandidateState.SUPERSEDED:
        if current.candidate_id in seen:
            raise RuntimeError(
                f"QC candidate chain contains a cycle for {job_id} scene {scene_id}."
            )
        seen.add(current.candidate_id)
        children = [
            item
            for item in candidates
            if item.parent_candidate_id == current.candidate_id
        ]
        if not children:
            break
        current = max(children, key=lambda item: (item.revision, item.created_at))
    return current


def _candidate_counts(candidates: tuple) -> str:
    counts: dict[str, int] = {}
    for candidate in candidates:
        if candidate is None:
            key = "MISSING"
        else:
            key = candidate.state.value
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _has_complete_qc_evaluation(store: PipelineStateStore, candidate_id: str) -> bool:
    return any(item.state == "COMPLETE" for item in store.qc_evaluations(candidate_id))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--scene-limit", type=int, default=6)
    args = parser.parse_args()
    if args.scene_limit <= 0:
        raise RuntimeError("--scene-limit must be a positive integer.")

    _configure_environment()
    storage = StorageLayout(Path(os.environ["TENMIN_STORAGE_ROOT"]))
    store = PipelineStateStore(storage.database_path)
    store.initialize()
    job = store.load_job(args.job_id)
    if job is None:
        raise RuntimeError(f"Job not found: {args.job_id}")

    scene_ids = tuple(scene.scene_id for scene in sorted(job.scenes, key=lambda item: item.scene_id))
    selected_scene_ids = scene_ids[: args.scene_limit]

    settings = replace(
        QualityControlSettings.from_environment(),
        quality_control_enabled=True,
        auto_advance_pass=False,
    )
    controller, supervisor = _build_stack(storage, settings)

    before = store.snapshot()
    print(f"GLOBAL_PIPELINE_BEFORE={_snapshot_line(before)}")

    existing_tiers = tuple(item.tier for item in store.qc_candidates(args.job_id))
    if QcTier.ORIGINAL not in existing_tiers:
        originals = controller.register_original_candidates(job)
        if originals:
            print(f"REGISTERED_ORIGINALS={','.join(item.candidate_id for item in originals)}")

    print(f"JOB={args.job_id}")
    print(f"SCENES_SELECTED={','.join(map(str, selected_scene_ids))}")

    while True:
        latest_by_scene: dict[int, object] = {}
        qwen_batch: list[object] = []
        changed = False

        for scene_id in selected_scene_ids:
            candidate = _latest_actionable_candidate(
                store=store,
                job_id=args.job_id,
                scene_id=scene_id,
            )
            latest_by_scene[scene_id] = candidate
            if candidate is None:
                continue

            if candidate.state == QcCandidateState.PENDING_GENERATION:
                document = controller._revision_document(store, candidate)
                supervisor.render_qc_candidates(job, ((candidate, document),))
                changed = True
                continue

            if candidate.state == QcCandidateState.GENERATING:
                output = Path(candidate.source_video_path)
                if output.is_file() and output.stat().st_size > 0:
                    controller.store.complete_qc_candidate_generation(
                        candidate.candidate_id,
                        source_video_path=str(output),
                        source_video_sha256=_sha256_file(output),
                    )
                    changed = True
                    continue
                if candidate.generation_prompt_id is not None:
                    continue
                candidate = controller.store.set_qc_candidate_state(
                    candidate.candidate_id,
                    QcCandidateState.PENDING_GENERATION,
                    next_action=candidate.next_action,
                )
                changed = True
                latest_by_scene[scene_id] = candidate
                continue

            if candidate.state == QcCandidateState.PENDING_QC:
                qwen_batch.append(candidate)
                continue

            if candidate.state == QcCandidateState.QC_RUNNING:
                qwen_batch.append(candidate)
                continue

            if candidate.state == QcCandidateState.PASS_PENDING_HUMAN:
                continue

            if candidate.state == QcCandidateState.HOLD_FOR_REVIEW:
                if candidate.tier == QcTier.B2:
                    continue
                continue

        if qwen_batch:
            running, pending = supervisor.comfy.queue_counts()
            if running or pending:
                time.sleep(2.0)
                continue
            supervisor.release_memory()
            backend = controller.backend_factory()
            controller._active_backend = backend
            identity = backend.start()
            try:
                for candidate in qwen_batch:
                    refreshed = store.qc_candidate(candidate.candidate_id)
                    if _has_complete_qc_evaluation(store, refreshed.candidate_id):
                        controller.route_completed_evaluation(
                            job,
                            refreshed.candidate_id,
                            backend=backend,
                            planner_identity=asdict(identity),
                        )
                        changed = True
                        continue
                    controller._evaluate_candidate(refreshed, backend, identity)
                    controller.route_completed_evaluation(
                        job,
                        refreshed.candidate_id,
                        backend=backend,
                        planner_identity=asdict(identity),
                    )
                    changed = True
                    continue
            finally:
                backend.close()
                if controller._active_backend is backend:
                    controller._active_backend = None

        if not changed:
            any_inflight = any(
                candidate is not None
                and candidate.state
                in {
                    QcCandidateState.PENDING_GENERATION,
                    QcCandidateState.GENERATING,
                    QcCandidateState.PENDING_QC,
                    QcCandidateState.QC_RUNNING,
                }
                for candidate in latest_by_scene.values()
            )
            if not any_inflight:
                break
            time.sleep(2.0)
            continue

        time.sleep(2.0)

    after = store.snapshot()
    final_candidates = tuple(
        _latest_actionable_candidate(
            store=store,
            job_id=args.job_id,
            scene_id=scene_id,
        )
        for scene_id in selected_scene_ids
    )
    print(f"GLOBAL_PIPELINE_AFTER={_snapshot_line(after)}")
    print(f"SIDECAR_CHANGED_GLOBAL_PIPELINE={'YES' if after != before else 'NO'}")
    print(f"SELECTED_SIX_CANDIDATE_STATE_COUNTS={_candidate_counts(final_candidates)}")
    print(f"SIDECAR_ALIVE={1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
