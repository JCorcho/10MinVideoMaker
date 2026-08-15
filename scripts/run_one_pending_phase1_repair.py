"""Run one repair candidate end-to-end from persisted state."""

from __future__ import annotations

import argparse
import hashlib
import os
import traceback
from dataclasses import replace
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tenminvideomaker.assets import ComfyLoraAssetClient
from tenminvideomaker.comfy_http import ComfyHttpClient
from tenminvideomaker.qc_backend import LlamaCppHttpBackend
from tenminvideomaker.qc_config import QualityControlSettings
from tenminvideomaker.qc_controller import _identity_mapping, Phase1QcController
from tenminvideomaker.qc_contracts import QcCandidateState, QcDecision, QcTier
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
    "TENMIN_QC_MODEL_PATH": "D:\\AI Shared Models\\LM Studio\\models\\HauhauCS\\Qwen3.6-27B-Uncensored-HauhauCS-Balanced\\Qwen3.6-27B-Uncensored-HauhauCS-Balanced-IQ3_M.gguf",
    "TENMIN_QC_PROJECTOR_PATH": "D:\\AI Shared Models\\LM Studio\\models\\HauhauCS\\Qwen3.6-27B-Uncensored-HauhauCS-Balanced\\mmproj-Qwen3.6-27B-Uncensored-HauhauCS-Balanced-f16.gguf",
    "TENMIN_QC_LLAMA_SHA256": "e25313077d8ed57a838c475ce2f3d31422881212caf2ddac2c18385e5e49ae69",
    "TENMIN_QC_MODEL_SHA256": "08c29d8693b04c58ea0c9f4d97e4e4c0f08099f87148c362b62c0f1f8ec7650c",
    "TENMIN_QC_PROJECTOR_SHA256": "81c5dfa3cd41367eb7b3f6e480def1a8cfaf50532ebae791931578b5c7a01437",
    "TENMIN_QC_GPU_UUID": "GPU-36264e58-dfbb-6e82-46f2-99e3b9ff6198",
    "TENMIN_QC_GPU_NAME": "NVIDIA GeForce RTX 4080 SUPER",
    "TENMIN_COMFY_URL": "http://127.0.0.1:8188",
}

_PENDING_RESTART_ACTION = "requeue_after_restart"


def _configure_environment(root: Path) -> None:
    for key, value in QC_ENV.items():
        os.environ.setdefault(key, str(value))
    os.environ["TENMIN_STORAGE_ROOT"] = str(root.resolve())
    os.environ.setdefault("TENMIN_COMFY_URL", "http://127.0.0.1:8188")
    vendor = Path(os.environ["TENMIN_QC_LLAMA_VENDOR_ROOT"]).resolve()
    path_value = os.environ.get("PATH", "")
    if str(vendor).casefold() not in {item.casefold() for item in path_value.split(os.pathsep) if item}:
        os.environ["PATH"] = f"{vendor}{os.pathsep}{path_value}"


def _qc_owned_runtime_layout(
    qc_settings: QualityControlSettings,
    storage: StorageLayout,
) -> StorageLayout:
    runtime_root = (PROJECT_ROOT / "runtime" / "qc-owned").resolve()
    disallowed = [
        storage.root,
        qc_settings.llama_vendor_root,
        qc_settings.llama_executable,
        qc_settings.llama_executable.parent if qc_settings.llama_executable is not None else None,
        qc_settings.model_path,
        qc_settings.model_path.parent if qc_settings.model_path is not None else None,
        qc_settings.projector_path,
        qc_settings.projector_path.parent if qc_settings.projector_path is not None else None,
    ]

    def _contains(root: Path, candidate: Path) -> bool:
        root_parts = tuple(part.casefold() for part in root.resolve().parts)
        candidate_parts = tuple(part.casefold() for part in candidate.resolve().parts)
        if len(candidate_parts) >= len(root_parts):
            return candidate_parts[: len(root_parts)] == root_parts
        return root_parts[: len(candidate_parts)] == candidate_parts

    for path in [entry for entry in disallowed if entry is not None]:
        if _contains(path, runtime_root) or _contains(runtime_root, path):
            raise RuntimeError("Refusing to place QC-owned runtime inside a shared root path.")
    return StorageLayout(runtime_root)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_controller_stack(layout: StorageLayout, qc_settings: QualityControlSettings):
    comfy = ComfyHttpClient(os.environ["TENMIN_COMFY_URL"])
    ffmpeg = os.environ.get("TENMIN_FFMPEG", "ffmpeg")
    ffprobe = os.environ.get("TENMIN_FFPROBE", "ffprobe")
    settings = replace(SupervisorSettings.from_environment(), require_human_review=True)
    controller = Phase1QcController(
        store=PipelineStateStore(layout.database_path),
        layout=layout,
        settings=qc_settings,
        backend_factory=lambda: LlamaCppHttpBackend(
            qc_settings,
            LlamaCppProcess(
                qc_settings,
                _qc_owned_runtime_layout(qc_settings, layout),
            ),
        ),
        prompt_root=Path(__file__).resolve().parents[1] / "prompts",
        ffmpeg_command=ffmpeg,
        ffprobe_command=ffprobe,
    )
    supervisor = PipelineSupervisor(
        store=controller.store,
        mail_client=type("noop", (), {"send_request": lambda self, **_: ""})(),
        asset_manager=ComfyLoraAssetClient(comfy),
        comfy=comfy,
        assembler=FfmpegAssembler(layout.finals_root, ffmpeg_executable=ffmpeg),
        settings=settings,
        storage=layout,
        chunk_assembler=SceneChunkAssembler(ffmpeg_executable=ffmpeg, ffprobe_executable=ffprobe),
        qc_controller=controller,
        delivery=None,
    )
    return controller.store, controller, supervisor, comfy


def _pending_generation_count(store: PipelineStateStore, job_id: str) -> int:
    return sum(
        1 for c in store.qc_candidates(job_id)
        if c.tier in {QcTier.A1, QcTier.B1, QcTier.B2}
        and c.state == QcCandidateState.PENDING_GENERATION
    )


def _find_candidate(store: PipelineStateStore, job_id: str, candidate_id: str):
    candidate = store.qc_candidate(candidate_id)
    if candidate is None:
        raise RuntimeError(f"Expected repair candidate {candidate_id} not found.")
    if candidate.tier not in {QcTier.A1, QcTier.B1, QcTier.B2}:
        raise RuntimeError(f"Candidate {candidate_id} is not A1/B1/B2.")
    if candidate.state == QcCandidateState.SUPERSEDED:
        raise RuntimeError(f"Candidate {candidate_id} is SUPERSEDED and cannot be repaired.")
    if candidate.state not in {
        QcCandidateState.GENERATING,
        QcCandidateState.PENDING_GENERATION,
        QcCandidateState.PENDING_QC,
    }:
        raise RuntimeError(f"Candidate {candidate_id} has state {candidate.state.value} and cannot be repaired.")
    if getattr(candidate, "job_id", None) is not None and candidate.job_id != job_id:
        raise RuntimeError(f"Candidate {candidate_id} is for job {candidate.job_id}, expected {job_id}.")
    return candidate


def _latest_qc_decision(store: PipelineStateStore, candidate_id: str):
    evaluations = [ev for ev in store.qc_evaluations(candidate_id) if ev.state == "COMPLETE"]
    if not evaluations:
        return None
    return evaluations[-1].normalized_decision


def _expected_revision_video(layout: StorageLayout, job_id: str, scene_id: int, revision: int) -> Path:
    return (
        layout.root
        / "jobs"
        / job_id
        / "scenes"
        / f"scene_{scene_id:04d}"
        / "revisions"
        / f"{revision:04d}"
        / "video.mp4"
    )


def _is_generation_prompt_absent(comfy: ComfyHttpClient, prompt_id: str | None) -> bool:
    if not prompt_id:
        return True
    if comfy.prompt_is_queued(prompt_id):
        return False
    return comfy.completed_prompt(prompt_id) is None


def _wait_for_idle_comfy(comfy: ComfyHttpClient, *, max_wait_seconds: int = 600) -> None:
    deadline = time.monotonic() + max_wait_seconds
    while time.monotonic() < deadline:
        running, pending = comfy.queue_counts()
        if running == 0 and pending == 0:
            return
        print(f"waiting_for_idle_comfy running={running} pending={pending}")
        time.sleep(2.0)
    raise RuntimeError("ComfyUI queue did not become idle within the waiting window.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    _configure_environment(root)
    layout = StorageLayout(root)
    store, controller, supervisor, comfy = _build_controller_stack(
        layout, replace(QualityControlSettings.from_environment(), quality_control_enabled=True, auto_advance_pass=False)
    )

    store.initialize()
    job = store.load_job(args.job_id)
    if job is None or job.job_id != args.job_id:
        raise RuntimeError(f"Job {args.job_id} not found in {root}")

    start_count = _pending_generation_count(store, args.job_id)

    candidate = _find_candidate(store, args.job_id, args.candidate_id)
    print(
        f"selected {candidate.tier.value} candidate: scene_id={candidate.scene_id} "
        f"candidate_id={candidate.candidate_id} revision={candidate.revision} "
        f"state={candidate.state.value}"
    )

    target_video = _expected_revision_video(layout, job.job_id, candidate.scene_id, candidate.revision)
    if (
        candidate.state == QcCandidateState.GENERATING
        and not target_video.is_file()
        and _is_generation_prompt_absent(comfy, candidate.generation_prompt_id)
    ):
        candidate = store.set_qc_candidate_state(
            candidate.candidate_id,
            QcCandidateState.PENDING_GENERATION,
            next_action=_PENDING_RESTART_ACTION,
        )
        print("requeued stuck repairing generating candidate after comfy restart")
        start_count += 1

    _wait_for_idle_comfy(comfy)

    try:
        if not target_video.is_file():
            document = controller._revision_document(store, candidate)
            supervisor.render_qc_candidates(job, ((candidate, document),))
            _wait_for_idle_comfy(comfy)
            if not target_video.is_file():
                raise RuntimeError(f"Expected render output was not produced at scene_{candidate.scene_id:04d} revision_{candidate.revision:04d}.")
            if target_video.stat().st_size == 0:
                raise RuntimeError(f"Expected scene_{candidate.scene_id:04d} revision_{candidate.revision:04d} MP4 is zero-byte.")
            candidate = store.complete_qc_candidate_generation(
                candidate.candidate_id,
                source_video_path=str(target_video),
                source_video_sha256=_file_sha256(target_video),
            )
        else:
            if candidate.source_video_path != str(target_video):
                raise RuntimeError("Existing candidate video path does not match revision_0002 path.")
            if target_video.stat().st_size == 0:
                raise RuntimeError(f"Expected scene_{candidate.scene_id:04d} revision_{candidate.revision:04d} MP4 is zero-byte.")
            if candidate.state in {QcCandidateState.PENDING_GENERATION, QcCandidateState.GENERATING}:
                candidate = store.complete_qc_candidate_generation(
                    candidate.candidate_id,
                    source_video_path=str(target_video),
                    source_video_sha256=_file_sha256(target_video),
                )

        if candidate.state != QcCandidateState.PENDING_QC:
            raise RuntimeError(f"Candidate not in PENDING_QC after completion: {candidate.state.value}")

        supervisor.release_memory()
        _wait_for_idle_comfy(comfy)

        backend = controller.backend_factory()
        identity = backend.start()
        try:
            evaluation = controller._evaluate_candidate(candidate, backend, identity)
            decision = evaluation.normalized_decision
            routed = controller.route_completed_evaluation(
                job,
                candidate.candidate_id,
                backend=backend,
                planner_identity=_identity_mapping(identity),
            )
        finally:
            backend.close()

        print(f"qwen_normalized_decision={decision.value}")
        print(f"candidate_final_state={routed.state.value}")
        if decision == QcDecision.FAIL:
            children = [
                c for c in store.qc_candidates(args.job_id)
                if c.parent_candidate_id == candidate.candidate_id
            ]
            print(f"child_created={len(children) > 0}")
            if children:
                print(f"child_state={children[0].state.value}")
        elif decision == QcDecision.PASS:
            print("child_created=False")
        else:
            print("child_created=False")
    except Exception as error:  # noqa: BLE001
        print("ONE REAL REPAIR BLOCKED — EXACT RUNTIME ERROR IDENTIFIED")
        print(f"{type(error).__name__}: {error}")
        traceback.print_exc()
        return 1

    current = store.qc_candidate(candidate.candidate_id)
    end_count = _pending_generation_count(store, args.job_id)
    print(f"pending_generation_count={start_count} -> {end_count}")
    print(f"candidate_state={current.state.value}")
    print(f"qwen_decision={_latest_qc_decision(store, candidate.candidate_id)}")

    if end_count >= start_count and current.state == QcCandidateState.PENDING_GENERATION:
        print("ONE REAL REPAIR BLOCKED — EXACT RUNTIME ERROR IDENTIFIED")
        return 1

    print("ONE REAL REPAIR COMPLETED — AUTOMATION PATH PROVEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
