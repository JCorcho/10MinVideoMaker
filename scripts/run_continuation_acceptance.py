"""Run the bounded four-case LTX 2.3 continuation acceptance matrix.

The runner never starts the supervisor, writes only project-owned D-drive
storage, and deliberately leaves automatic continuation rollout disabled.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tenminvideomaker.chunk_artifacts import sha256_file
from tenminvideomaker.comfy_http import ComfyHttpClient, ComfyHttpError, find_video_output
from tenminvideomaker.continuation_acceptance import (
    AcceptancePlans,
    build_acceptance_job,
    build_acceptance_plans,
)
from tenminvideomaker.continuation_workflow import (
    build_continuation_stage1_workflow,
    build_continuation_stage2_workflow,
)
from tenminvideomaker.storage import StorageLayout, write_json_atomic
from tenminvideomaker.workflow_builder import validate_against_object_info

ACCEPTANCE_SCHEMA_VERSION = 4
DIAGNOSTIC_GUIDE_FRAME_COUNT = 17
LATER_VISIBLE_OVERLAP_FRAME_COUNT = 25
BASE_FINAL_FRAME_INDEX = 120
BASE_DIAGNOSTIC_GUIDE_START_FRAME_INDEX = 96
BASE_LATER_VISIBLE_OVERLAP_START_FRAME_INDEX = (
    BASE_FINAL_FRAME_INDEX - LATER_VISIBLE_OVERLAP_FRAME_COUNT + 1
)
DIRECT_HANDOFF_BASE_SEAM_FRAME_INDEX = 103
DIRECT_HANDOFF_CONTINUATION_SEAM_FRAME_INDEX = 8
RAW_SUFFIXES = (".mkv",)


class AcceptanceRunError(RuntimeError):
    """Raised when one matrix case cannot complete or be measured safely."""


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the D-drive four-case LTX continuation acceptance matrix. "
            "It does not start or modify the supervisor."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-job-id")
    source.add_argument("--source-payload-file", type=Path)
    parser.add_argument(
        "--source-frame",
        type=Path,
        help="Exact cached frame; required with --source-payload-file.",
    )
    parser.add_argument("--source-scene-id", required=True, type=int)
    parser.add_argument("--source-revision", default=1, type=int)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--timeout-seconds", default=2700.0, type=float)
    parser.add_argument("--poll-seconds", default=3.0, type=float)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate all live graphs without queuing a prompt.",
    )
    return parser


def _new_run_id() -> str:
    return "continuation-acceptance-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _read_saved_job_payload(storage: StorageLayout, job_id: str) -> Mapping[str, Any]:
    database = storage.database_path.resolve()
    if not database.is_file():
        raise AcceptanceRunError(f"Saved pipeline database is missing: {database}")
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT payload_json FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise AcceptanceRunError(f"Saved source job {job_id!r} was not found.")
    try:
        payload = json.loads(row[0])
    except (TypeError, json.JSONDecodeError) as error:
        raise AcceptanceRunError("Saved source payload is unreadable.") from error
    if not isinstance(payload, Mapping):
        raise AcceptanceRunError("Saved source payload is not an object.")
    return payload


def _read_payload_file(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AcceptanceRunError(f"Source payload file is unreadable: {path}") from error
    if not isinstance(payload, Mapping):
        raise AcceptanceRunError("Source payload file must contain a JSON object.")
    return payload


def _ensure_empty_queue(comfy: ComfyHttpClient) -> None:
    running, pending = comfy.queue_counts()
    if running or pending:
        raise AcceptanceRunError(
            f"ComfyUI queue is not empty ({running} running, {pending} pending); "
            "acceptance will not overlap another render."
        )


def _stat_snapshot(document: Mapping[str, Any]) -> dict[str, int] | None:
    devices = document.get("devices")
    if not isinstance(devices, list) or not devices:
        return None
    device = devices[0]
    if not isinstance(device, Mapping):
        return None
    total = device.get("vram_total")
    free = device.get("vram_free")
    if (
        isinstance(total, bool)
        or isinstance(free, bool)
        or not isinstance(total, int)
        or not isinstance(free, int)
        or total < free
    ):
        return None
    return {"vram_total": total, "vram_free": free, "vram_used": total - free}


def _wait_with_telemetry(
    comfy: ComfyHttpClient,
    prompt_id: str,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    started = time.monotonic()
    deadline = started + timeout_seconds
    peak_vram = 0
    samples = 0
    while time.monotonic() < deadline:
        snapshot = _stat_snapshot(comfy.system_stats())
        if snapshot is not None:
            peak_vram = max(peak_vram, snapshot["vram_used"])
            samples += 1
        history = comfy.completed_prompt(prompt_id)
        if history is not None:
            return history, {
                "runtime_seconds": round(time.monotonic() - started, 3),
                "peak_vram_bytes": peak_vram,
                "telemetry_samples": samples,
            }
        time.sleep(poll_seconds)
    comfy.cancel_owned_prompt(prompt_id)
    raise AcceptanceRunError(
        f"Acceptance prompt {prompt_id} exceeded {timeout_seconds:g} seconds and was cancelled."
    )


def _run_graph(
    comfy: ComfyHttpClient,
    *,
    label: str,
    workflow: Mapping[str, Any],
    output_node_id: str | None,
    destination: Path | None,
    timeout_seconds: float,
    poll_seconds: float,
) -> Mapping[str, Any]:
    prompt_id = comfy.queue_prompt(workflow)
    history, telemetry = _wait_with_telemetry(
        comfy,
        prompt_id,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )
    result: dict[str, Any] = {"label": label, "prompt_id": prompt_id, **telemetry}
    if output_node_id is not None:
        if destination is None:
            raise AcceptanceRunError(f"{label} is missing its D-drive destination.")
        metadata = find_video_output(
            history,
            output_node_id,
            expected_suffixes=RAW_SUFFIXES,
        )
        comfy.download_output(metadata, destination)
        result.update(
            {
                "raw_video_path": str(destination),
                "raw_video_sha256": sha256_file(destination),
                "raw_video_bytes": destination.stat().st_size,
            }
        )
    return result


def _extract_frame(video: Path, frame_index: int, destination: Path) -> Path:
    if frame_index < 0:
        raise AcceptanceRunError("Frame index must be non-negative.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expression = f"select=eq(n\\,{frame_index})"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            expression,
            "-frames:v",
            "1",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not destination.is_file():
        detail = completed.stderr.strip() or "ffmpeg produced no frame"
        raise AcceptanceRunError(f"Could not extract frame {frame_index}: {detail}")
    return destination


def _probe_video(video: Path) -> Mapping[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,pix_fmt",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AcceptanceRunError(completed.stderr.strip() or "ffprobe failed.")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise AcceptanceRunError("ffprobe returned invalid JSON.") from error
    if not isinstance(result, Mapping):
        raise AcceptanceRunError("ffprobe returned an invalid document.")
    return result


def _image_difference(left: Path, right: Path) -> Mapping[str, float | str]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:  # pragma: no cover - ComfyUI supplies both.
        raise AcceptanceRunError("Pillow and NumPy are required for acceptance metrics.") from error
    with Image.open(left) as image:
        left_pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    with Image.open(right) as image:
        right_pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    if left_pixels.shape != right_pixels.shape:
        raise AcceptanceRunError(
            f"Metric frames differ in size: {left_pixels.shape} versus {right_pixels.shape}."
        )
    difference = np.abs(left_pixels - right_pixels)
    luma = (
        difference[..., 0] * 0.2126
        + difference[..., 1] * 0.7152
        + difference[..., 2] * 0.0722
    )
    chroma = np.abs((left_pixels[..., 0] - left_pixels[..., 2]) - (right_pixels[..., 0] - right_pixels[..., 2]))
    return {
        "rgb_mae": round(float(difference.mean()), 6),
        "luma_mae": round(float(luma.mean()), 6),
        "red_blue_chroma_mae": round(float(chroma.mean()), 6),
    }


def _flow_discontinuity(
    previous_a: Path,
    previous_b: Path,
    current_a: Path,
    current_b: Path,
) -> Mapping[str, float | str]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {"status": "unavailable", "reason": "OpenCV is not installed."}

    def read(path: Path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise AcceptanceRunError(f"OpenCV could not read metric frame: {path}")
        return image

    previous_flow = cv2.calcOpticalFlowFarneback(
        read(previous_a), read(previous_b), None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    current_flow = cv2.calcOpticalFlowFarneback(
        read(current_a), read(current_b), None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return {
        "status": "measured",
        "mean_vector_discontinuity": round(
            float(np.linalg.norm(previous_flow - current_flow, axis=2).mean()), 6
        ),
        "previous_mean_motion": round(
            float(np.linalg.norm(previous_flow, axis=2).mean()), 6
        ),
        "current_mean_motion": round(
            float(np.linalg.norm(current_flow, axis=2).mean()), 6
        ),
    }


def _case_metrics(
    *,
    run_root: Path,
    base_video: Path,
    case_name: str,
    case_video: Path,
    guide_start_frame_index: int | None,
    guide_frame_count: int | None,
) -> Mapping[str, Any]:
    metrics_root = run_root / "metrics" / case_name
    report: dict[str, Any] = {
        "video": _probe_video(case_video),
        "manual_review": {
            "anatomy": "pending human review",
            "hand_object_contact": "pending human review",
            "identity_wardrobe": "pending human review",
            "camera_velocity": "pending human review",
            "visible_motion_restart": "pending human review",
        },
    }
    if case_name == "latent_overlap":
        base_previous = _extract_frame(
            base_video,
            DIRECT_HANDOFF_BASE_SEAM_FRAME_INDEX - 1,
            metrics_root / "base_0102.png",
        )
        base_boundary = _extract_frame(
            base_video,
            DIRECT_HANDOFF_BASE_SEAM_FRAME_INDEX,
            metrics_root / "base_0103.png",
        )
        case_boundary = _extract_frame(
            case_video,
            DIRECT_HANDOFF_CONTINUATION_SEAM_FRAME_INDEX,
            metrics_root / "case_0008.png",
        )
        case_next = _extract_frame(
            case_video,
            DIRECT_HANDOFF_CONTINUATION_SEAM_FRAME_INDEX + 1,
            metrics_root / "case_0009.png",
        )
        report["production_seam"] = {
            "base_end_frame": DIRECT_HANDOFF_BASE_SEAM_FRAME_INDEX,
            "continuation_start_frame": DIRECT_HANDOFF_CONTINUATION_SEAM_FRAME_INDEX,
        }
        report["seam_difference"] = _image_difference(base_boundary, case_boundary)
        report["seam_flow"] = _flow_discontinuity(
            base_previous,
            base_boundary,
            case_boundary,
            case_next,
        )
    elif guide_start_frame_index is None and guide_frame_count is None:
        base_previous = _extract_frame(base_video, 119, metrics_root / "base_0119.png")
        base_final = _extract_frame(base_video, 120, metrics_root / "base_0120.png")
        case_first = _extract_frame(case_video, 0, metrics_root / "case_0000.png")
        case_second = _extract_frame(case_video, 1, metrics_root / "case_0001.png")
        report["single_boundary_difference"] = _image_difference(base_final, case_first)
        report["motion_restart_flow"] = _flow_discontinuity(
            base_previous,
            base_final,
            case_first,
            case_second,
        )
    elif guide_start_frame_index is not None and guide_frame_count is not None:
        if (
            isinstance(guide_start_frame_index, bool)
            or not isinstance(guide_start_frame_index, int)
            or guide_start_frame_index < 0
            or isinstance(guide_frame_count, bool)
            or not isinstance(guide_frame_count, int)
            or guide_frame_count < 2
            or guide_start_frame_index + guide_frame_count
            > BASE_FINAL_FRAME_INDEX + 1
        ):
            raise AcceptanceRunError("guide_frame_count is outside the base window.")
        guide_start = _extract_frame(
            base_video,
            guide_start_frame_index,
            metrics_root / f"base_{guide_start_frame_index:04d}.png",
        )
        guide_end_frame_index = guide_start_frame_index + guide_frame_count - 1
        guide_previous = _extract_frame(
            base_video,
            guide_end_frame_index - 1,
            metrics_root / f"base_{guide_end_frame_index - 1:04d}.png",
        )
        guide_end = _extract_frame(
            base_video,
            guide_end_frame_index,
            metrics_root / f"base_{guide_end_frame_index:04d}.png",
        )
        case_first = _extract_frame(case_video, 0, metrics_root / "case_0000.png")
        case_overlap_end = _extract_frame(
            case_video,
            guide_frame_count - 1,
            metrics_root / f"case_{guide_frame_count - 1:04d}.png",
        )
        case_first_new = _extract_frame(
            case_video,
            guide_frame_count,
            metrics_root / f"case_{guide_frame_count:04d}.png",
        )
        report["guide_span"] = {
            "source_start_frame": guide_start_frame_index,
            "source_end_frame": guide_end_frame_index,
            "frame_count": guide_frame_count,
        }
        report["overlap_start_difference"] = _image_difference(guide_start, case_first)
        report["overlap_end_difference"] = _image_difference(guide_end, case_overlap_end)
        report["seam_flow"] = _flow_discontinuity(
            guide_previous,
            guide_end,
            case_overlap_end,
            case_first_new,
        )
    else:
        raise AcceptanceRunError(
            "guide_start_frame_index and guide_frame_count must be set together."
        )
    return report


def _live_contract_errors(
    comfy: ComfyHttpClient,
    workflows: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Mapping[str, tuple[str, ...]]:
    node_types = sorted(
        {
            node["class_type"]
            for workflow in workflows.values()
            for node in workflow.values()
            if isinstance(node.get("class_type"), str)
        }
    )
    object_info = {}
    for node_type in node_types:
        document = comfy.object_info(node_type)
        node = document.get(node_type)
        if isinstance(node, Mapping):
            object_info[node_type] = node
    return {
        name: errors
        for name, workflow in workflows.items()
        if (errors := validate_against_object_info(workflow, object_info))
    }


def main() -> int:
    args = argument_parser().parse_args()
    if args.timeout_seconds <= 0 or args.poll_seconds <= 0:
        raise SystemExit("--timeout-seconds and --poll-seconds must be positive.")
    storage = StorageLayout.configured()
    storage.ensure()
    run_id = args.run_id or _new_run_id()
    run_root = storage.root / "acceptance" / run_id
    if run_root.exists() or storage.job_root(run_id).exists():
        raise SystemExit(f"Acceptance run ID already exists: {run_id}")
    if args.source_payload_file is not None:
        if args.source_frame is None:
            raise SystemExit("--source-frame is required with --source-payload-file.")
        source_payload = _read_payload_file(args.source_payload_file)
        source_frame = args.source_frame.resolve()
        source_identifier = str(source_payload.get("job_id") or args.source_payload_file)
    else:
        if args.source_frame is not None:
            raise SystemExit("--source-frame is only valid with --source-payload-file.")
        assert args.source_job_id is not None
        source_payload = _read_saved_job_payload(storage, args.source_job_id)
        source_frame = storage.scene_frame_path(
            args.source_job_id,
            args.source_scene_id,
            args.source_revision,
        )
        source_identifier = args.source_job_id
    if not source_frame.is_file():
        raise SystemExit(f"Exact source frame is missing: {source_frame}")
    job = build_acceptance_job(
        source_payload,
        source_scene_id=args.source_scene_id,
        acceptance_job_id=run_id,
    )
    scene = job.scenes[0]
    plans: AcceptancePlans = build_acceptance_plans(job, revision=1)
    raw_base = storage.chunk_video_path(job.job_id, scene.scene_id, 1, 0, 1)
    raw_single = storage.chunk_video_path(job.job_id, scene.scene_id, 2, 0, 1)
    raw_guide = storage.chunk_video_path(job.job_id, scene.scene_id, 3, 0, 1)
    raw_latent = storage.chunk_video_path(job.job_id, scene.scene_id, 1, 1, 1)

    base_stage1 = build_continuation_stage1_workflow(
        job, scene, source_frame, plans.full, plans.base, revision=1, attempt_number=1
    )
    base_stage2 = build_continuation_stage2_workflow(
        job, scene, source_frame, plans.full, plans.base, revision=1, attempt_number=1
    )
    single_stage1 = build_continuation_stage1_workflow(
        job,
        scene,
        run_root / "frames" / "base_final.png",
        plans.diagnostic_plan,
        plans.diagnostic,
        revision=2,
        attempt_number=1,
    )
    single_stage2 = build_continuation_stage2_workflow(
        job,
        scene,
        run_root / "frames" / "base_final.png",
        plans.diagnostic_plan,
        plans.diagnostic,
        revision=2,
        attempt_number=1,
    )
    guide_stage1 = build_continuation_stage1_workflow(
        job,
        scene,
        run_root / "frames" / "base_final.png",
        plans.diagnostic_plan,
        plans.diagnostic,
        revision=3,
        attempt_number=1,
    )
    guide_stage2 = build_continuation_stage2_workflow(
        job,
        scene,
        run_root / "frames" / "base_final.png",
        plans.diagnostic_plan,
        plans.diagnostic,
        revision=3,
        attempt_number=1,
        initial_guide_path=raw_base,
        initial_guide_skip_frames=BASE_DIAGNOSTIC_GUIDE_START_FRAME_INDEX,
    )
    latent_stage1 = build_continuation_stage1_workflow(
        job,
        scene,
        source_frame,
        plans.full,
        plans.latent_overlap,
        revision=1,
        attempt_number=1,
        previous_attempt_number=1,
    )
    latent_stage2 = build_continuation_stage2_workflow(
        job,
        scene,
        source_frame,
        plans.full,
        plans.latent_overlap,
        revision=1,
        attempt_number=1,
        previous_attempt_number=1,
        previous_chunk_path=raw_base,
    )
    workflows = {
        "common_base_stage1": base_stage1.api,
        "common_base_stage2": base_stage2.workflow.api,
        "single_frame_stage1": single_stage1.api,
        "single_frame_stage2": single_stage2.workflow.api,
        "decoded_17_frame_stage1": guide_stage1.api,
        "decoded_17_frame_stage2": guide_stage2.workflow.api,
        "latent_overlap_stage1": latent_stage1.api,
        "latent_overlap_stage2": latent_stage2.workflow.api,
    }
    run_root.mkdir(parents=True, exist_ok=False)
    write_json_atomic(run_root / "acceptance-job.json", dict(job.raw))
    write_json_atomic(
        run_root / "run.json",
        {
            "schema_version": ACCEPTANCE_SCHEMA_VERSION,
            "run_id": run_id,
            "state": "prepared",
            "source": {
                "job_id": source_identifier,
                "scene_id": args.source_scene_id,
                "revision": args.source_revision,
                "frame_path": str(source_frame),
                "frame_sha256": sha256_file(source_frame),
            },
            "acceptance_job_id": job.job_id,
            "case_order": [
                "common_base",
                "single_frame",
                "decoded_17_frame",
                "latent_overlap",
            ],
            "production_rollout": "explicit; never changed by acceptance runner",
        },
    )
    for name, workflow in workflows.items():
        write_json_atomic(run_root / "workflows" / f"{name}.api.json", workflow)

    comfy = ComfyHttpClient(
        base_url=args.comfy_url,
        client_id=f"10MinVideoMaker-acceptance-{run_id}",
    )
    if not comfy.alive():
        raise SystemExit(f"ComfyUI is unavailable at {args.comfy_url}")
    errors = _live_contract_errors(comfy, workflows)
    if errors:
        write_json_atomic(run_root / "contract-errors.json", errors)
        raise SystemExit("Live node-contract validation failed; no prompt was queued.")
    if args.dry_run:
        write_json_atomic(
            run_root / "validation.json",
            {"state": "validated_no_render", "workflow_count": len(workflows)},
        )
        print(f"Acceptance graphs validated without rendering: {run_root}")
        return 0
    _ensure_empty_queue(comfy)
    run_document: dict[str, Any] = {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "run_id": run_id,
        "state": "running",
        "source_job_id": source_identifier,
        "source_scene_id": args.source_scene_id,
        "cases": {},
        "production_rollout": "explicit; human review required",
    }
    try:
        run_document["cases"]["common_base"] = {
            "stage1": _run_graph(
                comfy,
                label="common_base_stage1",
                workflow=base_stage1.api,
                output_node_id=None,
                destination=None,
                timeout_seconds=args.timeout_seconds,
                poll_seconds=args.poll_seconds,
            ),
            "stage2": _run_graph(
                comfy,
                label="common_base_stage2",
                workflow=base_stage2.workflow.api,
                output_node_id=base_stage2.workflow.output_node_id,
                destination=raw_base,
                timeout_seconds=args.timeout_seconds,
                poll_seconds=args.poll_seconds,
            ),
        }
        _extract_frame(raw_base, BASE_FINAL_FRAME_INDEX, run_root / "frames" / "base_final.png")
        run_document["cases"]["common_base"]["metrics"] = {
            "video": _probe_video(raw_base),
            "manual_review": "base reference only",
        }
        write_json_atomic(run_root / "run.json", run_document)

        for (
            name,
            stage1,
            stage2,
            destination,
            guide_start_frame_index,
            guide_frame_count,
        ) in (
            ("single_frame", single_stage1, single_stage2, raw_single, None, None),
            (
                "decoded_17_frame",
                guide_stage1,
                guide_stage2,
                raw_guide,
                BASE_DIAGNOSTIC_GUIDE_START_FRAME_INDEX,
                DIAGNOSTIC_GUIDE_FRAME_COUNT,
            ),
            (
                "latent_overlap",
                latent_stage1,
                latent_stage2,
                raw_latent,
                BASE_LATER_VISIBLE_OVERLAP_START_FRAME_INDEX,
                LATER_VISIBLE_OVERLAP_FRAME_COUNT,
            ),
        ):
            run_document["cases"][name] = {
                "stage1": _run_graph(
                    comfy,
                    label=f"{name}_stage1",
                    workflow=stage1.api,
                    output_node_id=None,
                    destination=None,
                    timeout_seconds=args.timeout_seconds,
                    poll_seconds=args.poll_seconds,
                ),
                "stage2": _run_graph(
                    comfy,
                    label=f"{name}_stage2",
                    workflow=stage2.workflow.api,
                    output_node_id=stage2.workflow.output_node_id,
                    destination=destination,
                    timeout_seconds=args.timeout_seconds,
                    poll_seconds=args.poll_seconds,
                ),
            }
            run_document["cases"][name]["metrics"] = _case_metrics(
                run_root=run_root,
                base_video=raw_base,
                case_name=name,
                case_video=destination,
                guide_start_frame_index=guide_start_frame_index,
                guide_frame_count=guide_frame_count,
            )
            write_json_atomic(run_root / "run.json", run_document)
    except (AcceptanceRunError, ComfyHttpError, OSError, subprocess.SubprocessError) as error:
        run_document["state"] = "failed"
        run_document["error"] = str(error)
        write_json_atomic(run_root / "run.json", run_document)
        print(f"Acceptance run failed; state preserved at {run_root}: {error}", file=sys.stderr)
        return 1
    finally:
        try:
            comfy.free_memory()
        except ComfyHttpError:
            pass

    run_document["state"] = "awaiting_human_review"
    run_document["decision"] = {
        "auto_approved": False,
        "reason": "Human seam, anatomy, identity, and runtime review is required.",
    }
    write_json_atomic(run_root / "run.json", run_document)
    print(f"Acceptance matrix complete; human review required: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
