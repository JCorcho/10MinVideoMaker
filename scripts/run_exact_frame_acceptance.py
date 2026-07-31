"""Run the production exact-last-frame continuation route on a safe fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tenminvideomaker.chunk_assembly import SceneChunkAssembler
from tenminvideomaker.comfy_http import ComfyHttpClient
from tenminvideomaker.continuation_acceptance import build_acceptance_job
from tenminvideomaker.continuation_renderer import ContinuationRenderer
from tenminvideomaker.state_store import PipelineStateStore
from tenminvideomaker.storage import StorageLayout, write_json_atomic
from scripts.run_continuation_acceptance import (
    _image_difference,
    _image_spatial_detail,
    _stage2_checkpoint_spatial_tokens,
)


_RUN_ID = re.compile(r"exact-frame-acceptance-\d{8}-\d{6}\Z")


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render one safe multi-window scene through the actual production "
            "ContinuationRenderer. The normal supervisor is not started."
        )
    )
    parser.add_argument("--source-payload-file", type=Path, required=True)
    parser.add_argument("--source-frame", type=Path, required=True)
    parser.add_argument("--source-scene-id", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=15.0,
        help="Safe accumulation proof duration; 15 seconds creates three 121-frame chunks.",
    )
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--timeout-seconds", type=float, default=21600.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


def main() -> int:
    args = argument_parser().parse_args()
    if not _RUN_ID.fullmatch(args.run_id):
        raise SystemExit("--run-id must use exact-frame-acceptance-YYYYMMDD-HHMMSS.")
    payload_path = args.source_payload_file.resolve(strict=True)
    frame_path = args.source_frame.resolve(strict=True)
    try:
        source_payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Source payload is unreadable: {error}") from error

    storage = StorageLayout.configured()
    storage.ensure()
    run_root = storage.root / "acceptance" / args.run_id
    if run_root.exists():
        raise SystemExit(f"Acceptance run ID already exists: {args.run_id}")
    run_root.mkdir(parents=True)
    job = build_acceptance_job(
        source_payload,
        source_scene_id=args.source_scene_id,
        acceptance_job_id=args.run_id,
        duration_seconds=args.duration_seconds,
    )
    scene = job.scenes[0]
    store = PipelineStateStore(run_root / "acceptance.sqlite3")
    store.claim_job(job)
    comfy = ComfyHttpClient(args.comfy_url)
    running, pending = comfy.queue_counts()
    if running or pending:
        raise SystemExit(
            "ComfyUI queue is not empty; exact-frame acceptance will not overlap another render."
        )
    assembler = SceneChunkAssembler(
        ffmpeg_executable=args.ffmpeg,
        ffprobe_executable=args.ffprobe,
    )
    renderer = ContinuationRenderer(
        store=store,
        storage=storage,
        comfy=comfy,
        assembler=assembler,
        timeout_seconds=args.timeout_seconds,
        max_attempts=2,
    )
    destination = storage.scene_clip_path(job.job_id, scene.scene_id, 1)
    result = renderer.render_scene(
        job,
        scene,
        frame_path,
        destination,
        revision=1,
        deliver_to_discord=False,
    )
    chunk_records = store.chunk_records(job.job_id, scene.scene_id, 1)
    attempts = []
    for record in chunk_records:
        if record.accepted_attempt_number is None:
            raise SystemExit(f"Chunk {record.chunk_index} has no accepted attempt.")
        selected = next(
            attempt
            for attempt in store.chunk_attempts(
                job.job_id,
                scene.scene_id,
                1,
                record.chunk_index,
            )
            if attempt.attempt_number == record.accepted_attempt_number
        )
        attempts.append(
            {
                "chunk_index": record.chunk_index,
                "attempt_number": selected.attempt_number,
                "artifact_hash": selected.artifact_hash,
                "video_path": selected.video_path,
                "source_frame": selected.parameters.get("source_frame"),
            }
        )
    probe = assembler.validate_scene(result.plan, destination)
    metrics_root = run_root / "metrics"
    stage2_tokens: list[list[int]] = []
    seams: list[dict[str, object]] = []
    for selected in attempts:
        stage2_tokens.append(
            _stage2_checkpoint_spatial_tokens(
                storage,
                job_id=job.job_id,
                scene_id=scene.scene_id,
                revision=1,
                chunk_index=int(selected["chunk_index"]),
                attempt_number=int(selected["attempt_number"]),
            )
        )
    for chunk_index in range(1, len(attempts)):
        predecessor = Path(str(attempts[chunk_index - 1]["video_path"]))
        current = Path(str(attempts[chunk_index]["video_path"]))
        source = Path(str(attempts[chunk_index]["source_frame"]["path"]))
        predecessor_final = assembler.extract_frame(
            predecessor,
            120,
            metrics_root / f"seam_{chunk_index:04d}_predecessor_120.png",
        )
        current_zero = assembler.extract_frame(
            current,
            0,
            metrics_root / f"seam_{chunk_index:04d}_current_000.png",
        )
        current_first_new = assembler.extract_frame(
            current,
            1,
            metrics_root / f"seam_{chunk_index:04d}_current_001.png",
        )
        base_detail = _image_spatial_detail(predecessor_final)
        first_new_detail = _image_spatial_detail(current_first_new)
        seams.append(
            {
                "chunk_index": chunk_index,
                "handoff_difference": _image_difference(predecessor_final, source),
                "decoded_frame_zero_difference": _image_difference(source, current_zero),
                "spatial_detail": {
                    "base_boundary": base_detail,
                    "continuation_first_new": first_new_detail,
                    "detail_retention_ratio": round(
                        first_new_detail["laplacian_variance"]
                        / base_detail["laplacian_variance"],
                        6,
                    ),
                },
            }
        )
    report = {
        "schema_version": 2,
        "run_id": args.run_id,
        "state": "awaiting_visual_review",
        "strategy": result.plan.strategy,
        "requested_duration_seconds": args.duration_seconds,
        "plan": result.plan.to_document(),
        "chunks": attempts,
        "stage2_video_spatial_tokens": stage2_tokens,
        "seams": seams,
        "assembled_scene": {
            "path": str(destination),
            "width": probe.width,
            "height": probe.height,
            "fps": str(probe.avg_frame_rate),
            "decoded_video_frames": probe.decoded_video_frames,
        },
    }
    write_json_atomic(run_root / "run.json", report)
    print(f"Exact-frame acceptance complete: {run_root}")
    print(f"Assembled scene: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
