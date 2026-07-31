"""Generate one non-explicit cached T2I frame for continuation calibration."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tenminvideomaker.comfy_http import ComfyHttpClient, ComfyHttpError
from tenminvideomaker.contracts import lora_identity, parse_job_payload
from tenminvideomaker.storage import StorageLayout, write_json_atomic
from tenminvideomaker.workflow_builder import (
    build_t2i_api_workflow,
    validate_against_object_info,
)

DEFAULT_PAYLOAD = PROJECT_ROOT / "examples" / "safe_continuation_source.json"
DEFAULT_LOCAL_LORA = "Tsunade_-_Naruto_-_Anima_LORA.safetensors"


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a safe, project-owned T2I source for LTX continuation tests."
    )
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--revision", type=int, default=1)
    parser.add_argument("--local-lora-filename", default=DEFAULT_LOCAL_LORA)
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser


def _object_info(comfy_url: str) -> dict[str, object]:
    try:
        with urlopen(comfy_url.rstrip("/") + "/object_info", timeout=30) as response:
            document = json.load(response)
    except (OSError, json.JSONDecodeError) as error:
        raise ComfyHttpError(f"Could not read live ComfyUI node contracts: {error}") from error
    if not isinstance(document, dict):
        raise ComfyHttpError("ComfyUI returned invalid node contracts.")
    return document


def main() -> int:
    args = argument_parser().parse_args()
    if args.revision < 1:
        raise SystemExit("--revision must be positive.")
    if args.timeout_seconds <= 0 or args.poll_seconds <= 0:
        raise SystemExit("--timeout-seconds and --poll-seconds must be positive.")
    try:
        raw = json.loads(args.payload.resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Safe source payload is unreadable: {error}") from error
    if not isinstance(raw, dict):
        raise SystemExit("Safe source payload must be a JSON object.")
    job_id = args.job_id or (
        "safe-continuation-source-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    )
    raw["job_id"] = job_id
    job = parse_job_payload(raw)
    if len(job.scenes) != 1:
        raise SystemExit("Safe source payload must contain exactly one scene.")
    scene = job.scenes[0]
    storage = StorageLayout.configured()
    storage.ensure()
    frame_path = storage.scene_frame_path(job.job_id, scene.scene_id, args.revision)
    source_root = storage.root / "acceptance" / job.job_id
    payload_path = source_root / "source-job.json"
    if frame_path.exists() or source_root.exists():
        raise SystemExit(f"Safe source job already exists: {job.job_id}")

    resolved = {lora_identity(job.character.lora): args.local_lora_filename}
    build = build_t2i_api_workflow(
        job,
        scene,
        resolved,
        revision=args.revision,
    )
    comfy = ComfyHttpClient(
        base_url=args.comfy_url,
        client_id=f"10MinVideoMaker-safe-source-{job.job_id}",
    )
    if not comfy.alive():
        raise SystemExit(f"ComfyUI is unavailable at {args.comfy_url}")
    running, pending = comfy.queue_counts()
    if running or pending:
        raise SystemExit(
            f"ComfyUI queue is not empty ({running} running, {pending} pending)."
        )
    errors = validate_against_object_info(build.api, _object_info(args.comfy_url))
    if errors:
        source_root.mkdir(parents=True, exist_ok=False)
        write_json_atomic(source_root / "contract-errors.json", list(errors))
        raise SystemExit("Safe T2I graph failed live contract validation; no prompt queued.")

    source_root.mkdir(parents=True, exist_ok=False)
    write_json_atomic(payload_path, raw)
    write_json_atomic(source_root / "workflow.api.json", build.api)
    try:
        comfy.free_memory()
        prompt_id = comfy.queue_prompt(build.api)
        write_json_atomic(source_root / "run.json", {"state": "running", "prompt_id": prompt_id})
        comfy.wait_for_prompt(
            prompt_id,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
        if not frame_path.is_file() or frame_path.stat().st_size < 1:
            raise ComfyHttpError(f"T2I prompt completed without the expected frame: {frame_path}")
        write_json_atomic(
            source_root / "run.json",
            {
                "state": "complete",
                "prompt_id": prompt_id,
                "payload_path": str(payload_path),
                "frame_path": str(frame_path),
            },
        )
    except (ComfyHttpError, OSError) as error:
        write_json_atomic(source_root / "run.json", {"state": "failed", "error": str(error)})
        raise SystemExit(f"Safe source generation failed: {error}") from error
    finally:
        try:
            comfy.free_memory()
        except ComfyHttpError:
            pass

    print(f"Safe source payload: {payload_path}")
    print(f"Safe cached frame: {frame_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
