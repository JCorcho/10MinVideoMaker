"""Validate representative continuation prompts against live ComfyUI contracts.

This tool reads ``/object_info`` only. It never queues or renders a prompt.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tenminvideomaker.comfy_http import ComfyHttpClient, ComfyHttpError
from tenminvideomaker.continuation import build_scene_frame_plan
from tenminvideomaker.continuation_workflow import (
    build_assembled_scene_delivery_workflow,
    build_continuation_decode_workflow,
    build_continuation_stage1_workflow,
    build_continuation_stage2_workflow,
)
from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.workflow_builder import validate_against_object_info


def _representative_job():
    source = json.loads(
        (PROJECT_ROOT / "examples" / "example_job.json").read_text(encoding="utf-8")
    )
    source = deepcopy(source)
    source["job_id"] = "no-render-continuation-validation"
    source["total_estimated_sec"] = 30
    source["scenes"][0]["estimated_sec"] = 30
    source["scenes"][0]["i2v"].pop("segments", None)
    return parse_job_payload(source)


def build_representative_workflows() -> dict[str, dict[str, dict[str, Any]]]:
    """Build initial, middle, final, and delivery prompts without executing them."""
    job = _representative_job()
    scene = job.scenes[0]
    plan = build_scene_frame_plan(
        job_id=job.job_id,
        scene_id=scene.scene_id,
        revision=1,
        requested_duration_seconds=scene.estimated_sec,
        base_seed=scene.i2v.seed,
        fallback_prompt=scene.i2v.prompt,
        fallback_negative=scene.i2v.negative,
        continuity=scene.i2v.continuity,
        raw_segments=scene.i2v.segments,
    )
    if plan.chunk_count < 3:
        raise RuntimeError("Representative continuation plan did not create three chunks.")

    frame_path = Path(
        r"D:\LTX_Supervisor_Storage\validation\starting_frame.png"
    )
    initial = plan.chunks[0]
    later = plan.chunks[1]
    final = plan.chunks[-1]

    def stage1(chunk):
        return build_continuation_stage1_workflow(
            job,
            scene,
            frame_path,
            plan,
            chunk,
            revision=1,
            attempt_number=1,
            previous_attempt_number=None if chunk.is_initial else 1,
        ).api

    def stage2(chunk):
        previous_path = (
            None
            if chunk.is_initial
            else Path(
                r"D:\LTX_Supervisor_Storage\validation"
                rf"\chunk_{chunk.index - 1:04d}.mkv"
            )
        )
        return build_continuation_stage2_workflow(
            job,
            scene,
            frame_path,
            plan,
            chunk,
            revision=1,
            attempt_number=1,
            previous_attempt_number=None if chunk.is_initial else 1,
            previous_chunk_path=previous_path,
        ).workflow.api

    decoded_guide = build_continuation_stage2_workflow(
        job,
        scene,
        frame_path,
        plan,
        initial,
        revision=1,
        attempt_number=1,
        initial_guide_path=Path(
            r"D:\LTX_Supervisor_Storage\validation\decoded-guide.mkv"
        ),
        initial_guide_skip_frames=96,
    ).workflow.api

    delivery = build_assembled_scene_delivery_workflow(
        job,
        scene,
        Path(r"D:\LTX_Supervisor_Storage\validation\assembled_scene.mp4"),
        "https://discord.com/api/webhooks/no-render/validation",
    ).api
    decode = build_continuation_decode_workflow(
        job,
        scene,
        plan,
        later,
        revision=1,
        attempt_number=1,
    ).api
    return {
        "stage1_initial": stage1(initial),
        "stage1_later": stage1(later),
        "stage1_final": stage1(final),
        "stage2_initial": stage2(initial),
        "stage2_decoded_guide": decoded_guide,
        "stage2_later": stage2(later),
        "stage2_final": stage2(final),
        "decode": decode,
        "delivery": delivery,
    }


def validate_live_workflows(
    comfy: ComfyHttpClient,
    *,
    workflows: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return contract errors after read-only live node-schema inspection."""
    built = dict(workflows or build_representative_workflows())
    node_types = sorted(
        {
            node["class_type"]
            for workflow in built.values()
            for node in workflow.values()
            if isinstance(node.get("class_type"), str)
        }
    )
    object_info: dict[str, Mapping[str, Any]] = {}
    for node_type in node_types:
        document = comfy.object_info(node_type)
        node = document.get(node_type)
        if isinstance(node, Mapping):
            object_info[node_type] = node

    errors: dict[str, tuple[str, ...]] = {}
    for name, workflow in built.items():
        workflow_errors = validate_against_object_info(workflow, object_info)
        if workflow_errors:
            errors[name] = workflow_errors
    return errors


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build representative continuation prompts and validate them against "
            "live ComfyUI /object_info. No prompt is queued."
        )
    )
    parser.add_argument(
        "--comfy-url",
        default="http://127.0.0.1:8188",
        help="Local ComfyUI HTTP base URL.",
    )
    return parser


def main() -> int:
    args = argument_parser().parse_args()
    try:
        workflows = build_representative_workflows()
        errors = validate_live_workflows(
            ComfyHttpClient(base_url=args.comfy_url),
            workflows=workflows,
        )
    except (ComfyHttpError, OSError, RuntimeError, ValueError) as error:
        print(f"Continuation no-render validation could not run: {error}", file=sys.stderr)
        return 2

    if errors:
        for name, messages in errors.items():
            print(f"{name}:", file=sys.stderr)
            for message in messages:
                print(f"  - {message}", file=sys.stderr)
        print(
            "No prompt was queued. Fix the reported live-contract mismatches.",
            file=sys.stderr,
        )
        return 1

    node_count = len(
        {
            node["class_type"]
            for workflow in workflows.values()
            for node in workflow.values()
        }
    )
    print(
        f"Validated {len(workflows)} continuation workflows against "
        f"{node_count} live node contracts. No prompt was queued."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
