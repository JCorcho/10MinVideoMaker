"""Verify a real adaptive-QC canary without rendering or external effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenminvideomaker.gui_service import SupervisorController
from tenminvideomaker.assets import predictable_lora_filename
from tenminvideomaker.contracts import effective_i2v_loras, lora_identity
from tenminvideomaker.qc_config import QualityControlSettings
from tenminvideomaker.qc_contracts import QcArtifactStage
from tenminvideomaker.review import validate_scene_edit
from tenminvideomaker.state_store import PipelineStateStore
from tenminvideomaker.storage import StorageLayout
from tenminvideomaker.workflow_builder import build_i2v_final_api_workflow


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    layout = StorageLayout(args.canary_root.resolve())
    os.environ["TENMIN_STORAGE_ROOT"] = str(layout.root)
    store = PipelineStateStore(layout.database_path)
    job = store.load_job(args.job_id)
    candidates = store.qc_candidates(args.job_id)
    originals = [item for item in candidates if item.parent_candidate_id is None]
    adaptive = [item for item in candidates if item.parent_candidate_id is not None]
    rendered_adaptive = [
        item for item in adaptive
        if item.source_video_sha256 and Path(item.source_video_path).is_file()
    ]
    decisions = {
        item.candidate_id: [
            evidence.normalized_decision.value
            for evidence in store.qc_evaluations(item.candidate_id)
            if evidence.normalized_decision is not None
        ]
        for item in originals
    }
    recipe_hashes = [item.recipe_sha256 for item in candidates if item.recipe_sha256]
    stage1_files = list(args.canary_root.rglob("stage1_handoff.safetensors"))
    audio_files = list(args.canary_root.rglob("stage1_audio.safetensors"))
    final_graph_proven = False
    if rendered_adaptive:
        selected = rendered_adaptive[0]
        revision = next(
            item for item in store.scene_revisions(args.job_id, selected.scene_id)
            if item.revision == selected.revision
        )
        validated = validate_scene_edit(job, selected.scene_id, revision.parameters)
        resolved = {
            f"i2v:{lora_identity(lora)}": predictable_lora_filename(lora.name)
            for lora in effective_i2v_loras(validated.job, validated.scene)
        }
        graph = build_i2v_final_api_workflow(
            validated.job, validated.scene, revision.frame_path,
            resolved,
            overrides=validated.workflow, revision=selected.revision,
        ).api
        types = [node["class_type"] for node in graph.values()]
        final_graph_proven = (
            types.count("10MinVideoMaker_LoadChunkLatent") == 2
            and "LTXVLatentUpsamplerTiled" in types
            and types.count("SamplerCustom") == 1
            and "10MinVideoMaker_SaveChunkLatent" not in types
        )
    supervisor = SimpleNamespace(
        store=store,
        qc_controller=SimpleNamespace(
            settings=QualityControlSettings(quality_control_enabled=True)
        ),
    )
    status = SupervisorController(supervisor, layout).qc_progress_document()
    proof = {
        "schema_version": "adaptive_qc_canary_proof_v1",
        "job_id": args.job_id,
        "canary_root": str(args.canary_root.resolve()),
        "two_original_drafts": len(originals) == 2 and all(
            item.artifact_stage == QcArtifactStage.DRAFT
            and Path(item.source_video_path).is_file()
            and sha256(Path(item.source_video_path)) == item.source_video_sha256
            for item in originals
        ),
        "qwen_evaluated_both": len(originals) == 2 and all(decisions[item.candidate_id] for item in originals),
        "adaptive_plan_created_for_both": len(adaptive) >= 2,
        "adaptive_cycle_rendered": bool(rendered_adaptive),
        "recipes_unique": len(recipe_hashes) == len(set(recipe_hashes)),
        "attempt_ledger_count": sum(len(store.adaptive_attempts(args.job_id, item.scene_id)) for item in originals),
        "start_frame_route": next((item.strategy for item in adaptive if item.scene_id == 2), None),
        "temporal_route": next((item.strategy for item in adaptive if item.scene_id == 10), None),
        "isolated_video_handoffs": len(stage1_files),
        "isolated_audio_handoffs": len(audio_files),
        "final_workflow_proven": final_graph_proven,
        "status_projection": {
            key: status.get(key) for key in (
                "active_scene_id", "active_attempt_number", "active_artifact_stage",
                "active_authority_tier", "active_strategy", "stage", "last_activity_at",
            )
        },
    }
    required = (
        "two_original_drafts", "qwen_evaluated_both", "adaptive_plan_created_for_both",
        "adaptive_cycle_rendered", "recipes_unique", "final_workflow_proven",
    )
    proof["passed"] = all(proof[key] for key in required)
    output = args.canary_root / "adaptive-qc-proof.json"
    output.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(proof, sort_keys=True))
    return 0 if proof["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
