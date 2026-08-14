"""Bounded non-production hardware smoke for the Phase-1 QC trust boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tenminvideomaker.qc_backend import (  # noqa: E402
    HeadlessVideoEvaluator,
    LlamaCppHttpBackend,
    RepairPlannerRequest,
    build_repair_planner_payload,
    load_production_rubric,
    load_repair_planner_prompt,
)
from tenminvideomaker.qc_config import QualityControlSettings  # noqa: E402
from tenminvideomaker.qc_contracts import QcEvidencePolicy  # noqa: E402
from tenminvideomaker.qc_llama import LlamaCppProcess  # noqa: E402
from tenminvideomaker.qc_video import sample_video_frames  # noqa: E402
from tenminvideomaker.storage import StorageLayout  # noqa: E402


def _gpu_line(uuid: str) -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.used,utilization.gpu,pstate",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return next(line.strip() for line in completed.stdout.splitlines() if uuid in line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--sentinel", required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--executable-sha256", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--projector-sha256", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--gpu-name", required=True)
    args = parser.parse_args()
    evidence = args.evidence_root.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    settings = QualityControlSettings(
        quality_control_enabled=True,
        auto_advance_pass=False,
        llama_executable=args.executable.resolve(),
        llama_vendor_root=args.vendor_root.resolve(),
        model_path=args.model.resolve(),
        projector_path=args.projector.resolve(),
        expected_executable_sha256=args.executable_sha256,
        expected_model_sha256=args.model_sha256,
        expected_projector_sha256=args.projector_sha256,
        expected_gpu_uuid=args.gpu_uuid,
        expected_gpu_name=args.gpu_name,
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "media": str(args.media.resolve()),
        "sentinel": args.sentinel,
        "gpu_before": _gpu_line(args.gpu_uuid),
        "success": False,
    }
    process = LlamaCppProcess(settings, StorageLayout(evidence / "owned-runtime"))
    backend = LlamaCppHttpBackend(settings, process)
    identity = None
    try:
        identity = backend.start()
        result["identity"] = {
            "backend_version": identity.backend_version,
            "gpu_uuid": identity.gpu_uuid,
            "gpu_name": identity.gpu_name,
            "device_telemetry": identity.device_telemetry,
            "owned_pid": identity.owned_pid,
            "launch_id": identity.launch_id,
            "started_at": identity.started_at,
            "effective_args": list(identity.effective_args),
            "stdout_log_path": identity.stdout_log_path,
            "stderr_log_path": identity.stderr_log_path,
        }
        result["gpu_loaded"] = _gpu_line(args.gpu_uuid)
        sampled = sample_video_frames(
            args.media,
            target_fps=2.0,
            ffprobe_command="ffprobe",
            ffmpeg_command="ffmpeg",
            temporary_root=evidence / "frames",
        )
        judged = HeadlessVideoEvaluator(
            backend,
            load_production_rubric(
                PROJECT_ROOT / "prompts" / "production_ltx_video_qc_v1.txt"
            ),
            policy=QcEvidencePolicy(),
            frames_per_window=4,
        ).evaluate_sampled(sampled)
        result["qc"] = {
            "decision": judged.normalized.decision.value,
            "raw_result": judged.raw_result,
            "frame_accounting": judged.frame_accounting,
        }
        prompt = load_repair_planner_prompt(
            PROJECT_ROOT / "prompts" / "production_i2v_repair_v1.txt"
        )
        input_hash = hashlib.sha256(b"phase1-independent-request-b").hexdigest()
        request = RepairPlannerRequest(
            job_identity={"job_id": "NONPRODUCTION-SMOKE"},
            scene_identity={"scene_id": 1},
            source_identity={
                "candidate_id": "candidate-smoke",
                "candidate_sha256": "a" * 64,
                "evaluation_id": "evaluation-smoke",
                "repair_input_sha256": input_hash,
            },
            current_i2v_prompt="A plain blue production test card remains still.",
            negative_prompt="No people, no unsafe content.",
            fixed_scene_facts={"subject": "plain blue test card"},
            generation_config={"seed": "1"},
            normalized_qc={"decision": "FAIL", "reason": "synthetic smoke only"},
            suspect_windows=[],
            previous_repairs=[],
            mutable_fields=("i2v.prompt",),
            locked_fields=("all fields except i2v.prompt",),
            repair_input_sha256=input_hash,
            prompt=prompt,
        )
        payload = json.dumps(build_repair_planner_payload(request), sort_keys=True)
        if args.sentinel in payload:
            raise RuntimeError("Independent request B contains request-A sentinel.")
        planned = backend.plan_repair(request)
        leaked = args.sentinel.casefold() in planned.raw_text.casefold()
        result["independent_b"] = {
            "payload_contains_sentinel": False,
            "response_contains_sentinel": leaked,
            "raw_response": planned.raw_text,
        }
        if leaked:
            raise RuntimeError("Independent request B response leaked request-A sentinel.")
        result["success"] = True
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    finally:
        try:
            backend.close()
        except BaseException as error:
            result["close_error"] = f"{type(error).__name__}: {error}"
        time.sleep(3)
        result["gpu_after"] = _gpu_line(args.gpu_uuid)
        result["owned_process_exited"] = process._process is None
        (evidence / "smoke-result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({key: value for key, value in result.items() if key not in {"qc", "independent_b", "traceback"}}, indent=2))
    return int(
        not result.get("success")
        or not result.get("owned_process_exited")
        or bool(result.get("close_error"))
    )


if __name__ == "__main__":
    raise SystemExit(main())
