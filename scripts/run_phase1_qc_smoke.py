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
import socket

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PRODUCTION_STORAGE_ROOT = Path(r"D:\LTX_Supervisor_Storage")
PRIMARY_MAIN_ROOT = Path(
    r"C:\AI\ComfyUI\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI"
    r"\custom_nodes\10MinVideoMaker"
)
PROTECTED_V3_ROOT = Path(
    r"C:\AI\GitWorktrees\10MinVideoMaker-recovery-ltx23-latent-overlap-v3"
)
SMOKE_EVIDENCE_ROOT = PROJECT_ROOT / "test-evidence"

from tenminvideomaker.qc_backend import (  # noqa: E402
    LlamaCppHttpBackend,
    RepairPlannerRequest,
    VisionJudgeRequest,
    build_repair_planner_payload,
    load_production_rubric,
    load_repair_planner_prompt,
)
from tenminvideomaker.qc_config import QualityControlSettings  # noqa: E402
from tenminvideomaker.qc_llama import LlamaCppProcess  # noqa: E402
from tenminvideomaker.qc_video import (  # noqa: E402
    chronological_windows,
    sample_video_frames,
)
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


def _gpu_snapshot(uuid: str, expected_name: str) -> dict[str, object]:
    line = _gpu_line(uuid)
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 5 or parts[0] != uuid or parts[1].casefold() != expected_name.casefold():
        raise RuntimeError("Configured GPU UUID/name did not match nvidia-smi telemetry.")
    return {
        "line": line,
        "uuid": parts[0],
        "name": parts[1],
        "memory_used_mib": int(parts[2]),
        "utilization_percent": int(parts[3]),
        "pstate": parts[4],
    }


def _numeric_compute_processes(output: str, gpu_uuid: str) -> list[str]:
    """Ignore Windows display-client rows whose memory is reported as N/A."""
    return [
        line.strip()
        for line in output.splitlines()
        if gpu_uuid in line
        and line.rsplit(",", 1)[-1].strip().isdigit()
    ]


def hardware_preflight(
    *, gpu_uuid: str, gpu_name: str, host: str, port: int
) -> dict[str, object]:
    """Fail closed if the physical 4080 or dedicated port is not clearly idle."""
    gpu = _gpu_snapshot(gpu_uuid, gpu_name)
    compute = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    matching_compute = _numeric_compute_processes(compute, gpu_uuid)
    try:
        with socket.create_connection((host, port), timeout=0.25):
            port_open = True
    except OSError:
        port_open = False
    reasons = []
    if int(gpu["memory_used_mib"]) > 2048:
        reasons.append("configured GPU uses more than 2048 MiB")
    if int(gpu["utilization_percent"]) > 10:
        reasons.append("configured GPU utilization exceeds 10 percent")
    if matching_compute:
        reasons.append("configured GPU has an existing compute process")
    if port_open:
        reasons.append("dedicated loopback port is already open")
    result = {
        "safe": not reasons,
        "gpu": gpu,
        "matching_compute_processes": matching_compute,
        "loopback_port_open": port_open,
        "reasons": reasons,
    }
    if reasons:
        raise RuntimeError("Hardware smoke preflight failed: " + "; ".join(reasons))
    return result


def _pid_exists(pid: int) -> bool:
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {int(pid)}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=10,
    )
    return completed.returncode == 0


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def ensure_nonproduction_smoke_paths(evidence_root: Path, media: Path) -> tuple[Path, Path]:
    """Reject production/protected paths before the smoke creates any output."""
    evidence = evidence_root.resolve()
    source = media.resolve()
    if not _is_below(evidence, SMOKE_EVIDENCE_ROOT):
        raise ValueError(
            f"Smoke evidence root must remain below {SMOKE_EVIDENCE_ROOT}."
        )
    if _is_below(source, PRODUCTION_STORAGE_ROOT):
        raise ValueError("Smoke media must never come from production storage.")
    if _is_below(source, PRIMARY_MAIN_ROOT) or _is_below(source, PROTECTED_V3_ROOT):
        raise ValueError("Smoke media must not come from a protected repository.")
    if not source.is_file():
        raise ValueError("Smoke media must be an existing non-production file.")
    return evidence, source


def _probe_request(prompt, *, label: str, input_hash: str, sentinel: str | None):
    facts = {"subject": "plain blue non-production test card", "probe": label}
    current_prompt = "A plain blue production test card remains still."
    if sentinel is not None:
        facts["unique_request_a_sentinel"] = sentinel
        current_prompt += " Diagnostic token: " + sentinel
    return RepairPlannerRequest(
        job_identity={"job_id": "NONPRODUCTION-SMOKE"},
        scene_identity={"scene_id": 1},
        source_identity={
            "candidate_id": f"candidate-smoke-{label.lower()}",
            "candidate_sha256": hashlib.sha256(label.encode()).hexdigest(),
            "evaluation_id": f"evaluation-smoke-{label.lower()}",
            "repair_input_sha256": input_hash,
            "source_revision": 1,
            "source_document_sha256": hashlib.sha256(
                ("source-document-" + label).encode()
            ).hexdigest(),
        },
        current_i2v_prompt=current_prompt,
        negative_prompt="No people, no unsafe content.",
        fixed_scene_facts=facts,
        generation_config={"seed": "1" if label == "A" else "2"},
        normalized_qc={"decision": "FAIL", "reason": "synthetic smoke only"},
        suspect_windows=[],
        previous_repairs=[],
        mutable_fields=("i2v.prompt",),
        locked_fields=("all fields except i2v.prompt",),
        repair_input_sha256=input_hash,
        prompt=prompt,
    )


def run_context_isolation_probe(backend, prompt, sentinel: str) -> dict[str, object]:
    """Send sentinel-bearing A, then an independently constructed request B."""
    if not sentinel or len(sentinel) < 12:
        raise ValueError("Context-isolation sentinel must be unique and non-trivial.")
    hash_a = hashlib.sha256(("request-a|" + sentinel).encode()).hexdigest()
    hash_b = hashlib.sha256(b"independent-request-b").hexdigest()
    request_a = _probe_request(
        prompt, label="A", input_hash=hash_a, sentinel=sentinel
    )
    request_b = _probe_request(
        prompt, label="B", input_hash=hash_b, sentinel=None
    )
    payload_a = json.dumps(build_repair_planner_payload(request_a), sort_keys=True)
    payload_b = json.dumps(build_repair_planner_payload(request_b), sort_keys=True)
    if sentinel not in payload_a:
        raise RuntimeError("Request A did not contain its context-leak sentinel.")
    if sentinel in payload_b:
        raise RuntimeError("Independent request B contains request-A sentinel.")
    response_a = backend.plan_repair(request_a)
    response_b = backend.plan_repair(request_b)
    leaked = sentinel.casefold() in response_b.raw_text.casefold()
    if leaked:
        raise RuntimeError("Independent request B response leaked request-A sentinel.")
    return {
        "request_a_payload_contains_sentinel": True,
        "request_b_payload_contains_sentinel": False,
        "request_a_payload_sha256": hashlib.sha256(payload_a.encode()).hexdigest(),
        "request_b_payload_sha256": hashlib.sha256(payload_b.encode()).hexdigest(),
        "response_a": response_a.raw_text,
        "response_b": response_b.raw_text,
        "response_b_contains_sentinel": False,
    }


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
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required explicit authorization to launch the bounded worker.",
    )
    args = parser.parse_args()
    evidence, media = ensure_nonproduction_smoke_paths(
        args.evidence_root, args.media
    )
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
    preflight = hardware_preflight(
        gpu_uuid=args.gpu_uuid,
        gpu_name=args.gpu_name,
        host=settings.loopback_host,
        port=settings.loopback_port,
    )
    settings.validate_for_start()
    if not args.execute:
        print(json.dumps({"preflight": preflight, "launched": False}, indent=2))
        return 0
    result: dict[str, object] = {
        "schema_version": 1,
        "media": str(media),
        "sentinel": args.sentinel,
        "gpu_before": _gpu_line(args.gpu_uuid),
        "preflight": preflight,
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
            media,
            target_fps=2.0,
            ffprobe_command="ffprobe",
            ffmpeg_command="ffmpeg",
            temporary_root=evidence / "frames",
        )
        windows = chronological_windows(sampled, frames_per_window=4)
        if not windows:
            raise RuntimeError("Smoke media produced no judge window.")
        judged = backend.evaluate(
            VisionJudgeRequest.from_window(
                windows[0],
                rubric=load_production_rubric(
                    PROJECT_ROOT / "prompts" / "production_ltx_video_qc_v1.txt"
                ),
            )
        )
        if judged.response.parse_status != "parsed" or judged.response.decision is None:
            raise RuntimeError("Real judge smoke did not return valid structured QC JSON.")
        result["qc"] = {
            "decision": (
                None if judged.response.decision is None
                else judged.response.decision.value
            ),
            "raw_result": judged.response.raw_text,
            "structured_response": judged.response.to_dict(),
            "requests_sent": 1,
        }
        prompt = load_repair_planner_prompt(
            PROJECT_ROOT / "prompts" / "production_i2v_repair_v1.txt"
        )
        result["context_isolation"] = run_context_isolation_probe(
            backend, prompt, args.sentinel
        )
        result["success"] = True
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    finally:
        try:
            backend.close()
        except BaseException as error:
            result["close_error"] = f"{type(error).__name__}: {error}"
        deadline = time.monotonic() + settings.shutdown_timeout_seconds
        before_mib = int(preflight["gpu"]["memory_used_mib"])
        after = _gpu_snapshot(args.gpu_uuid, args.gpu_name)
        while int(after["memory_used_mib"]) > before_mib + 256 and time.monotonic() < deadline:
            time.sleep(1)
            after = _gpu_snapshot(args.gpu_uuid, args.gpu_name)
        result["gpu_after"] = after["line"]
        result["vram_returned"] = int(after["memory_used_mib"]) <= before_mib + 256
        owned_pid = None if identity is None else identity.owned_pid
        result["owned_process_exited"] = (
            process._process is None
            and (owned_pid is None or not _pid_exists(owned_pid))
        )
        (evidence / "smoke-result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({key: value for key, value in result.items() if key not in {"qc", "context_isolation", "traceback"}}, indent=2))
    return int(
        not result.get("success")
        or not result.get("owned_process_exited")
        or not result.get("vram_returned")
        or bool(result.get("close_error"))
    )


if __name__ == "__main__":
    raise SystemExit(main())
