"""Re-run Phase-1 QC evaluations for infrastructure-failed canary candidates."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import URLError

from tenminvideomaker.configuration import load_project_environment
from tenminvideomaker.qc_backend import BackendIdentity, LlamaCppHttpBackend
from tenminvideomaker.qc_config import QualityControlSettings
from tenminvideomaker.qc_controller import Phase1QcController
from tenminvideomaker.qc_contracts import QcCandidateState, QcDecision, QcTier
from tenminvideomaker.qc_llama import LlamaCppProcess
from tenminvideomaker.state_store import PipelineStateStore
from tenminvideomaker.storage import StorageLayout


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_SUFFIX = Path("state") / "pipeline.sqlite3"
_HEALTH_PATH = "/health"


def _backend_is_healthy(settings: QualityControlSettings) -> bool:
    import json
    import urllib.request

    url = f"http://{settings.loopback_host}:{settings.loopback_port}{_HEALTH_PATH}"
    try:
        with urllib.request.urlopen(url, timeout=1.5) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
            return payload.get("status") == "ok"
    except (OSError, URLError, json.JSONDecodeError):
        return False


def _identity_from_evaluation_row(
    qc_settings: QualityControlSettings,
    evaluation_record: Any,
) -> BackendIdentity | None:
    identity_payload = getattr(evaluation_record, "evaluator_identity", None)
    if not isinstance(identity_payload, Mapping):
        return None
    executable_sha256 = identity_payload.get("executable_sha256")
    model_sha256 = identity_payload.get("model_sha256")
    projector_sha256 = identity_payload.get("projector_sha256")
    if not (executable_sha256 and model_sha256 and projector_sha256):
        return None
    return BackendIdentity(
        evaluator_id=identity_payload.get("evaluator_id") or qc_settings.evaluator_id,
        evaluator_version=identity_payload.get("evaluator_version")
        or qc_settings.evaluator_version,
        backend_family=identity_payload.get("backend_family") or qc_settings.backend_family,
        backend_version=identity_payload.get("backend_version") or qc_settings.backend_version,
        executable_path=identity_payload.get("executable_path")
        or str(qc_settings.llama_executable),
        executable_sha256=executable_sha256,
        model_path=identity_payload.get("model_path") or str(qc_settings.model_path),
        model_sha256=model_sha256,
        model_id=str(identity_payload.get("model_id") or qc_settings.model_id),
        quantization=identity_payload.get("quantization") or qc_settings.quantization,
        projector_path=identity_payload.get("projector_path")
        or str(qc_settings.projector_path),
        projector_sha256=projector_sha256,
        projector_precision=identity_payload.get("projector_precision")
        or qc_settings.projector_precision,
        gpu_uuid=str(identity_payload.get("gpu_uuid") or qc_settings.expected_gpu_uuid or ""),
        gpu_name=str(identity_payload.get("gpu_name") or qc_settings.expected_gpu_name or ""),
        effective_args=tuple(),
        effective_config_sha256=qc_settings.effective_sha256(),
        owned_pid=-1,
        stdout_log_path="",
        stderr_log_path="",
        launch_id="",
        started_at="",
        device_telemetry="",
    )


def _identity_for_resume(
    store: PipelineStateStore,
    qc_settings: QualityControlSettings,
    fallback_identity: BackendIdentity | None = None,
) -> BackendIdentity | None:
    snapshot_job_id = store.snapshot().job_id
    if snapshot_job_id is not None:
        for candidate in store.qc_candidates(snapshot_job_id):
            if candidate.state in {
                QcCandidateState.SUPERSEDED,
                QcCandidateState.HOLD_FOR_REVIEW,
            }:
                continue
            for evaluation in store.qc_evaluations(candidate.candidate_id):
                if evaluation.state != "COMPLETE":
                    continue
                identity = _identity_from_evaluation_row(qc_settings, evaluation)
                if identity is not None and _backend_is_healthy(qc_settings):
                    return identity
    if fallback_identity is not None and _backend_is_healthy(qc_settings):
        return fallback_identity
    if (
        qc_settings.llama_executable is not None
        and qc_settings.model_path is not None
        and qc_settings.projector_path is not None
        and qc_settings.expected_executable_sha256 is not None
        and qc_settings.expected_model_sha256 is not None
        and qc_settings.expected_projector_sha256 is not None
    ):
        if not _backend_is_healthy(qc_settings):
            return None
        return BackendIdentity(
            evaluator_id=qc_settings.evaluator_id,
            evaluator_version=qc_settings.evaluator_version,
            backend_family=qc_settings.backend_family,
            backend_version=qc_settings.backend_version,
            executable_path=str(qc_settings.llama_executable),
            executable_sha256=qc_settings.expected_executable_sha256,
            model_path=str(qc_settings.model_path),
            model_sha256=qc_settings.expected_model_sha256,
            model_id=qc_settings.model_id,
            quantization=qc_settings.quantization,
            projector_path=str(qc_settings.projector_path),
            projector_sha256=qc_settings.expected_projector_sha256,
            projector_precision=qc_settings.projector_precision,
            gpu_uuid=qc_settings.expected_gpu_uuid or "",
            gpu_name=qc_settings.expected_gpu_name or "",
            effective_args=(),
            effective_config_sha256=qc_settings.effective_sha256(),
            owned_pid=-1,
            stdout_log_path="",
            stderr_log_path="",
            launch_id="",
            started_at="",
            device_telemetry="",
        )
    return None


def _collect_reference_identity(
    roots: Sequence[Path],
) -> BackendIdentity | None:
    settings = _load_qc_settings()
    for root in roots:
        db_path = root / DATABASE_SUFFIX
        if not db_path.is_file():
            continue
        store = PipelineStateStore(db_path)
        store.initialize()
        snapshot = store.snapshot()
        if snapshot.job_id is None:
            continue
        identity = _identity_for_resume(store, settings)
        if identity is not None:
            return identity
    return None


def _candidate_last_failure(candidate) -> Mapping[str, Any] | None:
    failure = candidate.last_failure
    if failure is None:
        return None
    if isinstance(failure, Mapping):
        return failure
    return None


def _group_infra_failures(store: PipelineStateStore, job_id: str) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for candidate in store.qc_candidates(job_id):
        if candidate.infrastructure_failure_count <= 0:
            continue
        message = ""
        failure = _candidate_last_failure(candidate)
        if failure is not None:
            message = str(failure.get("message", "")).strip()
        if not message:
            message = "[missing failure message]"
        counts[f"{candidate.tier.value}::{message}"] += 1
        if failure is not None:
            detail = failure.get("kind")
            if detail:
                counts[f"{candidate.tier.value}::kind::{detail}"] += 1
    return sorted(
        ((k, v) for k, v in counts.items()),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )


def _group_infra_failure_events(store: PipelineStateStore, job_id: str) -> int:
    return sum(candidate.infrastructure_failure_count for candidate in store.qc_candidates(job_id))


def _candidate_final_decision(store: PipelineStateStore, candidate) -> QcDecision | None:
    for evaluation in reversed(tuple(store.qc_evaluations(candidate.candidate_id))):
        if evaluation.normalized_decision is not None:
            return evaluation.normalized_decision
    if candidate.state == QcCandidateState.ACCEPTED:
        return QcDecision.PASS
    return None


def _collect_db_metrics(store: PipelineStateStore, job_id: str) -> dict[str, Any]:
    totals = {
        "scenes": len({item.scene_id for item in store.scene_records(job_id)}),
        "candidates": 0,
        "original": {"PASS": 0, "FAIL": 0, "UNCERTAIN": 0},
        "A1": {"PASS": 0, "FAIL": 0, "UNCERTAIN": 0},
        "B1": {"PASS": 0, "FAIL": 0, "UNCERTAIN": 0},
        "PASS_PENDING_HUMAN": 0,
        "HOLD_FOR_REVIEW": 0,
        "malformed_responses": 0,
        "refusals": 0,
        "completed_evaluations": 0,
    }
    for candidate in store.qc_candidates(job_id):
        totals["candidates"] += 1
        bucket = candidate.tier.value
        decision = _candidate_final_decision(store, candidate)
        if decision is not None and bucket in totals:
            tier_bucket = totals[bucket]
            if isinstance(tier_bucket, dict):
                key = decision.value
                if key in tier_bucket:
                    tier_bucket[key] += 1
        if candidate.state == QcCandidateState.PASS_PENDING_HUMAN:
            totals["PASS_PENDING_HUMAN"] += 1
        elif candidate.state == QcCandidateState.HOLD_FOR_REVIEW:
            totals["HOLD_FOR_REVIEW"] += 1

        for eval_record in store.qc_evaluations(candidate.candidate_id):
            if eval_record.state != "COMPLETE":
                continue
            totals["completed_evaluations"] += 1
            if eval_record.raw_result is None:
                totals["malformed_responses"] += 1
            elif eval_record.normalized_decision is None:
                totals["malformed_responses"] += 1
            if eval_record.raw_result and "refuse" in eval_record.raw_result.lower():
                totals["refusals"] += 1
    return totals


def _build_targets(store: PipelineStateStore, job_id: str) -> list[Any]:
    targets: list[Any] = []
    for candidate in store.qc_candidates(job_id):
        if candidate.tier not in {QcTier.ORIGINAL, QcTier.A1, QcTier.B1}:
            continue
        if candidate.infrastructure_failure_count <= 0:
            continue
        if candidate.scene_id < 1 or candidate.scene_id > 28:
            continue
        has_complete_decision = False
        for evaluation in store.qc_evaluations(candidate.candidate_id):
            if (
                evaluation.state == "COMPLETE"
                and evaluation.normalized_decision is not None
            ):
                has_complete_decision = True
                break
        if has_complete_decision:
            continue
        if candidate.state != QcCandidateState.PENDING_QC:
            candidate = store.set_qc_candidate_state(
                candidate.candidate_id,
                QcCandidateState.PENDING_QC,
                next_action="retry_evaluation",
            )
        targets.append(candidate)
    return sorted(targets, key=lambda item: item.scene_id)


def _apply_readonly_routing(
    store: PipelineStateStore,
    candidate,
    evaluation: Any,
) -> None:
    decision = evaluation.normalized_decision
    if decision == QcDecision.PASS:
        store.set_qc_candidate_state(
            candidate.candidate_id,
            QcCandidateState.PASS_PENDING_HUMAN,
            next_action="await_human",
        )
        return

    if decision == QcDecision.UNCERTAIN or decision == QcDecision.FAIL:
        store.set_qc_candidate_state(
            candidate.candidate_id,
            QcCandidateState.HOLD_FOR_REVIEW,
            next_action="hold_for_review",
        )
        return

    store.set_qc_candidate_state(
        candidate.candidate_id,
        QcCandidateState.HOLD_FOR_REVIEW,
        next_action="hold_for_review",
    )


def _safe_parse_failure_message(error: Exception) -> str:
    message = str(error).strip()
    if message:
        return message
    return f"{type(error).__name__}"


def _build_controller(
    canary_root: Path,
    qc_settings: QualityControlSettings,
) -> tuple[PipelineStateStore, Phase1QcController]:
    layout = StorageLayout(canary_root)
    runtime_layout = StorageLayout((canary_root / ".runtime" / "phase1-qc-recovery").resolve())
    ffmpeg = os.environ.get("TENMIN_FFMPEG", "ffmpeg")
    ffprobe = os.environ.get("TENMIN_FFPROBE", "ffprobe")
    store = PipelineStateStore(layout.database_path)
    controller = Phase1QcController(
        store=store,
        layout=layout,
        settings=qc_settings,
        backend_factory=lambda: LlamaCppHttpBackend(
            qc_settings,
            LlamaCppProcess(qc_settings, runtime_layout),
        ),
        prompt_root=PROJECT_ROOT / "prompts",
        ffmpeg_command=ffmpeg,
        ffprobe_command=ffprobe,
    )
    return store, controller


def _load_qc_settings() -> QualityControlSettings:
    project_environment = load_project_environment(
        PROJECT_ROOT,
        storage_layout=StorageLayout.configured(),
    )
    os.environ.update(
        {key: value for key, value in project_environment.items() if key.startswith("TENMIN_")}
    )
    return replace(
        QualityControlSettings.from_environment(project_environment),
        quality_control_enabled=True,
        auto_advance_pass=False,
    )


def _print_infra_failures(store: PipelineStateStore, job_id: str, root: Path) -> None:
    totals = _group_infra_failures(store, job_id)
    print(f"Infrastructure failures for {root}:")
    for key, count in totals[:20]:
        print(f"  {count:>4}x {key}")
    print(f"  total infra events: {_group_infra_failure_events(store, job_id)}")


def run_recovery(canary_roots: tuple[Path, ...]) -> dict[str, Any]:
    qc_settings = _load_qc_settings()
    qc_settings.validate_for_start()
    summary: dict[str, Any] = {
        "roots": [],
        "totals_before": {},
        "totals_after": {},
        "failure_groups": {},
    }
    reference_identity = _collect_reference_identity(canary_roots)

    for root in canary_roots:
        if not root.is_dir():
            raise RuntimeError(f"Canary root missing: {root}")

        db_path = root / DATABASE_SUFFIX
        if not db_path.is_file():
            raise RuntimeError(f"Missing QC DB: {db_path}")

        store, controller = _build_controller(root, qc_settings)
        store.initialize()
        snapshot = store.snapshot()
        if snapshot.job_id is None:
            raise RuntimeError(f"{root} has no claimed job.")
        job_id = snapshot.job_id
        job = store.load_job(job_id)
        before = _collect_db_metrics(store, job_id)

        print(f"\n--- {root} ---")
        _print_infra_failures(store, job_id, root)

        targets = _build_targets(store, job_id)
        print(f"Candidates queued for infra-rerun: {len(targets)}")

        evaluated = 0
        started_backend = False
        backend = None
        if targets:
            identity = _identity_for_resume(store, qc_settings, reference_identity)
            try:
                if identity is None:
                    backend = controller.backend_factory()
                    identity = backend.start()
                    started_backend = True
                else:
                    backend = controller.backend_factory()
                for candidate in targets:
                    try:
                        evaluation = controller._evaluate_candidate(
                            candidate,
                            backend,
                            identity,
                        )
                        _apply_readonly_routing(store, candidate, evaluation)
                        evaluated += 1
                    except Exception as error:
                        store.record_qc_infrastructure_failure(
                            candidate.candidate_id,
                            {
                                "kind": type(error).__name__,
                                "reason": _safe_parse_failure_message(error),
                                "message": _safe_parse_failure_message(error),
                            },
                        )
                        print(
                            f"  [{candidate.scene_id}] evaluation retry failed: "
                            f"{type(error).__name__}: {_safe_parse_failure_message(error)}"
                        )
            finally:
                if backend is not None and started_backend:
                    backend.close()
        after = _collect_db_metrics(store, job_id)
        failures_after = _group_infra_failures(store, job_id)
        print(f"Infrastructure-rerun complete for {root}; evaluated {evaluated}.")
        print("Decision totals by tier:")
        print(f"  ORIGINAL PASS/FAIL/UNCERTAIN: {after['original']}")
        print(f"  A1 PASS/FAIL/UNCERTAIN: {after['A1']}")
        print(f"  B1 PASS/FAIL/UNCERTAIN: {after['B1']}")
        print(f"  PASS_PENDING_HUMAN: {after['PASS_PENDING_HUMAN']}")
        print(f"  HOLD_FOR_REVIEW: {after['HOLD_FOR_REVIEW']}")
        print(f"  infra failure events now: {_group_infra_failure_events(store, job_id)}")
        summary["roots"].append(str(root))
        summary["totals_before"][str(root)] = before
        summary["totals_after"][str(root)] = after
        summary["failure_groups"][str(root)] = failures_after[:50]

        # Prime for later roots: keep an identity from any successful evaluator
        # evidence so each run can recover from the same server process.
        if reference_identity is None:
            reference_identity = _identity_for_resume(store, qc_settings, None)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover Phase-1 infrastructure-failed candidates without re-rendering."
        )
    )
    parser.add_argument(
        "--canary-root",
        action="append",
        type=Path,
        required=True,
        help="Canary root containing state/pipeline.sqlite3.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_recovery(tuple(sorted(set(args.canary_root), key=lambda p: str(p))))
    print("\n--- RECOVERY SUMMARY ---")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
