"""Blind four-video gate through the actual Phase-1 production QC evaluator."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tenminvideomaker.qc_backend import (  # noqa: E402
    HeadlessVideoEvaluator,
    LlamaCppHttpBackend,
    load_production_rubric,
)
from tenminvideomaker.qc_config import QualityControlSettings  # noqa: E402
from tenminvideomaker.qc_contracts import QcDecision, QcEvidencePolicy  # noqa: E402
from tenminvideomaker.qc_llama import LlamaCppProcess  # noqa: E402
from tenminvideomaker.qc_video import sample_video_frames  # noqa: E402
from tenminvideomaker.storage import (  # noqa: E402
    DEFAULT_STORAGE_ROOT,
    STORAGE_ENVIRONMENT_KEY,
    StorageLayout,
    write_immutable_json,
)

PRODUCTION_STORAGE_ROOT = DEFAULT_STORAGE_ROOT.resolve()
PROMPT_PATH = PROJECT_ROOT / "prompts" / "production_ltx_video_qc_v1.txt"


@dataclass(frozen=True)
class BlindSample:
    """The only sample data allowed to cross into preprocessing/inference."""

    sample_id: str
    path: Path


@dataclass(frozen=True)
class LabeledSample:
    blind: BlindSample
    label: str


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _forbidden_roots(environment: Mapping[str, str] | None = None) -> tuple[Path, ...]:
    values = os.environ if environment is None else environment
    configured = values.get(STORAGE_ENVIRONMENT_KEY, "").strip()
    roots = {PRODUCTION_STORAGE_ROOT}
    if configured:
        roots.add(Path(configured).expanduser().resolve())
    return tuple(sorted(roots, key=lambda item: str(item).casefold()))


def validate_benchmark_paths(
    samples: Sequence[LabeledSample],
    evidence_root: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> Path:
    forbidden = _forbidden_roots(environment)
    evidence = evidence_root.expanduser().resolve()
    if any(_is_below(evidence, root) for root in forbidden):
        raise ValueError("Benchmark evidence must remain outside production storage.")
    for sample in samples:
        source = sample.blind.path.expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"Benchmark media does not exist: {sample.blind.sample_id}.")
        if any(_is_below(source, root) for root in forbidden):
            raise ValueError(
                f"Benchmark media {sample.blind.sample_id} is inside production storage."
            )
    return evidence


def load_manifest(path: Path) -> tuple[LabeledSample, ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Benchmark manifest is unreadable or invalid JSON.") from error
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ValueError("Benchmark manifest schema_version must be 1.")
    raw_samples = value.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) != 4:
        raise ValueError("Benchmark manifest must contain exactly four samples.")
    samples: list[LabeledSample] = []
    identifiers: set[str] = set()
    for raw in raw_samples:
        if not isinstance(raw, Mapping):
            raise ValueError("Each benchmark sample must be an object.")
        sample_id = raw.get("sample_id")
        label = raw.get("label")
        media = raw.get("path")
        if (
            not isinstance(sample_id, str)
            or not sample_id.strip()
            or sample_id in identifiers
            or not isinstance(media, str)
            or label not in {"BAD", "GOOD"}
        ):
            raise ValueError("Benchmark sample identity, path, or label is invalid.")
        identifiers.add(sample_id)
        samples.append(
            LabeledSample(
                BlindSample(sample_id, Path(media).expanduser().resolve()),
                str(label),
            )
        )
    if [item.label for item in samples].count("BAD") != 3:
        raise ValueError("Benchmark manifest must contain exactly three BAD labels.")
    if [item.label for item in samples].count("GOOD") != 1:
        raise ValueError("Benchmark manifest must contain exactly one GOOD label.")
    return tuple(samples)


def deterministic_order(
    samples: Sequence[LabeledSample], seed: int
) -> tuple[LabeledSample, ...]:
    ordered = list(samples)
    random.Random(seed).shuffle(ordered)
    return tuple(ordered)


def _production_evaluator(backend: Any, settings: QualityControlSettings):
    return HeadlessVideoEvaluator(
        backend,
        load_production_rubric(PROMPT_PATH),
        policy=QcEvidencePolicy(
            settings.minimum_error_severity,
            settings.minimum_error_confidence,
            settings.minimum_strong_windows,
        ),
        frames_per_window=settings.frames_per_window,
    )


def _infer_one(
    sample: BlindSample,
    *,
    backend: Any,
    settings: QualityControlSettings,
    evidence_root: Path,
    sampler: Callable[..., Any],
    evaluator_factory: Callable[[Any, QualityControlSettings], Any],
    ffprobe_command: str,
    ffmpeg_command: str,
) -> dict[str, Any]:
    """Evaluate a blind sample; this function has no label parameter or access."""
    started = time.monotonic()
    sample_root = evidence_root / "samples" / sample.sample_id
    sampled = sampler(
        sample.path,
        target_fps=settings.sampling_fps,
        ffprobe_command=ffprobe_command,
        ffmpeg_command=ffmpeg_command,
        temporary_root=sample_root / "frames",
    )
    expected_preprocessing = settings.effective_document()["sampling"]["preprocessing"]
    if dict(sampled.preprocessing) != expected_preprocessing:
        raise RuntimeError("Sampled frames do not match production preprocessing identity.")
    result = evaluator_factory(backend, settings).evaluate_sampled(sampled)
    decision = result.normalized.decision.value
    inference = {
        "schema_version": 1,
        "sample_id": sample.sample_id,
        "source_path": str(sample.path),
        "source_sha256": hashlib.sha256(sample.path.read_bytes()).hexdigest(),
        "normalized_decision": decision,
        "normalized": result.normalized.to_dict(),
        "frame_accounting": result.frame_accounting,
        "raw_result": result.raw_result,
        "runtime_seconds": time.monotonic() - started,
    }
    write_immutable_json(sample_root / "inference.json", inference)
    return inference


def _score(inferences: Sequence[Mapping[str, Any]], labels: Mapping[str, str]) -> dict[str, Any]:
    counts = {
        "bad_caught": 0,
        "bad_missed": 0,
        "good_accepted": 0,
        "good_rejected_or_uncertain": 0,
        "infrastructure_or_malformed": 0,
    }
    rows = []
    for inference in inferences:
        sample_id = str(inference["sample_id"])
        label = labels[sample_id]
        decision = inference.get("normalized_decision")
        error = inference.get("error")
        if error or decision not in {item.value for item in QcDecision}:
            counts["infrastructure_or_malformed"] += 1
        elif label == "BAD" and decision == QcDecision.FAIL.value:
            counts["bad_caught"] += 1
        elif label == "BAD":
            counts["bad_missed"] += 1
        elif decision == QcDecision.PASS.value:
            counts["good_accepted"] += 1
        else:
            counts["good_rejected_or_uncertain"] += 1
        rows.append(
            {
                "sample_id": sample_id,
                "label": label,
                "normalized_decision": decision,
                "error": error,
            }
        )
    passed = counts == {
        "bad_caught": 3,
        "bad_missed": 0,
        "good_accepted": 1,
        "good_rejected_or_uncertain": 0,
        "infrastructure_or_malformed": 0,
    }
    return {**counts, "passed": passed, "samples": rows}


def run_microbenchmark(
    *,
    samples: Sequence[LabeledSample],
    evidence_root: Path,
    benchmark_seed: int,
    settings: QualityControlSettings,
    backend_factory: Callable[[], Any] | None = None,
    sampler: Callable[..., Any] = sample_video_frames,
    evaluator_factory: Callable[[Any, QualityControlSettings], Any] = _production_evaluator,
    ffprobe_command: str = "ffprobe",
    ffmpeg_command: str = "ffmpeg",
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    evidence = validate_benchmark_paths(samples, evidence_root, environment=environment)
    settings.validate_for_start()
    if evidence.exists() and any(evidence.iterdir()):
        raise ValueError("Benchmark evidence root must be new or empty.")
    evidence.mkdir(parents=True, exist_ok=True)
    ordered = deterministic_order(samples, benchmark_seed)
    labels = {item.blind.sample_id: item.label for item in ordered}
    run_started = time.monotonic()
    backend = (
        backend_factory()
        if backend_factory is not None
        else LlamaCppHttpBackend(
            settings,
            LlamaCppProcess(settings, StorageLayout(evidence / "owned-runtime")),
        )
    )
    identity = None
    inferences: list[dict[str, Any]] = []
    close_error = None
    try:
        identity = backend.start()
        for labeled in ordered:
            try:
                inference = _infer_one(
                    labeled.blind,
                    backend=backend,
                    settings=settings,
                    evidence_root=evidence,
                    sampler=sampler,
                    evaluator_factory=evaluator_factory,
                    ffprobe_command=ffprobe_command,
                    ffmpeg_command=ffmpeg_command,
                )
            except BaseException as error:
                inference = {
                    "schema_version": 1,
                    "sample_id": labeled.blind.sample_id,
                    "normalized_decision": None,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
                write_immutable_json(
                    evidence / "samples" / labeled.blind.sample_id / "inference.json",
                    inference,
                )
            inferences.append(inference)
    finally:
        try:
            backend.close()
        except BaseException as error:
            close_error = f"{type(error).__name__}: {error}"
    score = _score(inferences, labels)
    if close_error:
        score["infrastructure_or_malformed"] += 1
        score["passed"] = False
    report = {
        "schema_version": 1,
        "benchmark_seed": benchmark_seed,
        "blind_order": [item.blind.sample_id for item in ordered],
        "configuration": settings.effective_document(),
        "configuration_sha256": settings.effective_sha256(),
        "prompt_sha256": load_production_rubric(PROMPT_PATH).sha256,
        "backend_identity": None if identity is None else asdict(identity),
        "score": score,
        "runtime_seconds": time.monotonic() - run_started,
        "backend_close_error": close_error,
    }
    write_immutable_json(evidence / "benchmark-result.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--benchmark-seed", type=int, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required explicit authorization to launch the owned llama backend.",
    )
    args = parser.parse_args(argv)
    samples = load_manifest(args.manifest.resolve())
    evidence = validate_benchmark_paths(samples, args.evidence_root)
    order = [
        item.blind.sample_id
        for item in deterministic_order(samples, args.benchmark_seed)
    ]
    if not args.execute:
        print(
            json.dumps(
                {
                    "validated": True,
                    "launched": False,
                    "benchmark_seed": args.benchmark_seed,
                    "blind_order": order,
                    "evidence_root": str(evidence),
                },
                indent=2,
            )
        )
        return 0
    settings = QualityControlSettings.from_environment()
    report = run_microbenchmark(
        samples=samples,
        evidence_root=evidence,
        benchmark_seed=args.benchmark_seed,
        settings=settings,
        ffprobe_command=os.environ.get("TENMIN_FFPROBE", "ffprobe"),
        ffmpeg_command=os.environ.get("TENMIN_FFMPEG", "ffmpeg"),
    )
    print(json.dumps(report["score"], indent=2))
    return 0 if report["score"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
