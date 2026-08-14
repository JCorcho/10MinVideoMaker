"""Headless, tool-free VLM judge boundary and deterministic evaluator."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .qc_config import QualityControlSettings
from .qc_contracts import (
    JudgeResponse,
    JudgeWindowResult,
    NormalizedEvaluation,
    QcEvidencePolicy,
    is_strong_evidence,
    normalize_window_results,
    parse_judge_response,
)
from .qc_video import (
    QcWindow,
    SampledVideo,
    build_frame_accounting,
    chronological_windows,
    shifted_confirmation_window,
)


VISION_SYSTEM_PROMPT_VERSION = "production_vlm_qc_system_v1"
VISION_REQUEST_RECIPE_VERSION = "production_vlm_qc_request_v1"
VISION_CONFIRMATION_RECIPE_VERSION = "production_vlm_qc_confirmation_v1"

VISION_SYSTEM_PROMPT = (
    "You are a visual production-quality-control tool. Adult or intimate subject "
    "matter is content-neutral and is never itself a defect, but all visible "
    "anatomy—including explicit intimate anatomy and body-to-body contact—must be "
    "inspected for malformed, fused, duplicated, detached, morphing, or temporally "
    "inconsistent geometry. Describe explicit anatomy only as needed to identify a "
    "visible defect. If errors is non-empty, decision must be FAIL; PASS requires "
    "errors to be empty. Return only the requested JSON."
)

CONFIRMATION_SUFFIX = (
    "INDEPENDENT SECOND-PASS REVIEW. Inspect this overlapping timeline window from "
    "scratch without assuming that the previous window was correct or incorrect. "
    "Apply the normal QC criteria exactly. Distinguish visible production defects "
    "from normal expression changes, intentional cuts, perspective, occlusion, body "
    "contact, motion, and stylization. Adult subject matter remains content-neutral."
)


class QcBackendError(RuntimeError):
    """Raised when the isolated backend transport or response envelope fails."""


@dataclass(frozen=True)
class ProductionRubric:
    version: str
    text: str
    sha256: str
    system_prompt_version: str = VISION_SYSTEM_PROMPT_VERSION
    system_prompt_sha256: str = hashlib.sha256(VISION_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    request_recipe_version: str = VISION_REQUEST_RECIPE_VERSION
    confirmation_recipe_version: str = VISION_CONFIRMATION_RECIPE_VERSION


def load_production_rubric(path: Path) -> ProductionRubric:
    text = Path(path).read_text(encoding="utf-8")
    return ProductionRubric(
        version=Path(path).stem,
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


@dataclass(frozen=True)
class BackendIdentity:
    evaluator_id: str
    evaluator_version: str
    backend_family: str
    backend_version: str
    executable_path: str
    executable_sha256: str
    model_path: str
    model_sha256: str
    model_id: str
    quantization: str
    projector_path: str
    projector_sha256: str
    projector_precision: str
    gpu_uuid: str
    gpu_name: str
    effective_args: tuple[str, ...]
    effective_config_sha256: str
    owned_pid: int
    stdout_log_path: str
    stderr_log_path: str


@dataclass(frozen=True)
class VisionJudgeRequest:
    window: QcWindow
    rubric: ProductionRubric
    encoded_images: tuple[str, ...]
    independent_confirmation: bool = False

    @classmethod
    def from_window(
        cls, window: QcWindow, *, rubric: ProductionRubric
    ) -> "VisionJudgeRequest":
        encoded = tuple(
            "data:image/jpeg;base64,"
            + base64.b64encode(frame.bytes()).decode("ascii")
            for frame in window.frames
        )
        return cls(
            window=window,
            rubric=rubric,
            encoded_images=encoded,
            independent_confirmation=window.confirmation_of_window is not None,
        )


@dataclass(frozen=True)
class VisionJudgeEvaluation:
    response: JudgeResponse
    input_tokens: int | None = None
    output_tokens: int | None = None


class QcBackend(Protocol):
    def start(self) -> BackendIdentity: ...

    def evaluate(self, request: VisionJudgeRequest) -> VisionJudgeEvaluation: ...

    def close(self) -> None: ...


def build_vision_judge_payload(
    request: VisionJudgeRequest, *, model_alias: str = "production-vlm-qc"
) -> dict[str, object]:
    timestamps = ", ".join(
        f"{value:.3f}" for value in request.window.timestamps_seconds
    )
    prompt = request.rubric.text
    if request.independent_confirmation:
        prompt = f"{prompt}\n\n{CONFIRMATION_SUFFIX}"
    instruction = (
        f"{prompt}\n\nThis is timeline window {request.window.window_number}. "
        "The attached images are chronological video frames at timestamps in "
        f"seconds: [{timestamps}]. Use only visible visual-quality evidence. "
        "Return the required JSON object only."
    )
    content: list[dict[str, object]] = [{"type": "text", "text": instruction}]
    content.extend(
        {"type": "image_url", "image_url": {"url": image}}
        for image in request.encoded_images
    )
    return {
        "model": model_alias,
        "messages": [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "stream": False,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_prompt": False,
    }


class LlamaCppHttpBackend:
    """OpenAI-compatible transport with one fresh, tool-free request per call."""

    def __init__(self, settings: QualityControlSettings, process: Any):
        self.settings = settings
        self.process = process

    def start(self) -> BackendIdentity:
        return self.process.start()

    def evaluate(self, request: VisionJudgeRequest) -> VisionJudgeEvaluation:
        payload = build_vision_judge_payload(request)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = Request(
            f"http://{self.settings.loopback_host}:{self.settings.loopback_port}"
            "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(
                http_request, timeout=self.settings.request_timeout_seconds
            ) as response:
                envelope = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise QcBackendError(
                f"llama.cpp returned HTTP {error.code}: {detail[-1200:]}"
            ) from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise QcBackendError(f"llama.cpp request failed: {error}") from error
        try:
            content = envelope["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, Mapping)
                )
            usage = envelope.get("usage") or {}
        except (KeyError, IndexError, TypeError) as error:
            raise QcBackendError("llama.cpp returned an invalid response envelope.") from error
        return VisionJudgeEvaluation(
            response=parse_judge_response(str(content)),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    def close(self) -> None:
        self.process.close()


@dataclass(frozen=True)
class HeadlessEvaluationResult:
    normalized: NormalizedEvaluation
    frame_accounting: Mapping[str, object]
    raw_result: str
    window_evaluations: tuple[VisionJudgeEvaluation, ...]
    confirmation_evaluation: VisionJudgeEvaluation | None


class HeadlessVideoEvaluator:
    """Blind deterministic window controller; it has no project mutation tools."""

    def __init__(
        self,
        backend: Any,
        rubric: ProductionRubric,
        *,
        policy: QcEvidencePolicy | None = None,
        frames_per_window: int = 4,
    ):
        self.backend = backend
        self.rubric = rubric
        self.policy = policy or QcEvidencePolicy()
        self.frames_per_window = frames_per_window

    def evaluate_sampled(self, sampled: SampledVideo) -> HeadlessEvaluationResult:
        planned = chronological_windows(sampled, frame_count=self.frames_per_window)
        processed: list[QcWindow] = []
        evaluations: list[VisionJudgeEvaluation] = []
        early_exit = False
        for window in planned:
            evaluation = self.backend.evaluate(
                VisionJudgeRequest.from_window(window, rubric=self.rubric)
            )
            processed.append(window)
            evaluations.append(evaluation)
            strong_count = sum(
                any(is_strong_evidence(error, self.policy) for error in item.response.errors)
                for item in evaluations
            )
            if strong_count >= self.policy.minimum_strong_windows:
                early_exit = True
                break

        main_results = tuple(
            JudgeWindowResult(
                window_number=window.window_number,
                source_frame_indices=window.source_frame_indices,
                timestamps_seconds=window.timestamps_seconds,
                response=evaluation.response,
            )
            for window, evaluation in zip(processed, evaluations)
        )
        confirmation_window: QcWindow | None = None
        confirmation_evaluation: VisionJudgeEvaluation | None = None
        confirmation_result: JudgeWindowResult | None = None
        strong_positions = tuple(
            index
            for index, result in enumerate(main_results)
            if any(is_strong_evidence(error, self.policy) for error in result.response.errors)
        )
        if not early_exit and len(processed) == len(planned) and len(strong_positions) == 1:
            suspect_window = processed[strong_positions[0]]
            confirmation_window = shifted_confirmation_window(
                sampled, suspect_window, frame_count=self.frames_per_window
            )
            if confirmation_window is not None:
                confirmation_evaluation = self.backend.evaluate(
                    VisionJudgeRequest.from_window(
                        confirmation_window, rubric=self.rubric
                    )
                )
                confirmation_result = JudgeWindowResult(
                    window_number=confirmation_window.window_number,
                    source_frame_indices=confirmation_window.source_frame_indices,
                    timestamps_seconds=confirmation_window.timestamps_seconds,
                    response=confirmation_evaluation.response,
                    confirmation_of_window=confirmation_window.confirmation_of_window,
                )
        normalized = normalize_window_results(
            main_results, confirmation_result, self.policy
        )
        reason = (
            "two strong independent normal windows"
            if early_exit
            else normalized.normalization_reason
        )
        accounting = build_frame_accounting(
            sampled,
            planned_windows=planned,
            processed_windows=processed,
            confirmation=confirmation_window,
            early_exit=early_exit,
            early_exit_reason=reason,
        )
        raw_parts = [
            f"--- window {result.window_number}: "
            f"{result.timestamps_seconds[0]:.2f}s-{result.timestamps_seconds[-1]:.2f}s "
            f"· final {result.response.decision.value if result.response.decision else 'UNPARSED'} "
            f"· model {result.response.model_decision.value if result.response.model_decision else 'INVALID'} ---\n"
            f"{result.response.raw_text}"
            for result in main_results
        ]
        if confirmation_result is not None:
            raw_parts.append(
                f"--- adjudication of window {confirmation_result.confirmation_of_window}: "
                f"{confirmation_result.timestamps_seconds[0]:.2f}s-"
                f"{confirmation_result.timestamps_seconds[-1]:.2f}s ---\n"
                f"{confirmation_result.response.raw_text}"
            )
        return HeadlessEvaluationResult(
            normalized=normalized,
            frame_accounting=accounting,
            raw_result="\n\n".join(raw_parts),
            window_evaluations=tuple(evaluations),
            confirmation_evaluation=confirmation_evaluation,
        )
