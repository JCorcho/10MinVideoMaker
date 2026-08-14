"""Strict, versioned configuration for the opt-in production QC lane."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping


class QualityControlConfigurationError(ValueError):
    """Raised when QC configuration is unsafe or internally inconsistent."""


def _strict_bool(values: Mapping[str, str], key: str, default: bool) -> bool:
    if key not in values:
        return default
    value = values[key].strip().casefold()
    if value in {"1", "true"}:
        return True
    if value in {"0", "false"}:
        return False
    raise QualityControlConfigurationError(
        f"{key} must be one of: true, false, 1, 0."
    )


def _optional_path(values: Mapping[str, str], key: str) -> Path | None:
    value = values.get(key)
    if value is None or not value.strip():
        return None
    return Path(value.strip()).expanduser().resolve()


def _integer(
    values: Mapping[str, str], key: str, default: int, minimum: int, maximum: int
) -> int:
    raw = values.get(key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise QualityControlConfigurationError(f"{key} must be an integer.") from error
    if not minimum <= value <= maximum:
        raise QualityControlConfigurationError(
            f"{key} must be between {minimum} and {maximum}."
        )
    return value


def _optional_sha256(values: Mapping[str, str], key: str) -> str | None:
    value = (values.get(key) or "").strip().lower()
    if not value:
        return None
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise QualityControlConfigurationError(f"{key} must be a SHA-256 hex digest.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class QualityControlSettings:
    """Validated Phase-1 policy and the exact evaluator configuration."""

    schema_version: int = 1
    quality_control_enabled: bool = False
    auto_advance_pass: bool = False
    loopback_host: str = "127.0.0.1"
    loopback_port: int = 18081
    llama_executable: Path | None = None
    llama_vendor_root: Path | None = None
    model_path: Path | None = None
    projector_path: Path | None = None
    expected_executable_sha256: str | None = None
    expected_model_sha256: str | None = None
    expected_projector_sha256: str | None = None
    expected_gpu_uuid: str | None = None
    expected_gpu_name: str | None = None
    evaluator_id: str = "tenminvideomaker.production-vlm-qc"
    evaluator_version: str = "phase1-v2"
    backend_family: str = "llama.cpp"
    backend_version: str = "2.28.2"
    model_id: str = "Qwen3.6 27B Uncensored HauhauCS Balanced"
    quantization: str = "IQ3_M GGUF"
    projector_precision: str = "FP16"
    prompt_version: str = "production_ltx_video_qc_v1"
    context_length: int = 16384
    parallel_slots: int = 1
    image_min_tokens: int = 1024
    startup_timeout_seconds: int = 300
    request_timeout_seconds: int = 900
    shutdown_timeout_seconds: int = 30
    sampling_fps: float = 2.0
    frames_per_window: int = 4
    preprocessing_version: str = "vlm-qc-lab-f634ca2-image-v1"
    validated_lab_commit: str = "f634ca2ab7ca95ddd9abde7fe840031eba0696f4"
    image_decode_backend: str = "ffmpeg_selected_png_rgb24"
    image_max_short_edge: int = 512
    image_max_pixels: int = 458_752
    image_dimension_multiple: int = 16
    image_resize_interpolation: str = "opencv_inter_area"
    image_jpeg_quality: int = 88
    image_orientation: str = "encoded_pixels_no_autorotate"
    image_color_pipeline: str = "rgb24_to_opencv_bgr_to_jpeg"
    minimum_error_severity: int = 3
    minimum_error_confidence: float = 0.85
    minimum_strong_windows: int = 2

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> "QualityControlSettings":
        values = os.environ if environment is None else environment
        return cls(
            quality_control_enabled=_strict_bool(
                values, "TENMIN_QUALITY_CONTROL_ENABLED", False
            ),
            auto_advance_pass=_strict_bool(
                values, "TENMIN_QC_AUTO_ADVANCE_PASS", False
            ),
            loopback_port=_integer(
                values, "TENMIN_QC_LOOPBACK_PORT", 18081, 1024, 65535
            ),
            llama_executable=_optional_path(values, "TENMIN_QC_LLAMA_EXECUTABLE"),
            llama_vendor_root=_optional_path(values, "TENMIN_QC_LLAMA_VENDOR_ROOT"),
            model_path=_optional_path(values, "TENMIN_QC_MODEL_PATH"),
            projector_path=_optional_path(values, "TENMIN_QC_PROJECTOR_PATH"),
            expected_executable_sha256=_optional_sha256(
                values, "TENMIN_QC_LLAMA_SHA256"
            ),
            expected_model_sha256=_optional_sha256(values, "TENMIN_QC_MODEL_SHA256"),
            expected_projector_sha256=_optional_sha256(
                values, "TENMIN_QC_PROJECTOR_SHA256"
            ),
            expected_gpu_uuid=(values.get("TENMIN_QC_GPU_UUID") or "").strip() or None,
            expected_gpu_name=(values.get("TENMIN_QC_GPU_NAME") or "").strip() or None,
            startup_timeout_seconds=_integer(
                values, "TENMIN_QC_STARTUP_TIMEOUT_SECONDS", 300, 1, 3600
            ),
            request_timeout_seconds=_integer(
                values, "TENMIN_QC_REQUEST_TIMEOUT_SECONDS", 900, 1, 7200
            ),
            shutdown_timeout_seconds=_integer(
                values, "TENMIN_QC_SHUTDOWN_TIMEOUT_SECONDS", 30, 1, 300
            ),
        )

    def validate_for_start(self) -> None:
        """Fail closed before any evaluator process is launched."""
        if not self.quality_control_enabled:
            raise QualityControlConfigurationError(
                "The QC kill switch is disabled; evaluator launch is forbidden."
            )
        fixed_values = {
            "schema_version": (self.schema_version, 1),
            "evaluator_id": (self.evaluator_id, "tenminvideomaker.production-vlm-qc"),
            "evaluator_version": (self.evaluator_version, "phase1-v2"),
            "backend_family": (self.backend_family, "llama.cpp"),
            "backend_version": (self.backend_version, "2.28.2"),
            "model_id": (self.model_id, "Qwen3.6 27B Uncensored HauhauCS Balanced"),
            "quantization": (self.quantization, "IQ3_M GGUF"),
            "projector_precision": (self.projector_precision, "FP16"),
            "prompt_version": (self.prompt_version, "production_ltx_video_qc_v1"),
            "context_length": (self.context_length, 16384),
            "parallel_slots": (self.parallel_slots, 1),
            "image_min_tokens": (self.image_min_tokens, 1024),
            "sampling_fps": (self.sampling_fps, 2.0),
            "frames_per_window": (self.frames_per_window, 4),
            "preprocessing_version": (
                self.preprocessing_version,
                "vlm-qc-lab-f634ca2-image-v1",
            ),
            "validated_lab_commit": (
                self.validated_lab_commit,
                "f634ca2ab7ca95ddd9abde7fe840031eba0696f4",
            ),
            "image_decode_backend": (
                self.image_decode_backend,
                "ffmpeg_selected_png_rgb24",
            ),
            "image_max_short_edge": (self.image_max_short_edge, 512),
            "image_max_pixels": (self.image_max_pixels, 458_752),
            "image_dimension_multiple": (self.image_dimension_multiple, 16),
            "image_resize_interpolation": (
                self.image_resize_interpolation,
                "opencv_inter_area",
            ),
            "image_jpeg_quality": (self.image_jpeg_quality, 88),
            "image_orientation": (
                self.image_orientation,
                "encoded_pixels_no_autorotate",
            ),
            "image_color_pipeline": (
                self.image_color_pipeline,
                "rgb24_to_opencv_bgr_to_jpeg",
            ),
            "minimum_error_severity": (self.minimum_error_severity, 3),
            "minimum_error_confidence": (self.minimum_error_confidence, 0.85),
            "minimum_strong_windows": (self.minimum_strong_windows, 2),
        }
        changed = [
            name for name, (actual, expected) in fixed_values.items()
            if actual != expected
        ]
        if changed:
            raise QualityControlConfigurationError(
                "Validated Phase-1 QC settings changed: " + ", ".join(changed) + "."
            )
        if self.loopback_host != "127.0.0.1":
            raise QualityControlConfigurationError("QC must bind to IPv4 loopback only.")
        required_files = {
            "TENMIN_QC_LLAMA_EXECUTABLE": self.llama_executable,
            "TENMIN_QC_MODEL_PATH": self.model_path,
            "TENMIN_QC_PROJECTOR_PATH": self.projector_path,
        }
        for key, path in required_files.items():
            if path is None or not path.is_file():
                raise QualityControlConfigurationError(
                    f"{key} must identify the validated existing file."
                )
        expected_hashes = {
            "TENMIN_QC_LLAMA_SHA256": (
                self.llama_executable,
                self.expected_executable_sha256,
            ),
            "TENMIN_QC_MODEL_SHA256": (self.model_path, self.expected_model_sha256),
            "TENMIN_QC_PROJECTOR_SHA256": (
                self.projector_path,
                self.expected_projector_sha256,
            ),
        }
        for key, (path, expected) in expected_hashes.items():
            if expected is None or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                raise QualityControlConfigurationError(
                    f"{key} is required to bind the validated QC asset."
                )
            assert path is not None
            actual = _sha256_file(path)
            if actual != expected.lower():
                raise QualityControlConfigurationError(
                    f"{key} does not match the configured QC asset."
                )
        if self.llama_vendor_root is None or not self.llama_vendor_root.is_dir():
            raise QualityControlConfigurationError(
                "TENMIN_QC_LLAMA_VENDOR_ROOT must identify the validated vendor directory."
            )
        if not self.expected_gpu_uuid or not re.fullmatch(
            r"GPU-[A-Za-z0-9-]+", self.expected_gpu_uuid
        ):
            raise QualityControlConfigurationError(
                "TENMIN_QC_GPU_UUID must identify the physical QC GPU; ordinals are unsafe."
            )
        if not self.expected_gpu_name or "RTX 4080 SUPER" not in self.expected_gpu_name.upper():
            raise QualityControlConfigurationError(
                "TENMIN_QC_GPU_NAME must identify the validated RTX 4080 SUPER."
            )

    def effective_document(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy": {
                "quality_control_enabled": self.quality_control_enabled,
                "auto_advance_pass": self.auto_advance_pass,
            },
            "evaluator": {
                "id": self.evaluator_id,
                "version": self.evaluator_version,
                "backend_family": self.backend_family,
                "backend_version": self.backend_version,
                "model_id": self.model_id,
                "quantization": self.quantization,
                "projector_precision": self.projector_precision,
                "prompt_version": self.prompt_version,
            },
            "assets": {
                "llama_executable": str(self.llama_executable) if self.llama_executable else None,
                "llama_vendor_root": str(self.llama_vendor_root) if self.llama_vendor_root else None,
                "model_path": str(self.model_path) if self.model_path else None,
                "projector_path": str(self.projector_path) if self.projector_path else None,
                "expected_executable_sha256": self.expected_executable_sha256,
                "expected_model_sha256": self.expected_model_sha256,
                "expected_projector_sha256": self.expected_projector_sha256,
            },
            "gpu": {
                "expected_uuid": self.expected_gpu_uuid,
                "expected_name": self.expected_gpu_name,
            },
            "server": {
                "host": self.loopback_host,
                "port": self.loopback_port,
                "context_length": self.context_length,
                "parallel_slots": self.parallel_slots,
                "image_min_tokens": self.image_min_tokens,
                "startup_timeout_seconds": self.startup_timeout_seconds,
                "request_timeout_seconds": self.request_timeout_seconds,
                "shutdown_timeout_seconds": self.shutdown_timeout_seconds,
            },
            "sampling": {
                "fps": self.sampling_fps,
                "frames_per_window": self.frames_per_window,
                "preprocessing": {
                    "version": self.preprocessing_version,
                    "validated_lab_commit": self.validated_lab_commit,
                    "decoder": self.image_decode_backend,
                    "max_short_edge": self.image_max_short_edge,
                    "max_pixels": self.image_max_pixels,
                    "dimension_multiple": self.image_dimension_multiple,
                    "resize_interpolation": self.image_resize_interpolation,
                    "jpeg_quality": self.image_jpeg_quality,
                    "orientation": self.image_orientation,
                    "color_pipeline": self.image_color_pipeline,
                },
            },
            "evidence_policy": {
                "minimum_error_severity": self.minimum_error_severity,
                "minimum_error_confidence": self.minimum_error_confidence,
                "minimum_strong_windows": self.minimum_strong_windows,
            },
        }

    def effective_sha256(self) -> str:
        encoded = json.dumps(
            self.effective_document(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
