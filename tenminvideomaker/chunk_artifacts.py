"""Safe D-drive checkpoints for bounded LTX video and audio latents."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .storage import StorageLayout, write_json_atomic

LATENT_CHECKPOINT_SCHEMA_VERSION = 1
_ALLOWED_TENSOR_KEYS = frozenset({"samples", "noise_mask", "batch_index"})
_ALLOWED_SCALAR_KEYS = frozenset({"downscale_ratio_spacial"})
_VIDEO_ARTIFACT_KINDS = frozenset({"stage1_handoff", "stage2_video"})
_AUDIO_ARTIFACT_KINDS = frozenset({"stage2_audio"})
_ARTIFACT_KINDS = _VIDEO_ARTIFACT_KINDS | _AUDIO_ARTIFACT_KINDS


class ChunkArtifactError(RuntimeError):
    """Raised when a continuation checkpoint is unsafe, incomplete, or corrupt."""


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _tensor_descriptor(tensor: Any) -> dict[str, object]:
    return {
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
    }


def _validate_latent(
    latent: Mapping[str, Any],
    *,
    artifact_kind: str,
) -> None:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - ComfyUI always supplies torch.
        raise ChunkArtifactError("PyTorch is required for latent checkpoints.") from error

    if not isinstance(latent, Mapping):
        raise ChunkArtifactError("LATENT checkpoint input must be a mapping.")
    unknown = set(latent) - _ALLOWED_TENSOR_KEYS - _ALLOWED_SCALAR_KEYS
    if unknown:
        raise ChunkArtifactError(
            "LATENT checkpoint contains unsupported keys: "
            + ", ".join(sorted(str(key) for key in unknown))
        )
    samples = latent.get("samples")
    if not isinstance(samples, torch.Tensor):
        raise ChunkArtifactError("LATENT checkpoint requires a tensor named samples.")
    if artifact_kind in _VIDEO_ARTIFACT_KINDS:
        if samples.ndim != 5:
            raise ChunkArtifactError("LTX video samples must be a 5-D tensor.")
        batch, channels, frames, height, width = samples.shape
        if batch != 1 or channels != 128 or min(frames, height, width) < 1:
            raise ChunkArtifactError(
                "LTX video samples must have shape [1, 128, frames, height, width]."
            )
    elif artifact_kind in _AUDIO_ARTIFACT_KINDS:
        if (
            samples.ndim < 2
            or samples.ndim > 5
            or int(samples.shape[0]) != 1
            or any(int(size) < 1 for size in samples.shape)
        ):
            raise ChunkArtifactError(
                "LTX audio samples must be a non-empty 2-D to 5-D batch-one tensor."
            )
        batch = int(samples.shape[0])
        channels = int(samples.shape[1])
    else:
        raise ChunkArtifactError("Unsupported latent checkpoint artifact kind.")
    if not samples.is_floating_point():
        raise ChunkArtifactError("LTX video samples must use a floating-point dtype.")

    noise_mask = latent.get("noise_mask")
    if noise_mask is not None:
        if not isinstance(noise_mask, torch.Tensor):
            raise ChunkArtifactError("LATENT noise_mask must be a tensor.")
        if artifact_kind in _VIDEO_ARTIFACT_KINDS:
            if noise_mask.ndim != 5 or (
                noise_mask.shape[0] != batch
                or noise_mask.shape[1] not in {1, channels}
                or tuple(noise_mask.shape[2:]) != (frames, height, width)
            ):
                raise ChunkArtifactError(
                    "LATENT noise_mask shape does not match video samples."
                )
        elif tuple(noise_mask.shape) != tuple(samples.shape):
            raise ChunkArtifactError(
                "LATENT noise_mask shape does not match audio samples."
            )
    batch_index = latent.get("batch_index")
    if batch_index is not None and (
        not isinstance(batch_index, torch.Tensor) or batch_index.ndim != 1
    ):
        raise ChunkArtifactError("LATENT batch_index must be a 1-D tensor.")
    downscale = latent.get("downscale_ratio_spacial")
    if downscale is not None and (
        isinstance(downscale, bool)
        or not isinstance(downscale, (int, float))
        or not math.isfinite(float(downscale))
        or float(downscale) <= 0
    ):
        raise ChunkArtifactError(
            "LATENT downscale_ratio_spacial must be a positive number."
        )


def save_latent_checkpoint(
    layout: StorageLayout,
    latent: Mapping[str, Any],
    *,
    job_id: str,
    scene_id: int,
    revision: int,
    chunk_index: int,
    attempt_number: int,
    artifact_kind: str = "stage1_handoff",
) -> tuple[Path, Mapping[str, Any]]:
    """Atomically persist one bounded latent and finalize its hash manifest."""
    destination = layout.chunk_checkpoint_path(
        job_id,
        scene_id,
        revision,
        chunk_index,
        attempt_number,
        artifact_kind,
    )
    _validate_latent(latent, artifact_kind=artifact_kind)
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - ComfyUI supplies safetensors.
        raise ChunkArtifactError("safetensors is required for latent checkpoints.") from error

    manifest_path = layout.chunk_checkpoint_manifest_path(
        job_id,
        scene_id,
        revision,
        chunk_index,
        attempt_number,
        artifact_kind,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )
    tensors = {
        key: value.detach().to(device="cpu").contiguous()
        for key, value in latent.items()
        if key in _ALLOWED_TENSOR_KEYS and value is not None
    }
    scalar_metadata = {
        key: value
        for key, value in latent.items()
        if key in _ALLOWED_SCALAR_KEYS and value is not None
    }
    embedded_metadata = {
        "schema_version": str(LATENT_CHECKPOINT_SCHEMA_VERSION),
        "artifact_kind": artifact_kind,
        "job_id": job_id,
        "scene_id": str(scene_id),
        "revision": str(revision),
        "chunk_index": str(chunk_index),
        "attempt_number": str(attempt_number),
        "scalar_metadata": json.dumps(
            scalar_metadata,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    try:
        save_file(tensors, str(temporary), metadata=embedded_metadata)
        # Windows requires a writable descriptor for FlushFileBuffers, which
        # Python's os.fsync delegates to.
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    file_hash = sha256_file(destination)
    manifest: dict[str, Any] = {
        "schema_version": LATENT_CHECKPOINT_SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "job_id": job_id,
        "scene_id": scene_id,
        "revision": revision,
        "chunk_index": chunk_index,
        "attempt_number": attempt_number,
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint_path": str(destination),
        "sha256": file_hash,
        "byte_size": destination.stat().st_size,
        "tensors": {
            key: _tensor_descriptor(value)
            for key, value in tensors.items()
        },
        "scalar_metadata": scalar_metadata,
    }
    write_json_atomic(manifest_path, manifest)
    return destination, manifest


def load_latent_checkpoint(
    layout: StorageLayout,
    *,
    job_id: str,
    scene_id: int,
    revision: int,
    chunk_index: int,
    attempt_number: int,
    artifact_kind: str = "stage1_handoff",
    expected_temporal_tokens: int | None = None,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Verify a finalized checkpoint before returning CPU tensors to ComfyUI."""
    if (
        expected_temporal_tokens is not None
        and (
            isinstance(expected_temporal_tokens, bool)
            or not isinstance(expected_temporal_tokens, int)
            or expected_temporal_tokens < 1
        )
    ):
        raise ChunkArtifactError(
            "expected_temporal_tokens must be a positive integer."
        )
    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - ComfyUI supplies safetensors.
        raise ChunkArtifactError("safetensors is required for latent checkpoints.") from error

    checkpoint = layout.chunk_checkpoint_path(
        job_id,
        scene_id,
        revision,
        chunk_index,
        attempt_number,
        artifact_kind,
    )
    manifest_path = layout.chunk_checkpoint_manifest_path(
        job_id,
        scene_id,
        revision,
        chunk_index,
        attempt_number,
        artifact_kind,
    )
    if not checkpoint.is_file() or not manifest_path.is_file():
        raise ChunkArtifactError("Latent checkpoint is incomplete or missing.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ChunkArtifactError("Latent checkpoint manifest is unreadable.") from error
    expected_identity = {
        "schema_version": LATENT_CHECKPOINT_SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "job_id": job_id,
        "scene_id": scene_id,
        "revision": revision,
        "chunk_index": chunk_index,
        "attempt_number": attempt_number,
        "checkpoint_path": str(checkpoint),
    }
    for field, expected in expected_identity.items():
        if manifest.get(field) != expected:
            raise ChunkArtifactError(
                f"Latent checkpoint manifest {field} does not match its request."
            )
    if manifest.get("byte_size") != checkpoint.stat().st_size:
        raise ChunkArtifactError("Latent checkpoint byte size does not match its manifest.")
    if manifest.get("sha256") != sha256_file(checkpoint):
        raise ChunkArtifactError("Latent checkpoint SHA-256 verification failed.")
    try:
        tensors = load_file(str(checkpoint), device="cpu")
    except (OSError, RuntimeError, ValueError) as error:
        raise ChunkArtifactError("Latent checkpoint tensor data is unreadable.") from error
    latent: dict[str, Any] = dict(tensors)
    scalar_metadata = manifest.get("scalar_metadata", {})
    if not isinstance(scalar_metadata, Mapping):
        raise ChunkArtifactError("Latent checkpoint scalar metadata is invalid.")
    latent.update(scalar_metadata)
    _validate_latent(latent, artifact_kind=artifact_kind)
    if (
        artifact_kind in _VIDEO_ARTIFACT_KINDS
        and expected_temporal_tokens is not None
        and int(latent["samples"].shape[2]) != expected_temporal_tokens
    ):
        raise ChunkArtifactError(
            "Latent checkpoint temporal shape does not match its continuation plan."
        )
    descriptors = manifest.get("tensors")
    if not isinstance(descriptors, Mapping) or {
        key: _tensor_descriptor(value) for key, value in tensors.items()
    } != descriptors:
        raise ChunkArtifactError("Latent checkpoint tensor descriptors do not match.")
    return latent, manifest


def latent_checkpoint_is_valid(
    layout: StorageLayout,
    **coordinates: Any,
) -> bool:
    try:
        load_latent_checkpoint(layout, **coordinates)
    except ChunkArtifactError:
        return False
    return True
