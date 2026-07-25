"""Deterministic project artifact paths and image persistence."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

from .constants import PRODUCTION_HEIGHT, PRODUCTION_WIDTH
from .storage import StorageError, StorageLayout

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ArtifactError(ValueError):
    """Raised when a project artifact cannot be named or persisted safely."""


def scene_frame_path(
    job_id: str,
    scene_id: int,
    *,
    layout: StorageLayout | None = None,
    revision: int = 1,
) -> Path:
    if not _JOB_ID_RE.fullmatch(job_id):
        raise ArtifactError("job_id contains unsafe path characters.")
    if isinstance(scene_id, bool) or not isinstance(scene_id, int) or scene_id < 1:
        raise ArtifactError("scene_id must be a positive integer.")
    try:
        return (layout or StorageLayout.configured()).scene_frame_path(
            job_id,
            scene_id,
            revision,
        )
    except StorageError as error:
        raise ArtifactError(str(error)) from error


def scene_clip_path(
    job_id: str,
    scene_id: int,
    *,
    layout: StorageLayout | None = None,
    revision: int = 1,
) -> Path:
    if not _JOB_ID_RE.fullmatch(job_id):
        raise ArtifactError("job_id contains unsafe path characters.")
    if isinstance(scene_id, bool) or not isinstance(scene_id, int) or scene_id < 1:
        raise ArtifactError("scene_id must be a positive integer.")
    try:
        return (layout or StorageLayout.configured()).scene_clip_path(
            job_id,
            scene_id,
            revision,
        )
    except StorageError as error:
        raise ArtifactError(str(error)) from error


def save_scene_frame(
    images: Any,
    job_id: str,
    scene_id: int,
    revision: int = 1,
) -> Path:
    """Atomically save the first ComfyUI IMAGE batch item as the scene source frame."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:  # pragma: no cover - ComfyUI supplies both packages.
        raise ArtifactError(f"Image persistence dependencies are unavailable: {error}") from error

    array = images.detach().cpu().numpy() if hasattr(images, "detach") else np.asarray(images)
    if array.ndim != 4 or array.shape[0] < 1 or array.shape[-1] not in {3, 4}:
        raise ArtifactError("images must be a ComfyUI IMAGE tensor shaped [B,H,W,C].")
    if array.shape[1:3] != (PRODUCTION_HEIGHT, PRODUCTION_WIDTH):
        raise ArtifactError(
            f"Scene frame must be {PRODUCTION_WIDTH}x{PRODUCTION_HEIGHT}; "
            f"received {array.shape[2]}x{array.shape[1]}."
        )

    path = scene_frame_path(job_id, scene_id, revision=revision)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".png.tmp")
    mode = "RGBA" if array.shape[-1] == 4 else "RGB"
    pixels = np.clip(array[0] * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode=mode).save(temporary, format="PNG")
    os.replace(temporary, path)
    return path
