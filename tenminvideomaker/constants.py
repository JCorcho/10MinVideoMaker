"""Immutable production constraints for the automated video pipeline."""

from __future__ import annotations

import math

PRODUCTION_WIDTH = 704
PRODUCTION_HEIGHT = 1248
PRODUCTION_FPS = 24
MAX_SCENE_SECONDS = 32.0

I2V_SAMPLER = "lcm"
I2V_FIRST_PASS_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
I2V_UPSCALE_PASS_SIGMAS = (0.909375, 0.725, 0.421875, 0.0)
I2V_SPATIAL_UPSCALER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

MANDATORY_I2V_LORAS = (
    ("LTX2.3_DMD_reshaped_r256.safetensors", 1.0),
    ("JoyAI-Echo-content_r256.safetensors", 0.5),
)


def frame_count_for_seconds(seconds: float) -> int:
    """Return the smallest 8n+1 frame count that covers the requested duration."""
    if not 0 < seconds <= MAX_SCENE_SECONDS:
        raise ValueError(f"Scene duration must be greater than 0 and no more than {MAX_SCENE_SECONDS:g} seconds.")
    return 8 * math.ceil((seconds * PRODUCTION_FPS) / 8) + 1


def seconds_for_frame_count(frame_count: int) -> float:
    """Return LTX timeline duration for a valid 8n+1 frame count."""
    if frame_count < 9 or (frame_count - 1) % 8 != 0:
        raise ValueError("LTX frame count must use the 8n+1 rule and contain at least 9 frames.")
    return (frame_count - 1) / PRODUCTION_FPS
