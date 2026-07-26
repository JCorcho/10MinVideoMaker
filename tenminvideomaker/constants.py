"""Immutable production constraints for the automated video pipeline."""

from __future__ import annotations

import math

LTX_SPATIAL_DIMENSION_STEP = 32
I2V_SPATIAL_SCALE_FACTOR = 2

# 768x1344 is the nearest x2-spatial-upscale route to vertical 9:16 where
# both the production and first-pass dimensions obey LTX's 32-pixel grid.
PRODUCTION_WIDTH = 768
PRODUCTION_HEIGHT = 1344
PRODUCTION_FPS = 24
MAX_SCENE_SECONDS = 32.0

I2V_BASE_WIDTH = PRODUCTION_WIDTH // I2V_SPATIAL_SCALE_FACTOR
I2V_BASE_HEIGHT = PRODUCTION_HEIGHT // I2V_SPATIAL_SCALE_FACTOR

if (
    PRODUCTION_WIDTH != I2V_BASE_WIDTH * I2V_SPATIAL_SCALE_FACTOR
    or PRODUCTION_HEIGHT != I2V_BASE_HEIGHT * I2V_SPATIAL_SCALE_FACTOR
    or any(
        dimension % LTX_SPATIAL_DIMENSION_STEP
        for dimension in (
            PRODUCTION_WIDTH,
            PRODUCTION_HEIGHT,
            I2V_BASE_WIDTH,
            I2V_BASE_HEIGHT,
        )
    )
):
    raise RuntimeError("LTX production and x2 first-pass dimensions must be exact multiples of 32.")

I2V_SAMPLER = "lcm"
I2V_FIRST_PASS_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
I2V_UPSCALE_PASS_SIGMAS = (0.909375, 0.725, 0.421875, 0.0)
I2V_SPATIAL_UPSCALER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
I2V_DYNAMIC_BASE_MODEL = "LTXV 2.3"

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
