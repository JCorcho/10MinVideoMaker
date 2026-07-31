# Native Full-Resolution Continuation Design

## Problem

The current continuation route samples a native 768x1344 LTX second pass but
discards its video latent. It saves the 384x672 first-pass handoff under the
`stage2_video` artifact name, decodes that lower-resolution latent, and applies
RealESRGAN x2. This preserves broad style better than the rejected cel-shaded
diagnostics, but it cannot retain the fine facial, hair, skin, fabric, or edge
detail present in the source frame. The resulting production clips are visibly
soft and do not meet the project's quality requirements.

## Decision

Use the official native two-stage LTX route for production pictures and audio.
The bounded first-pass latent remains the only state passed into
`LTXVExtendSampler`. The spatially upscaled and second-pass sampled 768x1344
video latent becomes the durable `stage2_video` checkpoint and the only visual
source for raw chunk muxing, recovery, scene assembly, and later continuation
guides. RealESRGAN is removed from continuation generation and recovery.

The initial second pass continues to reinject the exact cached T2I frame at
full resolution. Each later second pass continues to use the preceding accepted
raw window's 25-frame full-resolution visible overlap at the existing aligned
frame index. This keeps motion continuation in stage one while anchoring the
appearance users actually see in stage two.

## Fail-Closed Artifact Contract

Artifact validation distinguishes the two video representations:

- `stage1_handoff`: spatial latent shape 21x12, corresponding to 384x672.
- `stage2_video`: spatial latent shape 42x24, corresponding to 768x1344.

A half-resolution latent can no longer be saved or recovered as
`stage2_video`. Existing schema-3 attempts produced by the blurry route are not
reused by the corrected implementation because generation identity changes;
their files remain as immutable historical evidence.

## Acceptance Gate

Automatic rollout uses a new validation schema and filename so the old
cel-shaded approval cannot authorize this route. The new gate requires:

1. Live structural validation of every representative continuation graph.
2. A safe, fully clothed, realism-adjacent source at 768x1344.
3. Native 42x24 second-pass video checkpoints and 768x1344 raw windows.
4. Positive peak-VRAM evidence on the 16 GB GPU with no OOM.
5. Objective spatial-detail measurements at the source/base boundary and the
   production seam, recorded without treating natural motion blur as a hard
   failure by itself.
6. Visual review confirming retained realism/semi-realism, identity, anatomy,
   useful fine detail, continuous motion, and no unusable smear or blur.
7. Valid assembled H.264/yuv420p video with stereo 48 kHz AAC.

The supervisor remains stopped until the new approval is hash-bound to the
corrected implementation and live node contracts.

## Alternatives Rejected

- A different external video restorer adds model/VRAM cost and may hallucinate
  details; it does not fix discarding the native full-resolution latent.
- Blending first- and second-pass pixels risks double edges, ghosting, and seam
  instability.
- Keeping RealESRGAN as an automatic fallback would hide a failed second pass
  by emitting structurally valid but visibly unacceptable video.

## Verification

- Focused workflow and artifact tests must fail against the blurry route and
  pass after correction.
- The full unit suite, compile check, live no-render workflow validator, and
  `git diff --check` must pass.
- A bounded safe GPU acceptance run must pass the new quality gate before
  `TENMIN_LTX_CONTINUATION_MODE=auto` is resumed.
