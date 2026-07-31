# Continuation Acceptance Human Review — 2026-07-31

## Decision

Acceptance run `continuation-acceptance-20260731-065935` produced all four
mechanically valid cases at 768×1344 and 24 fps. The project owner approved no
method for production. Automatic continuation remains locked and
`continuation-validation-v1.json` must not be created from these results.

`single_frame` is the next diagnostic baseline because it alone retained the
source's cel-shaded anime/game style. It is not approved: continuation frame 1
introduces severe blur and fine-eye-detail loss, and the blur worsens during
camera motion.

The durable structured record and censored screenshots are stored alongside the
acceptance run:

`D:\LTX_Supervisor_Storage\acceptance\continuation-acceptance-20260731-065935\human-review.json`

## Single final frame

- Identity, face, hair, wardrobe, proportions, and source style remain stable.
- Existing base-window anatomy defects remain; the method does not repair them.
- Motion direction continues, but continuation frame 1 becomes visibly blurred
  and loses eye detail.
- Continuation frame 0 behaves as the conditioning anchor, not a new frame.
- Mechanical boundary RGB MAE: `3.626852`.
- Mechanical motion-vector discontinuity: `6.833459`.
- Production status: conditional next baseline, not approved.
- Exact assembled seam: base 0–120, then continuation 1 onward.

## Decoded 17-frame guide

- The cel-shaded character becomes photorealistic, like a live-action cosplay
  interpretation.
- Face and hair color change, while hairstyle geometry, wardrobe geometry,
  proportions, background, motion direction, and camera movement remain
  coherent.
- A visible left-hand defect improves while hand-to-bag contact remains stable.
- The seam reads as continuous motion, but the abrupt style conversion makes it
  invalid for same-style production continuation.
- Mechanical guide-end RGB MAE: `56.684467`.
- Mechanical motion-vector discontinuity: `4.773929`.
- Production status: rejected.
- Research status: strongest preserved anime-to-live-action conversion lead.
- Exact assembled seam: base 0–112, then continuation 17 onward.

The exact stage-one and stage-two API workflows are frozen under
`experiments/ltx23-style-conversion/`. They are hash-bound in `manifest.json`.
This is one reproducible test result, not a general conversion-quality claim.

## Latent 25-frame overlap

- The source style changes to semi-realistic 3D rather than remaining
  cel-shaded.
- Wardrobe and object contact remain coherent and the hand defect improves.
- Hair becomes hijab-like, forward walking stops, and remaining motion becomes
  robotic.
- Mechanical guide-end RGB MAE: `29.171497`.
- Mechanical motion-vector discontinuity: `1.702146`.
- Production status: rejected.
- Research status: weaker preserved anime-to-semi-realistic-3D lead.
- Exact assembled seam: base 0–120, then continuation 25 onward.

## Review UI interpretation

The side-by-side videos and exact boundary stills remain available. The
full-width assembled player below them is the authoritative playback for seam
judgment because it removes conditioning overlap:

- `single_frame`: drops continuation frame 0;
- `decoded_17_frame`: drops continuation frames 0–16 and ends the base at 112;
- `latent_overlap`: drops continuation frames 0–24.

Assembled files are cached as unwatermarked H.264 video-only review proxies on
D:. They do not change raw FFV1 windows, pipeline state, current jobs, or rollout
approval.
