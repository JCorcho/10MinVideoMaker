# 10MinVideoMaker follow-up work

- [ ] Complete the bounded continuation acceptance matrix on the target 16 GB GPU: `common_base`,
  `single_frame`, `decoded_17_frame`, and `latent_overlap`. Capture positive peak VRAM, runtime, and the exact
  external checkpoint/text-encoder/upscaler/DMD/JoyAI hashes and provenance. Human-review flow discontinuity,
  anatomy, and second-pass seams. Do not claim production quality or enable `auto` until all six decisions pass.
- [ ] After acceptance, create `<TENMIN_STORAGE_ROOT>\state\continuation-validation-v1.json` (default
  `D:\LTX_Supervisor_Storage\state\continuation-validation-v1.json`) with the reviewer/timestamp; a hash covering
  the current continuation generation/routing/recovery implementation; hashes covering every node contract used
  by the representative live continuation graphs; external-asset evidence; four generation results; peak VRAM;
  and accepted decisions. Any covered implementation or representative node-contract change invalidates that
  approval and requires revalidation.
- [ ] Strengthen the legacy single-window I2V crash boundary around temporary `VHS_VideoCombine` output. Recover a
  completed prompt's history output when possible, or persist directly to project-owned durable staging, so a crash
  after rendering but before the supervisor copies the clip to its versioned D-drive directory does not require
  rerendering that scene. Continuation prompts already persist prompt ownership and use lossless project-owned
  `window.mkv`; preserve deterministic paths and completed-scene resume behavior for both routes.
