# Native Full-Resolution Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the blurry half-resolution continuation output with the native 768x1344 LTX second-pass video while preserving bounded latent continuation and crash recovery.

**Architecture:** Stage one remains the bounded motion/continuation state. Stage two saves and decodes its native full-resolution video latent and matching audio; artifact validation enforces representation-specific spatial shapes, and a new hash-bound realism-adjacent acceptance gate controls automatic rollout.

**Tech Stack:** Python 3.12, unittest, ComfyUI API graphs, LTX 2.3 two-pass sampling, safetensors, FFmpeg/FFprobe, SQLite durable state.

## Global Constraints

- Read and write only inside the authorized project root and project-owned `D:\LTX_Supervisor_Storage`; shared ComfyUI is API/read-only except for authorized test generation.
- Do not inspect or use protected VRGDGirl/Violets projects.
- Production output remains 768x1344 at 24 fps and LTX frame counts remain `8n+1`.
- Never interrupt another active ComfyUI prompt; bounded test generation requires an empty queue.
- Use test-first changes, `apply_patch`, focused commits, full verification, and push only `main` in this repository.

---

### Task 1: Enforce native stage-two artifact identity

**Files:**
- Modify: `tests/test_chunk_artifacts.py`
- Modify: `tenminvideomaker/chunk_artifacts.py`

**Interfaces:**
- Consumes: existing `save_latent_checkpoint()` and `load_latent_checkpoint()` calls.
- Produces: artifact-kind-specific 21x12 and 42x24 spatial validation without changing the ComfyUI node signature.

- [ ] Add tests proving `stage1_handoff` accepts only 21x12 and `stage2_video` accepts only 42x24.
- [ ] Run the focused tests and verify they fail because spatial identity is not enforced.
- [ ] Implement the minimal artifact-kind spatial check.
- [ ] Run the focused tests and verify they pass.

### Task 2: Restore native second-pass video routing

**Files:**
- Modify: `tests/test_continuation_workflow.py`
- Modify: `tests/test_continuation_renderer.py`
- Modify: `tenminvideomaker/continuation_workflow.py`
- Modify: `tenminvideomaker/continuation_renderer.py`
- Modify: `tenminvideomaker/constants.py`

**Interfaces:**
- Consumes: `LTXVSeparateAVLatent` outputs and existing `stage2_video`/`stage2_audio` checkpoints.
- Produces: `stage2_video` from split output 0, direct full-resolution decode in generation and recovery, and runtime identity without RealESRGAN.

- [ ] Change workflow tests to require split video checkpointing and forbid pixel-upscaler nodes.
- [ ] Run focused tests and verify failure against the blurry route.
- [ ] Route `stage2_video` from the sampled split output and decode it directly.
- [ ] Remove RealESRGAN from continuation contract/runtime identity.
- [ ] Run focused tests and verify they pass.

### Task 3: Replace stale rollout approval with a detail-aware gate

**Files:**
- Modify: `tests/test_continuation_validation.py`
- Modify: `tests/test_run_continuation_acceptance.py`
- Modify: `tenminvideomaker/continuation_validation.py`
- Modify: `scripts/run_continuation_acceptance.py`
- Modify: `examples/safe_continuation_source.json`

**Interfaces:**
- Consumes: completed acceptance `run.json`, implementation hash, node-contract hash, and external-asset records.
- Produces: schema-v3 validation at `continuation-validation-v3.json` with native-full-resolution and realism/detail decisions plus recorded sharpness metrics.

- [ ] Add failing tests for the new validation filename/decisions, no RealESRGAN asset, and detail metrics.
- [ ] Implement the schema-v3 gate and deterministic Laplacian-detail metric extraction.
- [ ] Replace the cel-shaded fixture with a safe realism-adjacent adult scene.
- [ ] Run focused tests and no-render live graph validation.

### Task 4: Run bounded GPU quality acceptance

**Files:**
- Runtime only: `D:\LTX_Supervisor_Storage\acceptance\<run-id>`
- Runtime only: `D:\LTX_Supervisor_Storage\state\continuation-validation-v3.json`

**Interfaces:**
- Consumes: corrected graphs, safe source frame, empty ComfyUI queue.
- Produces: four bounded generations, native full-resolution checkpoints, sharpness/seam evidence, assembled proof, and hash-bound approval only after review.

- [ ] Generate the safe realism-adjacent T2I source.
- [ ] Run the bounded four-case acceptance matrix with telemetry.
- [ ] Inspect representative source/base/seam frames and the assembled production proxy.
- [ ] Verify native checkpoint shapes, video/audio profile, peak VRAM, detail, identity, anatomy, and motion.
- [ ] Write and revalidate schema-v3 approval only if every decision passes.

### Task 5: Documentation, verification, and restart

**Files:**
- Modify: `AI_DEVELOPMENT_RULES.md`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/user-guide.md`

**Interfaces:**
- Consumes: verified implementation and acceptance evidence.
- Produces: durable engineering/user guidance, a focused commit, pushed `main`, and one safely resumed GUI/supervisor.

- [ ] Document the root cause, new representation contract, recovery behavior, acceptance evidence, and user-facing consequences.
- [ ] Run the full unit suite, compile check, live no-render validator, and `git diff --check`.
- [ ] Commit and push the focused change.
- [ ] Start exactly one project GUI/supervisor and verify high-level health without exposing prompts.
