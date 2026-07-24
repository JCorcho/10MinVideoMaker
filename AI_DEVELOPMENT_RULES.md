# 10MinVideoMaker engineering rules

## Scope and repository identity

- Repository root: `C:\AI\ComfyUI\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI\custom_nodes\10MinVideoMaker`.
- This project is independent of other custom-node projects. Do not use their code, workflows, documentation, or Git history as implementation references.
- Keep new code, tests, workflows, documentation, and test assets inside this repository.
- Do not alter shared ComfyUI startup scripts, global model configuration, shared model files, or the running server unless a task explicitly authorizes that exact change.
- Authorized exceptions for this project: read-only inspection of the three reference workflows in `C:\AI\ComfyUI\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI\user\default\workflows\10minvideomaker`; new project workflows in that same directory; and new project-owned output directories on `D:`. Preserve the existing reference workflows unchanged.

## Architecture decisions

- The repository is a ComfyUI custom-node package. `__init__.py` is the single package entry point and owns `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`.
- Versioned, GUI-format ComfyUI workflows belong in `workflows/`. Pair them with API-format test fixtures only when a workflow must be run headlessly.
- Nodes must share routing and validation logic across every user surface. Do not create divergent editor and Wizard/modal implementations.
- Before coding any node or workflow, obtain the live input/output contract from the local ComfyUI API. Do not infer third-party node inputs or output slots.
- Production video geometry is fixed at 704×1248 and 24 fps. LTX I2V clips use the `8n + 1` frame rule, a maximum duration of 32 seconds, LCM for both sampler passes, the verified first-pass and upscale sigma schedules, and the LTX spatial upscaler.
- T2I retains the matching reference workflow sampler: Anima uses `er_sde` with `beta57`; Pony uses `res_5s_ode` then `res_3m_ode`, both with their reference values.
- Gmail polling and ComfyUI restart supervision run outside ComfyUI node execution. Nodes and the supervisor share one service layer so that authentication, state transitions, and validation cannot diverge.
- ComfyUI 0.27.1 on this machine discovers this project through legacy `NODE_CLASS_MAPPINGS`; a V3-only entrypoint
  imported but did not appear in `/object_info`. Keep node wrappers thin and framework-independent services
  authoritative until the live loader behavior changes.

## Implementation and testing

- Use `apply_patch` for source and documentation edits.
- Add focused regression tests for every routing fix or bug fix.
- Prefer no-render validation before image, video, or audio generation. Do not generate media, download models, or install dependencies without explicit task authorization.
- After each logical change, run applicable syntax checks and tests, then `git diff --check`.
- Preserve user-owned changes. Commit each completed logical change with a concise message. Verify the repository root before every Git operation.

## Documentation and handoff

- Update this file for architecture choices, routing decisions, reproduction steps, and verification commands that future agents need.
- Update `README.md` whenever a user-facing feature is completed.
- Record each completed logical change below, including the exact files changed and tests run.

## Change log

### 2026-07-24 — initial repository scaffold

- Changed files: `.gitignore`, `README.md`, `AI_DEVELOPMENT_RULES.md`, `__init__.py`, `workflows/.gitkeep`, `tests/.gitkeep`.
- Architecture: start with an empty, load-safe ComfyUI package mapping. Functional nodes and their routes will be added only after their live contracts are verified.
- Reproduction: open this directory as a ComfyUI custom-node package; importing `__init__.py` must succeed without optional dependencies.
- Verification commands:
  - `python -m py_compile __init__.py`
  - `git diff --check`
  - `git status --short --branch`

### 2026-07-24 — resolved generation profile and durable core

- Reference facts were read once from the approved workflow directory and will be rebuilt independently; no reference JSON will be modified or copied into this project.
- T2I references: `CyberRealistic_AnimaSemi_V6.0.safetensors` with Anima's `er_sde`/`beta57` sampler; `cyberrealisticPony_v180Coreshift.safetensors` with the Pony two-pass sampler settings above.
- I2V reference facts: LCM on both passes; first-pass sigmas `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0`; upscale sigmas `0.909375, 0.725, 0.421875, 0.0`; spatial upscaler `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`; DMD `LTX2.3_DMD_reshaped_r256.safetensors` at 1.0; JoyAI `JoyAI-Echo-content_r256.safetensors` at 0.5.

### 2026-07-24 — ComfyUI control-node surface

- Changed files: `__init__.py`, `tenminvideomaker/nodes.py`, `tenminvideomaker/assembly.py`,
  `tests/test_nodes.py`, `tests/test_assembly.py`, `README.md`, `docs/architecture.md`,
  `AI_DEVELOPMENT_RULES.md`.
- Architecture: seven V1-compatible node wrappers expose the shared contract, state, Gmail, asset, cleanup, and
  assembly services. Gmail polling nodes never sleep; the future supervisor owns the five-minute schedule.
- Routing: the stitching node uses FFprobe before FFmpeg concat, rejecting any clip that differs from 704×1248
  or 24 fps. Dynamic LoRAs are de-duplicated by case-insensitive name before resolution.
- Live verification: all seven node types were returned by ComfyUI `/object_info`; the no-render
  `10MinVideoMaker_ReleaseMemory` API prompt completed successfully.
- Verification commands:
  - `python -m unittest discover -s tests -v`
  - `python -m compileall -q tenminvideomaker __init__.py`
  - `git diff --check`
