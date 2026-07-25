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
- T2I retains the matching reference workflow sampler: Anima uses one 30-step `er_sde`/`beta57` pass and no
  detailer; Pony uses 30-step `res_3m_ode` then 30-step `res_5s_ode`, followed by the reference
  `bbox/face_yolov8m.pt` detector and `FaceDetailer` settings.
- Gmail polling and ComfyUI restart supervision run outside ComfyUI node execution. Nodes and the supervisor share one service layer so that authentication, state transitions, and validation cannot diverge.
- One-click setup remains project-local. Store non-secrets in ignored `.env`; encrypt App Passwords, OAuth client
  secrets, and refresh tokens with current-user Windows DPAPI in ignored `runtime/secrets.json`. Process environment
  variables override saved values. Persistent OAuth uses a desktop loopback callback, PKCE/state, offline access, and
  the full Gmail IMAP/SMTP scope.
- ComfyUI 0.27.1 on this machine discovers this project through legacy `NODE_CLASS_MAPPINGS`; a V3-only entrypoint
  imported but did not appear in `/object_info`. Keep node wrappers thin and framework-independent services
  authoritative until the live loader behavior changes.
- The exact Grok schema uses `character.lora.base` to select Anima/Pony and
  `character.lora.recommended_weight` for the global T2I character LoRA. Scene LoRAs continue to use `weight`.
- The LTX x2 spatial-upscale route uses an internal 352×624 first-pass latent and produces the fixed 704×1248
  saved clip. Because 624 is not divisible by the live latent node's 32-pixel step, the spatial decode is
  704×1216; route decoded frames through a final core `ImageScale` using Lanczos, 704×1248, and centered crop before
  video combine. Do not expose or save an alternate production size.
- I2V uses `VHS_VideoCombine` temporary output. The supervisor retrieves its exact history metadata through
  `/view` and writes the project clip under `D:\output\10minfinals\.work`; do not scan or move shared output folders.
- Controlled restart must verify the port-8188 owner is the expected Easy Install embedded Python executable before
  stopping it. Never weaken that path check.
- Standalone automation must resolve LoRAs through the loopback-only project route registered in the live ComfyUI
  process. This makes `folder_paths.get_folder_paths("loras")` and `get_filename_list("loras")` authoritative; do
  not reconstruct model paths in the supervisor process.
- Dynamic LoRA identity is Civitai version ID when available, otherwise normalized download URL. Display names are
  not asset identities. A repeated version keeps the first occurrence, so the global T2I character weight wins when
  Grok repeats that asset in a scene.
- Civitai metadata remains public and must be validated before a transfer. Store the Civitai API token with the other
  DPAPI secrets, attach it only to Civitai download URLs, never log it, and verify the supplied SHA-256 when present.
- An all-scene asset failure pauses the saved job in `error` and must not send a new request email. Manual retry
  requeues only unfinished scenes while preserving completed scenes and attempt counters.
- Assembly/profile failures must also transition to `error` and preserve successful scenes instead of escaping the
  supervisor tick in `stitching`.

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

### 2026-07-24 — dynamic scene workflows

- Changed files: `tenminvideomaker/artifacts.py`, `tenminvideomaker/contracts.py`,
  `tenminvideomaker/nodes.py`, `tenminvideomaker/workflow_builder.py`,
  `tenminvideomaker/workflow_export.py`, `scripts/export_workflows.py`,
  `examples/example_job.json`, `workflows/10MinVideoMaker_*`, `tests/test_artifacts.py`,
  `tests/test_contracts.py`, `tests/test_nodes.py`, `tests/test_workflow_builder.py`,
  `tests/test_workflow_export.py`, `README.md`, `docs/architecture.md`, `docs/user-guide.md`.
- Contract correction: the global character LoRA consumes the example schema's `recommended_weight` and required
  `base`; dynamic stage LoRAs consume `weight`.
- Routing: Anima and Pony use their separate verified T2I samplers. I2V uses the exact cached frame, two LCM passes,
  separate verified sigmas, the x2 spatial upscaler, DMD 1.0, JoyAI 0.5, dynamic model-only LoRAs, generated audio,
  and feed-forward chunking.
- Persistence: T2I frames are atomically written to
  `D:\output\10minfinals\.work\{job_id}\frames\scene_{id}.png`.
- GUI export: project-local layout code is used because the shared skill bundle does not contain its referenced
  `workflow_layout.py`. Export performs live contract validation, deterministic dependency-depth layout, overlap
  inspection, crossing reporting, and group-bound checks.
- Verification:
  - 40 unit tests passed.
  - All Anima, Pony, and I2V generated connections passed live `/object_info` type validation.
  - All GUI exports had zero node overlaps and zero group-bound violations.
  - No model was loaded and no media was rendered.

### 2026-07-24 — unattended supervisor

- Changed files: `tenminvideomaker/comfy_http.py`, `tenminvideomaker/state_store.py`,
  `tenminvideomaker/supervisor.py`, `tenminvideomaker/workflow_builder.py`,
  `scripts/run_supervisor.py`, `scripts/restart_comfyui.ps1`, `tests/test_comfy_http.py`,
  `tests/test_state_store.py`, `tests/test_supervisor.py`, regenerated workflow JSON,
  `README.md`, `docs/architecture.md`, `docs/user-guide.md`, `AI_DEVELOPMENT_RULES.md`.
- State: scene records persist separate T2I/I2V attempt counts and last prompt ID. Interrupted/failed/cancelled scenes
  can be requeued while succeeded scenes remain immutable.
- Scheduling: the external supervisor owns the five-minute Gmail interval; no ComfyUI node sleeps.
- Recovery: prompt timeout cancels the matching prompt; transient stage errors retry only the unfinished stage;
  missing scene assets do not abort other scenes; fatal server loss uses a path-verified controlled restart.
- Output: temporary VHS results are downloaded via the local HTTP API to deterministic D-drive clip paths, then
  FFprobed and stream-copied into `{job_id}_final.mp4`.
- Validation:
  - 46 unit tests passed, including fake T2I → I2V → stitch → next-email execution, transient-stage retry, and
    per-scene asset-failure continuation.
  - PowerShell restart script parsed without syntax errors.
  - The supervisor entry point imported successfully and resolved the configured ComfyUI LoRA root.
  - FFmpeg and FFprobe were present on `PATH`.
  - No Gmail message was sent, no asset was downloaded, no model was loaded, and no media was rendered.

### 2026-07-24 — one-click setup and start

- Changed files: `.gitignore`, `.env.example`, `Start 10MinVideoMaker.bat`,
  `tenminvideomaker/configuration.py`, `tenminvideomaker/oauth.py`, `tenminvideomaker/mail.py`,
  `scripts/setup_and_start.py`, `scripts/run_supervisor.py`, `tests/test_configuration.py`,
  `tests/test_oauth.py`, `tests/test_mail.py`, `tests/test_setup_and_start.py`, `README.md`,
  `docs/architecture.md`, `docs/user-guide.md`, `AI_DEVELOPMENT_RULES.md`.
- Setup: double-clicking the project launcher detects missing Gmail details, supports App Password or Google OAuth2,
  offers the optional-settings editor, validates both transports without sending mail, health-checks local ComfyUI,
  and then replaces the setup process with the durable supervisor.
- OAuth routing: a Desktop-app authorization-code flow uses loopback redirect, PKCE, state, offline access, and
  `https://mail.google.com/`; Gmail access tokens are refreshed from the DPAPI-protected refresh token and cached only
  in memory.
- Persistence: `.env` contains only allowlisted non-secret settings. Secrets are encrypted with current-user Windows
  DPAPI in ignored `runtime/secrets.json`; explicit process values retain precedence.
- Reproduction:
  - Double-click `Start 10MinVideoMaker.bat` for interactive setup and start.
  - Run `python scripts\setup_and_start.py --setup-only` to configure and authenticate without starting the loop.
  - Run `python scripts\setup_and_start.py --setup-only --skip-gmail-check` only for offline UI diagnostics.
- Verification commands:
  - `python -m unittest discover -s tests -v`
  - `python -m compileall -q tenminvideomaker scripts __init__.py`
  - `python scripts\setup_and_start.py --help`
  - `python scripts\run_supervisor.py --help`
  - `git diff --check`
- Results: all 58 tests passed under system Python and the Easy Install embedded Python; both launcher/supervisor
  help entry points succeeded; the local ComfyUI `/system_stats` health check succeeded.
- Side-effect boundary: validation did not send mail, open OAuth, start the supervisor, download assets, load models,
  or render media.

### 2026-07-24 — SMTP XOAUTH2 callback compatibility

- Changed files: `tenminvideomaker/mail.py`, `tests/test_mail.py`, `AI_DEVELOPMENT_RULES.md`.
- Cause: `smtplib.SMTP.auth()` requests its initial response by calling the authentication callback with zero
  arguments. The OAuth callback required one positional challenge argument, so credential validation failed before
  Gmail received the XOAUTH2 payload.
- Fix: SMTP and IMAP OAuth callbacks accept an optional challenge argument. The regression fake deliberately invokes
  the SMTP callback with zero arguments and verifies the generated XOAUTH2 response.
- Verification commands:
  - `python -m unittest discover -s tests -v`
  - `python -m compileall -q tenminvideomaker scripts __init__.py`
  - `git diff --check`
- Results: all 8 focused mail tests and all 59 project tests passed; embedded-Python compilation passed; live
  OAuth refresh plus SMTP/IMAP authentication succeeded without sending or reading email.

### 2026-07-24 — SMTP XOAUTH2 command sequencing

- Changed files: `tenminvideomaker/mail.py`, `tests/test_mail.py`, `AI_DEVELOPMENT_RULES.md`.
- Cause: unlike `SMTP.login()`, `SMTP.auth()` does not automatically issue `EHLO`. Gmail returned `503 EHLO/HELO
  first`; Python's SMTP library treats a `503` response to `AUTH` as already authenticated, so the earlier `NOOP`
  validation falsely passed. `send_message()` then sent `EHLO`, followed by an unauthenticated `MAIL FROM` rejected
  with `530 Authentication Required`.
- Fix: issue `ehlo_or_helo_if_needed()` before XOAUTH2. Credential validation now checks an authenticated,
  non-delivering `MAIL FROM` envelope and immediately clears it with `RSET`, rather than relying on `NOOP`.
- Regression: the SMTP fake records and asserts `EHLO` before `AUTH`, while retaining the zero-argument callback
  check from the prior repair.
- Verification commands:
  - `python -m unittest discover -s tests -v`
  - `python -m compileall -q tenminvideomaker scripts __init__.py`
  - live OAuth SMTP `MAIL FROM` followed by `RSET` (no message body or delivery)
  - `git diff --check`
- Results: all 8 focused mail tests and all 59 project tests passed under the Easy Install embedded Python; live
  OAuth SMTP envelope validation plus IMAP authentication succeeded without sending or reading email.

### 2026-07-24 — durable I2V staging follow-up

- Changed files: `TODO.md`, `AI_DEVELOPMENT_RULES.md`.
- Follow-up: record stronger recovery for the narrow interval after temporary VHS rendering completes but before
  the supervisor copies the clip to its deterministic D-drive path. The current runtime behavior is unchanged.
- Verification command: `git diff --check`.

### 2026-07-24 — live LoRA resolution, Civitai authentication, and saved-job retry

- Changed files: `__init__.py`, `.env.example`, `scripts/run_supervisor.py`,
  `scripts/setup_and_start.py`, `tenminvideomaker/assets.py`, `tenminvideomaker/comfy_http.py`,
  `tenminvideomaker/configuration.py`, `tenminvideomaker/contracts.py`, `tenminvideomaker/nodes.py`,
  `tenminvideomaker/server_api.py`, `tenminvideomaker/state_store.py`,
  `tenminvideomaker/supervisor.py`, `tenminvideomaker/workflow_builder.py`,
  `tests/test_assets.py`, `tests/test_comfy_http.py`, `tests/test_configuration.py`,
  `tests/test_contracts.py`, `tests/test_server_api.py`, `tests/test_setup_and_start.py`,
  `tests/test_state_store.py`, `tests/test_supervisor.py`, `tests/test_workflow_builder.py`,
  `README.md`, `docs/architecture.md`, `docs/user-guide.md`, `AI_DEVELOPMENT_RULES.md`.
- Cause: the external supervisor imported a separate `folder_paths` context, so it did not necessarily see the
  running server's active LoRA roots. Civitai metadata was anonymous but file downloads redirected to account login.
  Asset identity also used the JSON display name, allowing duplicate versions with different names.
- Routing: the supervisor now calls a loopback-only ComfyUI route that resolves against the live process's roots and
  selectable filenames. Dynamic assets use version/URL identity, public metadata validation, encrypted Civitai-token
  downloads, canonical filename discovery, disk preflight, atomic transfer, and hash verification. Resolved live
  filenames are injected into generated workflows.
- Recovery: asset errors are printed immediately. An all-scene failure pauses in `error` without requesting another
  job; the launcher offers to retry the saved job while preserving successes and attempt counters.
- Reproduction:
  - Double-click `Start 10MinVideoMaker.bat`, configure a Civitai API key from Account Settings when prompted, and
    accept the saved-job retry.
  - For no-render validation, restart ComfyUI and resolve the already-installed mandatory DMD/JoyAI files through
    `POST /10minvideomaker/assets/resolve`.
- Verification commands:
  - `python -m unittest discover -s tests -v`
  - `python -m compileall -q tenminvideomaker scripts __init__.py`
  - `python scripts/setup_and_start.py --help`
  - `python scripts/run_supervisor.py --help`
  - `git diff --check`
- Results: all 75 tests passed under system and Easy Install embedded Python. Live ComfyUI 0.27.1 registered all
  eight project nodes, and its loopback route found both installed mandatory I2V LoRAs by their real selectable
  filenames. No Gmail message was sent, no model was downloaded or loaded, and no media was rendered.

### 2026-07-24 — exact LTX output geometry and assembly recovery

- Changed files: `tenminvideomaker/workflow_builder.py`, `tenminvideomaker/supervisor.py`,
  `tests/test_workflow_builder.py`, `tests/test_supervisor.py`, regenerated `workflows/10MinVideoMaker_*`,
  `README.md`, `docs/architecture.md`, `docs/user-guide.md`, `AI_DEVELOPMENT_RULES.md`.
- Cause: 704×1248 is divisible by 32, but the x2 spatial upscaler's half-height is 624, which is not divisible by
  the live `EmptyLTXVLatentVideo` 32-pixel step. The first production job consequently decoded every I2V scene at
  704×1216, and the strict assembly preflight correctly rejected it.
- Routing: decoded I2V frames now pass through core `ImageScale` with Lanczos scale-to-fill and centered crop at
  704×1248 before `VHS_VideoCombine`. The crop removes roughly nine pixels per horizontal edge without stretching
  subjects. Regenerated GUI/API workflows also synchronize the previously stale T2I templates with dynamic
  character-LoRA loading.
- Recovery: assembly/profile failures now pause the saved job in `error`, retain successful clips, and do not repeat
  on each polling tick. Job `20260724-2249`'s eight legacy clips were backed up under
  `D:\output\10minfinals\.work\20260724-2249\clips\source-704x1216`, normalized to 704×1248 with their original
  frame counts and AAC streams preserved, then revalidated before final assembly.
- Verification:
  - 77 tests passed under system and Easy Install embedded Python.
  - Live `/object_info` validation accepted the regenerated workflows; all have zero overlaps and no nodes outside
    their group.
  - FFprobe confirmed all eight repaired clips are 704×1248 at 24 fps with audio and unchanged frame counts.
  - The one-tick supervisor recovery stream-copied the eight clips into
    `D:\output\10minfinals\20260724-2249_final.mp4` (704×1248, 24 fps, 3,848 frames, 160.35 seconds, stereo AAC),
    transitioned to `waiting_for_grok`, and sent the next-job request email. The continuous supervisor was left
    stopped after the one-tick recovery.

### 2026-07-24 — corrected Pony passes and restored face detailing

- Changed files: `tenminvideomaker/workflow_builder.py`, `tenminvideomaker/workflow_export.py`,
  `tests/test_workflow_builder.py`, `tests/test_workflow_export.py`, regenerated
  `workflows/10MinVideoMaker_T2I_*.json`, `README.md`, `docs/architecture.md`, `docs/user-guide.md`,
  `AI_DEVELOPMENT_RULES.md`.
- Routing: Pony now executes `res_3m_ode` first and `res_5s_ode` second, both at 30 steps and CFG 6, then sends
  the decoded image through `UltralyticsDetectorProvider` using `bbox/face_yolov8m.pt` and the approved reference
  `FaceDetailer` settings before the deterministic scene-frame saver. The detailer uses the scene seed in fixed mode.
  Anima remains one 30-step `er_sde`/`beta57` pass at CFG 4.5 with no face detailer.
- GUI serialization: sampler/detailer seed controls now emit ComfyUI's separate `fixed` control widget. Without it,
  later values shifted left in the canvas, making Pony CFG 6 appear as a six-step sampler even though the API graph
  contained 30 steps.
- Reproduction: run `python scripts\export_workflows.py --install-approved-shared-copies`, refresh the ComfyUI
  Workflows sidebar, and open `10MinVideoMaker_T2I_Pony.json`.
- Verification:
  - `python -m unittest discover -s tests -v`
  - embedded-Python unit tests and compilation
  - live `/object_info` export validation
  - `git diff --check`
  - no media rendered and no model loaded.
