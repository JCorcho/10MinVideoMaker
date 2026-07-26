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
- Production video geometry is fixed at 768×1344 and 24 fps. LTX I2V clips use the `8n + 1` frame rule, a maximum duration of 32 seconds, LCM for both sampler passes, the verified first-pass and upscale sigma schedules, and the LTX spatial upscaler.
- T2I retains the matching reference workflow sampler: Anima uses one 30-step `er_sde`/`beta57` pass and no
  detailer; Pony uses 30-step `res_3m_ode` then 30-step `res_5s_ode`, followed by the reference
  `bbox/face_yolov8m.pt` detector and `FaceDetailer` settings.
- Gmail polling and ComfyUI restart supervision run outside ComfyUI node execution. Nodes and the supervisor share one service layer so that authentication, state transitions, and validation cannot diverge.
- The supported supervisor surface is a loopback FastAPI browser GUI plus one worker. Explicitly enabled private-LAN
  mode binds only the GUI to `0.0.0.0` behind HTTP Basic credentials stored with Windows DPAPI; ComfyUI remains loopback.
  New Gmail jobs
  auto-start by default for 24/7 operation. The `--hold-new-jobs-for-review` GUI launch option enters jobs in
  `awaiting_review` for testing and requires explicit approval. A cross-process project lock must prevent the
  legacy console supervisor and GUI worker from owning the state machine simultaneously.
- All persistent runtime data belongs under `D:\LTX_Supervisor_Storage`: configuration, DPAPI secrets, SQLite,
  source payloads, versioned scene frames/clips/manifests, finals, logs, and temporary files. C: remains source and
  model-loading storage only. A custom `TENMIN_STORAGE_ROOT` must still resolve to D:.
- Scene edits must use the same typed contract and workflow builders as automatic jobs. Store each edit as a new
  immutable revision. Video-only revisions require an existing frame; image-only remake must not exist in the API,
  state enum, or UI. Preserve unsigned 64-bit seeds as strings across the browser boundary.
- Render scheduling is model-residency aware on the 16 GB GPU. For each automatic job, complete every required T2I
  frame before one intentional model release, then complete every required LTX I2V clip before releasing LTX.
  Remake batches must preflight selected revisions, group image+video remakes by Anima/Pony family, render all of
  those frames, then run every eligible video (including video-only remakes) as one LTX phase. Never call
  ComfyUI's free-memory endpoint between scenes in the same phase.
- Collision choices are `after_current` and `interrupt_current`. Interrupt may cancel only prompts carrying this
  project's ComfyUI client ID, then atomically preserve/abandon the active job before a remake batch runs. Never
  clear the global ComfyUI queue or restart a healthy server for this action.
- The visible supervisor must emit a redacted status heartbeat during both polling sleeps and long blocking work.
  It may report state, job/scene IDs, safe asset names, and queue counts, but never prompts, workflow bodies, URLs,
  tokens, App Passwords, OAuth secrets, or Discord webhooks.
- One-click setup remains project-local. Store non-secrets in the D-drive settings file; encrypt App Passwords,
  OAuth client secrets, refresh tokens, Civitai tokens, and Discord webhooks with current-user Windows DPAPI in the
  D-drive secrets file. Process environment
  variables override saved values. Persistent OAuth uses a desktop loopback callback, PKCE/state, offline access,
  the full Gmail IMAP/SMTP scope, and read-only Google Drive scope.
- Gmail payload precedence is JSON attachment, valid plain-text body JSON, then a supported Google Drive file link.
  Drive retrieval must derive download URLs from a validated file ID, reject folder/arbitrary hosts, cap content at
  5 MiB, and keep transport/authentication failures unread for retry.
- ComfyUI 0.27.1 on this machine discovers this project through legacy `NODE_CLASS_MAPPINGS`; a V3-only entrypoint
  imported but did not appear in `/object_info`. Keep node wrappers thin and framework-independent services
  authoritative until the live loader behavior changes.
- The exact Grok schema uses `character.lora.base` to select Anima/Pony and
  `character.lora.recommended_weight` for the global T2I character LoRA. Scene LoRAs continue to use `weight`.
- The LTX x2 spatial-upscale route uses an internal 384×672 first-pass latent and produces the fixed 768×1344
  saved clip. Every first-pass and production axis is divisible by 32; route decoded frames directly to video
  combine. Do not expose or save an alternate production size or post-decode resize/crop stage.
- I2V uses `VHS_VideoCombine` temporary output. The supervisor retrieves its exact history metadata through
  `/view` and writes the project clip into the matching versioned directory below
  `D:\LTX_Supervisor_Storage\jobs`; do not scan or move shared output folders.
- Controlled restart must verify the port-8188 owner is the expected Easy Install embedded Python executable before
  stopping it. Never weaken that path check.
- Standalone automation must resolve LoRAs through the loopback-only project route registered in the live ComfyUI
  process. This makes `folder_paths.get_folder_paths("loras")` and `get_filename_list("loras")` authoritative; do
  not reconstruct model paths in the supervisor process.
- Dynamic LoRA identity is Civitai version ID when available, otherwise normalized download URL. Display names are
  not asset identities. A repeated version keeps the first occurrence, so the global T2I character weight wins when
  Grok repeats that asset in a scene.
- T2I and I2V LoRA eligibility is strictly separated. Exclude every effective T2I LoRA from I2V by stable asset
  identity. Require a verified Civitai LTX 2.x `baseModel` for all remaining dynamic I2V LoRAs before accepting any
  manifest or installed file; accept compact `LTXV2` and versioned 2.x labels for the LTX 2.3 target, but reject LTX
  1.x, image-model, and unverifiable assets. Key resolved filenames with the I2V validation context. The workflow
  builder must fail closed when that validation-specific key is absent. Exact mandatory DMD/JoyAI local files are
  the only dynamic-metadata exception.
- Civitai metadata remains public and must be validated before a transfer. Store the Civitai API token with the other
  DPAPI secrets, attach it only to Civitai download URLs, never log it, and verify the supplied SHA-256 when present.
- An all-scene asset failure pauses the saved job in `error` and must not send a new request email. Manual retry
  requeues only unfinished scenes while preserving completed scenes and attempt counters.
- The launcher must offer resume/abandon for every active saved state, not only `error`. Declining must cancel only
  ComfyUI prompts carrying the project client ID, atomically mark unfinished scenes cancelled, preserve job/scene
  audit history, and clear the active pipeline pointer to `idle`; returning without a state transition is invalid.
- Assembly/profile failures must also transition to `error` and preserve successful scenes instead of escaping the
  supervisor tick in `stitching`.
- Patreon delivery is a parallel output branch. Preserve clean cached frames and clean deterministic clips as the
  generation/assembly source of truth. Apply the exact `wm.png` settings only before the DiscordSendSave nodes.
  Discord image is lossless PNG quality 100; Discord video is H.264 quality 65 at production fps with decoded audio.
  Both senders must disable local output, prompt/workflow metadata, CDN logging, and GitHub updates. Never commit the
  real webhook; runtime graphs load it from DPAPI, versioned templates use a placeholder, and approved shared GUI
  copies may receive the configured secret during export.

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
- Routing: the stitching node uses FFprobe before FFmpeg concat, rejecting any clip that differs from 768×1344
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

### 2026-07-24 — Google Drive job handoffs

- Changed files: `.env.example`, `tenminvideomaker/drive.py`, `tenminvideomaker/mail.py`,
  `tenminvideomaker/oauth.py`, `tenminvideomaker/configuration.py`, `scripts/setup_and_start.py`,
  `tests/test_drive.py`, `tests/test_mail.py`, `tests/test_oauth.py`, `tests/test_configuration.py`,
  `tests/test_setup_and_start.py`, `README.md`, `docs/architecture.md`, `docs/user-guide.md`,
  `AI_DEVELOPMENT_RULES.md`.
- Routing: inbox extraction preserves attachment-first behavior, then checks plain-text JSON, then accepts a
  validated Google Drive file link from plain text or HTML. Public files download anonymously; private files use the
  Drive API with the same cached OAuth access token. Download/auth failures keep the email unread for retry.
- Security: only supported `drive.google.com` file URLs are accepted; folder and arbitrary URLs are ignored. The
  downloader derives its own Google endpoint from the file ID, restricts final redirect hosts, caps content at
  5 MiB, requires UTF-8, and passes the result through the existing strict job contract.
- Setup: OAuth now requests `drive.readonly` in addition to `mail.google.com`. The launcher detects the old mail-only
  scope marker, reuses DPAPI-protected client credentials for one-time browser reauthorization, opens the Drive API
  enablement page, and validates read-only API access without downloading a file.
- Verification:
  - `python -m unittest discover -s tests -v`
  - embedded-Python unit tests and compilation
  - `python scripts/setup_and_start.py --help`
  - `git diff --check`
- Results: 91 tests passed under system and Easy Install embedded Python. No email was sent, no Drive file was
  downloaded, no media was rendered, and no model was loaded.

### 2026-07-24 — dedicated Gmail completion subject

- Changed files: `tenminvideomaker/mail.py`, `tests/test_mail.py`, `README.md`, `docs/architecture.md`,
  `docs/user-guide.md`, `AI_DEVELOPMENT_RULES.md`.
- Routing: outbound requests retain `Run the LTX video pipeline`; the message body now tells Grok to send a new
  email with exact subject `LTX_JOB_COMPLETE`. IMAP searches unread completion mail, and both the transport reader
  and durable poller enforce exact decoded-subject equality so request mail, replies, forwards, and substring
  matches cannot retrigger the pipeline.
- Drive envelope: Grok's abbreviated body metadata is ignored as an incomplete job and its supported Drive file
  link is downloaded through the existing validated transport. Only the downloaded full `job_id`/`scenes` contract
  is accepted.
- Reproduction: send an unread message from an allowed sender with subject `LTX_JOB_COMPLETE` and the Drive metadata
  envelope. A request-subject message or `Re: LTX_JOB_COMPLETE` must remain unclaimed.
- Verification commands:
  - `python -m unittest discover -s tests -v`
  - embedded-Python unit tests and compilation
  - `git diff --check`
- Results: all 94 tests passed under system and Easy Install embedded Python; compilation and
  `git diff --check` passed. No Gmail message was sent, no Drive file was downloaded, no media was rendered, and no
  model was loaded.

### 2026-07-24 — authenticated fallback for private Drive redirects

- Changed files: `tenminvideomaker/drive.py`, `tests/test_drive.py`, `README.md`, `docs/architecture.md`,
  `docs/user-guide.md`, `AI_DEVELOPMENT_RULES.md`.
- Cause: a private Drive file sent the anonymous download route to Google's sign-in surface. The final-host guard
  correctly rejected that response, but raised a terminal download error instead of the internal access-failure
  signal, so the configured authenticated Drive API route was never attempted.
- Routing: a rejected anonymous content-host redirect now enters the existing OAuth API fallback without consuming
  the redirected response body. An authenticated API 404 is reported specifically as missing or not shared with the
  authorized Gmail account. Transport failures continue to leave the completion email unread for retry.
- Live diagnosis: Gmail SMTP/IMAP, Drive API validation, and the authorized Drive identity all succeeded. The
  handoff file itself returned authenticated Drive API 404 for the correctly configured Gmail identity, confirming
  that its sharing must be corrected by the file owner.
- Reproduction: have a private Drive download resolve to a Google sign-in URL, then verify that the second request is
  `drive/v3/files/{file_id}?alt=media` with bearer authentication. A 404 from that request must mention file sharing.
- Verification commands:
  - `python -m unittest discover -s tests -v`
  - embedded-Python unit tests and compilation
  - `git diff --check`
- Results: all 96 tests passed under system and Easy Install embedded Python; compilation and
  `git diff --check` passed. No Gmail message was sent, no media was rendered, and no model was loaded.

### 2026-07-24 — Windows launcher shortcut and project icon

- Changed files: `assets/10MinVideoMaker-icon.png`, `assets/10MinVideoMaker-icon.ico`,
  `scripts/install_windows_shortcuts.ps1`, `README.md`, `docs/user-guide.md`, `AI_DEVELOPMENT_RULES.md`.
- User surface: the installer creates current-user Desktop and Start Menu Programs shortcuts that invoke the
  existing `Start 10MinVideoMaker.bat` through `cmd.exe`, set the repository as the working directory, and use the
  project-owned multi-resolution icon. It does not start the launcher while installing.
- Reproduction:
  `powershell -ExecutionPolicy Bypass -File scripts\install_windows_shortcuts.ps1`.
- Icon source: generated with the built-in image-generation tool as a compact navy play/timer badge on a removable
  chroma-key background, converted to transparent PNG, and packaged as a 16/24/32/48/64/128/256-pixel ICO.
- Verification:
  - inspect both `.lnk` files through `WScript.Shell` without launching them;
  - confirm the icon contains all intended ICO sizes and its PNG has transparent corners;
  - `python -m unittest discover -s tests -v`;
  - `git diff --check`.
- Results: both shortcuts resolve to the project batch file through `cmd.exe`, with the repository working
  directory and project icon. All 96 tests passed under system and Easy Install embedded Python; no shortcut was
  launched, no Gmail message was sent, and no media was rendered.

### 2026-07-24 — explicit saved-job abandonment

- Changed files: `tenminvideomaker/state_store.py`, `scripts/setup_and_start.py`,
  `tests/test_state_store.py`, `tests/test_setup_and_start.py`, `README.md`, `docs/architecture.md`,
  `docs/user-guide.md`, `AI_DEVELOPMENT_RULES.md`.
- Cause: declining the saved-job retry returned from the launcher prompt without changing durable state. The
  supervisor therefore reopened the same `error` snapshot and remained paused on the rejected job.
- Recovery: the decline path now atomically marks unfinished scenes `cancelled`, clears prompt IDs, retains the
  original job payload, errors, successes, and attempt counters for audit, and resets the singleton pipeline state
  to `idle` with no active job.
- Verification commands:
  - `python -m unittest discover -s tests -p "test_state_store.py" -v`
  - `python -m unittest discover -s tests -p "test_setup_and_start.py" -v`
  - `python -m unittest discover -s tests -v`
  - `python -m compileall -q tenminvideomaker scripts __init__.py`
  - `git diff --check`
- Results: all 98 tests passed under system Python and the Easy Install embedded Python; compilation and
  `git diff --check` passed.
- Live recovery: job `20260725-0115` was atomically abandoned with all eight unfinished scenes marked `cancelled`.
  The already-running supervisor observed `idle` on its next scheduled tick, successfully sent a fresh request,
  and transitioned to `waiting_for_grok` without restarting ComfyUI or the supervisor.

### 2026-07-24 — strict LTXV 2.3 LoRA stage isolation

- Changed files: `tenminvideomaker/constants.py`, `contracts.py`, `assets.py`, `server_api.py`, `nodes.py`,
  `supervisor.py`, `workflow_builder.py`, `state_store.py`; focused tests in `tests/test_assets.py`,
  `test_contracts.py`, `test_server_api.py`, `test_state_store.py`, `test_supervisor.py`, and
  `test_workflow_builder.py`; revalidated workflow exports; `README.md`, `docs/architecture.md`,
  `docs/user-guide.md`, and this file.
- Cause: Grok repeated an Anima character LoRA under a scene I2V alias, and the prior router attached every
  `scenes[].i2v.loras[]` item to the LTX model without checking model family. The installed file bypassed public
  metadata lookup, contaminating completed clips.
- Routing: effective I2V LoRAs exclude all effective T2I identities. Remaining dynamic I2V assets must be verified
  by Civitai as `LTXV 2.3` before local lookup or download. Validated filenames use an `i2v:` context key, and
  workflow construction refuses an unvalidated dynamic I2V mapping. Mandatory DMD and JoyAI remain exact local
  constants.
- Recovery: `requeue_i2v_for_job` atomically preserves cached frame paths, clears clip paths and I2V attempts, and
  returns the saved job to asset resolution. Move contaminated clips to a project-owned quarantine directory before
  invoking it.
- Reproduction: provide one Civitai version in both `character.lora` and a differently named scene I2V entry, plus
  one actual LTXV 2.3 scene LoRA. The generated I2V graph must contain DMD, JoyAI, and only the verified LTX asset.
- Verification commands:
  - `python -m compileall -q tenminvideomaker tests scripts __init__.py`
  - `python -m unittest discover -s tests -v`
  - `python scripts/export_workflows.py --install-approved-shared-copies`
  - `git diff --check`

### 2026-07-24 — watermarked metadata-free Discord delivery

- Changed files: `.env.example`, `tenminvideomaker/delivery.py`, `configuration.py`, `state_store.py`, `supervisor.py`,
  `workflow_builder.py`, `scripts/run_supervisor.py`, `setup_and_start.py`, `export_workflows.py`, all six generated
  workflow JSON files, focused delivery/configuration/setup/workflow tests, `README.md`, `docs/architecture.md`,
  `docs/user-guide.md`, and this file.
- Reference settings: from the explicitly approved reference workflow only, use `wm.png`, bottom-right, scale 0.70,
  bicubic, transparency 0.40, rotation 0, padding 20×20, optical correction off/0.40, three switches, no fade,
  margin 0.10, fixed position, seed 0.
- Routing: the clean image still feeds `10MinVideoMaker_SaveSceneFrame`; its parallel watermarked edge feeds only
  `DiscordSendSaveImage` at PNG quality 100. Clean normalized video still feeds temporary `VHS_VideoCombine`; its
  parallel watermarked edge plus decoded audio feeds only `DiscordSendSaveVideo` at H.264 quality 65 and 24 fps.
- Privacy/storage: both Discord nodes disable `save_output`, prompt inclusion, workflow JSON, CDN storage, and
  GitHub updates. The imported webhook is DPAPI-encrypted and absent from tracked files. Versioned workflows use a
  placeholder; approved shared copies receive the encrypted runtime value during export.
- Verification commands:
  - `python -m compileall -q tenminvideomaker tests scripts __init__.py`
  - `python -m unittest discover -s tests -v`
  - live `/object_info` validation of both delivery graphs
  - `python scripts/export_workflows.py --install-approved-shared-copies`
  - `git diff --check`

### 2026-07-25 — LTX 2.x compatibility and active-job cancellation

- Changed files: `tenminvideomaker/assets.py`, `tenminvideomaker/comfy_http.py`,
  `scripts/setup_and_start.py`, focused tests in `tests/test_assets.py`, `tests/test_comfy_http.py`, and
  `tests/test_setup_and_start.py`, plus `README.md`, `docs/architecture.md`, `docs/user-guide.md`, and this file.
- Cause: Civitai reports some compatible LTX 2 LoRAs with the compact `LTXV2` base label, which the exact
  `LTXV 2.3` comparison rejected. The launcher also offered its saved-job choice only from `error` or the uncommon
  unfinished `waiting_for_grok` state, so restarting during T2I/I2V silently resumed the saved batch.
- Routing: the I2V gate now accepts verified LTX 2.x family labels for the LTX 2.3 target while retaining rejection
  of LTX 1.x and image-model LoRAs. The launcher prompts for every active saved state, can resume an assembly-only
  job, and on abandonment cancels pending/running ComfyUI prompts only when their queue metadata matches the
  `10MinVideoMaker-supervisor` client ID.
- Live recovery: supervisor PID 10896 and its scene prompt were stopped/interrupted; job `20260725-0505` was
  abandoned with five successful scenes preserved and fifteen unfinished scenes marked `cancelled`. No artifacts
  were deleted.
- Verification commands:
  - `python -m unittest discover -s tests -v`
  - `python -m compileall -q tenminvideomaker tests scripts __init__.py`
  - embedded-Python unit tests and compilation
  - live no-render resolution of an existing Civitai `LTXV2` LoRA through the loopback route
  - `git diff --check`

### 2026-07-25 — active redacted console progress

- Changed files: `.env.example`, `tenminvideomaker/comfy_http.py`, `supervisor.py`, `configuration.py`,
  `scripts/setup_and_start.py`, focused tests in `tests/test_comfy_http.py`, `test_configuration.py`, and
  `test_supervisor.py`, plus `README.md`, `docs/architecture.md`, `docs/user-guide.md`, and this file.
- Behavior: the supervisor runs an independent status reporter every 15 seconds by default, so the console remains
  active while the main loop sleeps, resolves assets, renders, or stitches. Heartbeats show durable state,
  job/scene IDs, and redacted ComfyUI queue counts. INFO phase logs cover Gmail checks, LoRA results, cache reuse,
  scene attempts, and assembly.
- Privacy: queue workflow data and generation prompts are never formatted into the log. Authentication values,
  download URLs, and the Discord webhook remain excluded.
- Configuration: `TENMIN_STATUS_INTERVAL_SECONDS` is an allowlisted non-secret setting and appears as **Console
  heartbeat seconds** in the same optional-settings editor as the other supervisor controls.
- Verification commands:
  - `python -m unittest discover -s tests -v`
  - embedded-Python unit tests and compilation
  - live supervisor restart while `waiting_for_grok`, followed by observed heartbeat output
  - `git diff --check`

### 2026-07-25 — non-consuming Gmail polling and Grok seed compatibility

- Changed files: `tenminvideomaker/mail.py`, `state_store.py`, focused tests in `tests/test_mail.py` and
  `tests/test_state_store.py`, plus `README.md`, `docs/architecture.md`, `docs/user-guide.md`, and this file.
- Cause: Gmail candidates were fetched with `RFC822`, which can set `\Seen` before the state machine decides which
  message to accept. The new `20260725-0614` Drive handoff also contained invalid JSON integer syntax,
  `"seed": 012345678`; a separate valid candidate resolved to already-accepted job `20260725-0115`.
- Mail routing: use `BODY.PEEK[]`, log redacted candidate/rejection/duplicate decisions, and permit an earlier
  `mail_messages.job_id=NULL` rejection record to be atomically upgraded when the same message later parses.
  Previously accepted message IDs and job IDs remain de-duplicated.
- Parser compatibility: after normal JSON parsing fails, strip redundant leading zeroes only from unquoted integer
  values keyed by `seed` or `original_seed`, then run the unchanged strict contract validator. JSON strings and all
  other invalid syntax remain untouched.
- Reproduction: deliver an exact-subject Drive envelope whose full file contains `"seed": 012345678`, and another
  unread message for an existing job ID. The first must normalize and validate; the second must be reported as a
  duplicate without being accepted.
- Verification commands:
  - `python -m unittest discover -s tests -p "test_mail.py" -v`
  - `python -m unittest discover -s tests -p "test_state_store.py" -v`
  - full system and embedded-Python unit suites and compilation
  - live no-render PEEK/parse check against the two mailbox candidates
  - `git diff --check`
- Results: all 125 tests passed under system Python and Easy Install embedded Python; both compilation passes and
  `git diff --check` passed. The live Drive file parsed as job `20260725-0614` with ten scenes after two seed
  normalizations. No workflow was queued during validation.

### 2026-07-25 — human-review GUI, D-drive storage, and versioned remakes

- Changed files: `.env.example`, `Start 10MinVideoMaker.bat`, `scripts/run_gui.py`, `scripts/run_supervisor.py`,
  `scripts/setup_and_start.py`; `tenminvideomaker/storage.py`, `ownership.py`, `review.py`, `gui_app.py`,
  `gui_service.py`, `state_store.py`, `supervisor.py`, `workflow_builder.py`, `artifacts.py`, `assembly.py`,
  `comfy_http.py`, `configuration.py`, `mail.py`, `nodes.py`, and `server_api.py`; `web/index.html`,
  `web/styles.css`, `web/app.js`; focused tests in `tests/test_storage.py`, `test_review.py`, `test_gui_app.py`,
  `test_gui_service.py`, `test_state_store.py`, `test_workflow_builder.py`, `test_artifacts.py`,
  `test_configuration.py`, and `test_setup_and_start.py`; `README.md`, `docs/architecture.md`,
  `docs/user-guide.md`, `TODO.md`, and this file.
- Framework: FastAPI uses the already-installed Easy Install embedded-Python packages. The frontend is dependency-free
  HTML/CSS/JavaScript, binds only to `127.0.0.1:8765`, and obtains sampler choices from live ComfyUI node contracts.
- Persistence: new runtime state is rooted at `D:\LTX_Supervisor_Storage`. The one-time migration uses SQLite
  backup and file copies, materializes source payloads, rewrites only the migrated database paths, and leaves old
  state/media untouched.
- Review/routing: Gmail claims auto-queue in normal GUI sessions; the explicit GUI review-hold launch option enters
  `awaiting_review`. The UI exposes human-readable source context, prompts,
  large seeds, effective LoRAs, both model-specific T2I passes, Pony detailer, both I2V samplers/sigmas, CFG and
  conditioning controls, chunking, upscaler, and locked production invariants. Edits are reparsed through the exact
  job contract and use the existing workflow builders through explicit overrides.
- Revisions/batches: only `video_only` and `image_and_video` exist. Drafts can span jobs; submission creates durable
  numbered scene revisions and manifests. Video-only reuses a verified cached frame. Image + Video writes a new
  revision frame and passes that exact path into I2V.
- Ownership/collision: a project lock and exact legacy-process detection enforce one supervisor owner. Active-render
  collisions either wait or cancel only this project's prompts, preserve the interrupted job history, and then run
  the queued revision batch.
- Live activation: GUI startup verifies that `10MinVideoMaker_SaveSceneFrame` exposes the `revision` input. It may
  invoke the existing path-verified ComfyUI restart only when the entire queue is empty; otherwise startup refuses
  and leaves active work untouched.
- Reproduction: launch `Start 10MinVideoMaker.bat`, select a historical job/scene, mark multiple scenes, edit a seed
  or prompt, choose a supported remake mode, and submit. Do not submit during no-render validation.
- Verification commands:
  - `python -m unittest discover -s tests -q`
  - Easy Install embedded-Python full unit suite with the repository inserted into `sys.path`
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `node --check web/app.js`
  - `git diff --check`

### 2026-07-25 — unattended-first GUI intake

- Changed files: `scripts/run_gui.py`, `tenminvideomaker/gui_service.py`, `web/app.js`, `.env.example`,
  `README.md`, `docs/user-guide.md`, this file, and focused supervisor/GUI tests.
- Intake policy: the standard GUI launch explicitly overrides legacy review configuration and starts every valid
  Gmail handoff automatically. `Start 10MinVideoMaker.bat --hold-new-jobs-for-review` is the opt-in testing mode;
  only that session claims new jobs as `awaiting_review` and exposes **Approve & Queue Job**.
- Reproduction: launch the normal batch file and send a valid `LTX_JOB_COMPLETE` handoff; it proceeds without
  approval. Relaunch with `--hold-new-jobs-for-review` to verify the approval control appears instead.
- Verification commands:
  - `python -m unittest discover -s tests -q`
  - Easy Install embedded-Python full unit suite with the repository inserted into `sys.path`
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `node --check web/app.js`
  - `git diff --check`

### 2026-07-25 — stage-batched model residency

- Changed files: `tenminvideomaker/supervisor.py`, `tenminvideomaker/gui_service.py`,
  `tests/test_supervisor.py`, `tests/test_gui_service.py`, `README.md`, `docs/architecture.md`,
  `docs/user-guide.md`, and this file.
- Routing: normal jobs now execute all required T2I frames before the LTX I2V pass. GUI remake batches preflight
  every selected revision, generate all image+video frames first (grouped by T2I family), then render all
  successful image+video and video-only revisions through LTX. The free-memory call remains only at phase/job
  boundaries.
- Reproduction: submit a two-scene job and verify the prompt order is T2I scene 1, T2I scene 2, I2V scene 1,
  I2V scene 2. Submit a five-item remake batch with four image+video items and one video-only item; all four
  frames must complete before any of the five I2V prompts begin.
- Verification commands:
  - `python -m unittest discover -s tests -q`
  - Easy Install embedded-Python full unit suite with the repository inserted into sys.path
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `git diff --check`

### 2026-07-25 — production configuration isolation repair

- Changed files: `tenminvideomaker/configuration.py`, `storage.py`, focused tests in
  `tests/test_configuration.py` and `tests/test_storage.py`, plus this file.
- Cause: configuration helpers defaulted every caller—including a temporary test project—to the production
  `D:\LTX_Supervisor_Storage` root. An earlier unit test therefore wrote its OAuth placeholders into the real
  settings and secret files. A prematurely initialized empty D-drive SQLite database could also block the
  first legacy-history backup.
- Repair: only this exact repository root may select the configured production D-drive layout implicitly.
  Temporary/embedded project roots default to their own `runtime-storage` directory. First migration detects and
  preserves an empty destination database before backing up a populated legacy database.
- Recovery: preserve the placeholder files as timestamped D-drive backups, restore the untouched legacy `.env`
  and DPAPI secret store, then run the normal non-destructive migration and credential validation.

### 2026-07-25 — scrollable GUI libraries and readable project names

- Changed files: `tenminvideomaker/gui_app.py`, `web/styles.css`, `web/app.js`,
  `tests/test_gui_app.py`, `README.md`, `docs/user-guide.md`, and this file.
- Cause: the fixed-height project/scene list calculations were inside CSS grid children whose automatic minimum
  height prevented shrinking; the page body intentionally hides outer overflow, so rows beyond the viewport were
  clipped instead of becoming scrollable.
- Repair: library panels are bounded flex columns with `min-height: 0`; each list owns stable independent vertical
  overflow. Project labels use `Character · MM/DD/YYYY`, preferring the source creation date, then the encoded job
  date, then the stored creation timestamp. The job ID remains the API and state-machine key.
- Reproduction: load a history containing more projects and scenes than fit vertically. Both left columns must
  scroll to their final card, while project cards and the selected-project heading show the readable label.

### 2026-07-25 — divisible LTX 2.3 spatial-upscale geometry

- Changed files: `tenminvideomaker/constants.py`, `tenminvideomaker/workflow_builder.py`,
  `tenminvideomaker/workflow_export.py`, `tenminvideomaker/nodes.py`, focused assembly/node/supervisor/workflow
  builder tests, regenerated `workflows/10MinVideoMaker_*`, `README.md`, `docs/architecture.md`,
  `docs/user-guide.md`, and this file.
- Decision: all new production image and video output is 768×1344 at 24 fps. This is a 4:7 ratio (about 1.59%
  wider than exact 9:16), but it makes both x2-route stages valid on LTX's 32-pixel dimension grid:
  384×672 first pass and 768×1344 final pass.
- Routing: `EmptyLTXVLatentVideo` starts at 384×672 and the x2 spatial upscaler reaches 768×1344 exactly. The
  prior 704×1216 correction `ImageScale` is removed; decoded video feeds `VHS_VideoCombine` and the parallel
  Discord watermark branch directly, avoiding a post-decode crop/resize.
- Reproduction: build `10MinVideoMaker_I2V_LTX23_TwoPass` and verify its first-pass `ImageScale` and
  `EmptyLTXVLatentVideo` are both 384×672, all four route dimensions are divisible by 32, and no node titled
  `Normalize decoded video` exists. Verify T2I and FFprobe preflight use 768×1344.
- Verification commands:
  - `python -m unittest discover -s tests -q`
  - Easy Install embedded-Python full unit suite with repository inserted into `sys.path`
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `git diff --check`
- Results: 143 tests passed under both system Python and Easy Install embedded Python. Regenerated repository and
  approved shared GUI workflows passed live `/object_info` validation with 384×672 first-pass and 768×1344 final
  profile metadata. No render was queued or interrupted.

### 2026-07-26 — mobile GUI, protected LAN access, and local LoRA pickers

- Changed files: `configuration.py`, `gui_app.py`, `scripts/run_gui.py`, `scripts/setup_and_start.py`,
  `web/app.js`, `web/styles.css`, focused GUI/configuration/setup tests, `.env.example`, `README.md`,
  `docs/architecture.md`, `docs/user-guide.md`, and this file.
- LAN routing: default remains `127.0.0.1:8765`; the optional launcher action stores a non-secret enable flag and a
  12+ character DPAPI-protected password. Only that explicit mode binds the GUI to `0.0.0.0`; every non-loopback
  browser/API/media/event request must authenticate as fixed user `10min`. ComfyUI remains loopback-only.
- Mobile routing: below 760px the workspace becomes stacked, library/scene sections retain their own scroll regions,
  detail controls become one column, and selecting a scene moves the viewport to its editor.
- LoRA routing: `/api/options` queries live `LoraLoader` and `LoraLoaderModelOnly` contracts. T2I and I2V use their
  separate option lists. Picking a local filename fills the human-facing name only; existing contract validation,
  download identity, and the I2V LTX 2.x gate remain authoritative.
- Reproduction: configure **Mobile LAN access** in launcher optional settings, restart GUI while no ComfyUI prompt is
  active, open logged private-IP URL on a phone, and sign in. Mark a scene for remake and use each stage's local LoRA
  picker; submit only after reviewing the existing URL and I2V compatibility.
- Verification commands:
  - `python -m unittest discover -s tests -q`
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `node --check web/app.js`
  - `git diff --check`
- Results: 144 embedded-Python tests passed. Live no-render ComfyUI contracts exposed 115 selectable LoRA filenames
  for each of `LoraLoader` and `LoraLoaderModelOnly`. No generation, download, or ComfyUI restart occurred during
  validation.

### 2026-07-26 — post-reboot GUI ComfyUI startup guard

- Changed files: `scripts/setup_and_start.py`, `scripts/run_gui.py`, focused setup/GUI tests, `README.md`,
  `docs/architecture.md`, `docs/user-guide.md`, and this file.
- Cause: `Start 10MinVideoMaker.bat` runs setup with `--setup-only` before it launches `run_gui.py`. That path
  skipped the existing local ComfyUI health/start guard, so a GUI launch immediately queried `/object_info` and
  failed with connection refused after a PC reboot.
- Repair: `ensure_comfyui` is the shared guard for both the console and GUI launchers. When the authorized loopback
  API is unavailable, it invokes only the existing path-verified restart helper, which starts the unchanged Easy
  Install `Start ComfyUI.bat` and therefore preserves Sage Attention flags. The GUI waits for health before any
  node-contract query or supervisor ownership action.
- Reproduction: restart the PC, leave ComfyUI closed, then double-click `Start 10MinVideoMaker.bat`. The launcher
  must report that it is starting the verified local server and open the GUI only after port 8188 is healthy.
- Verification commands:
  - Easy Install embedded-Python focused `test_setup_and_start.py` and `test_gui_app.py` suites with repository
    and `tests` paths inserted into `sys.path`
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `git diff --check`
