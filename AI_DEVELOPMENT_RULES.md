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
- Long-scene continuation uses feature flag `ltx_chunked_continuation_v1` and resolved strategy
  `ltx23_latent_overlap_v1`. Its initial model window is at most 121 frames/120 transitions. Each full
  extension uses the official `LTXVExtendSampler`, regenerates a fixed 24-frame overlap, and adds at most 96 new
  transitions. The exact presentation timeline is `round(seconds × 24)` frames; generation rounds transitions up
  to a multiple of eight, forms an `8n + 1` master, and trims only after delayed-commit assembly.
- Continuation video uses the style-stable stage-one handoff as source of truth. Tiled-decode it at 384×672 and
  deterministically enlarge it with `RealESRGAN_x2.pth`; never substitute the diffusion-repainted stage-two video.
  The second LCM/spatial pass remains required for synchronized audio. Later direct video decodes contain an
  eight-frame causal preroll. Assembly owns initial video/audio `0:104`, later non-final video `8:104` with audio
  `16:112`, and the corresponding shortened final ranges. Do not replace this route with literal-last-frame
  continuation, an image-model regeneration, or reinterpret a nonzero latent token as latent zero.
- Continuation rollout modes are `disabled`, `explicit`, and `auto`; `explicit` is the portable fail-safe default. A scene at or
  below 121 generation frames always uses the legacy route. Explicit mode requires
  `i2v.continuation.enabled=true`; auto mode permits an explicit false opt-out but must fail closed before startup
  without `<TENMIN_STORAGE_ROOT>\state\continuation-validation-v2.json` (default
  `D:\LTX_Supervisor_Storage\state\continuation-validation-v2.json`). That approval must match a hash covering the
  current continuation generation, routing, and recovery implementation plus hashes covering every node contract
  used by the representative live continuation graphs; record valid hashes, sources, and licenses for every
  required external asset; complete the four bounded comparison generations with positive latent-overlap peak
  VRAM; and accept all no-OOM/guider/motion/style/anatomy/A/V-profile/runtime decisions. Never invalidate or convert a
  completed legacy revision merely because the feature is enabled.
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
- Continuation remakes remain scene-level. Video-only reuses the selected cached frame but creates a new chunk chain
  from chunk zero; image-and-video creates a new frame and a new chain from chunk zero. Do not expose arbitrary
  per-chunk remake selection until the user-facing dependency and downstream-invalidation design is implemented.
- Render scheduling is model-residency aware on the 16 GB GPU. For each automatic job, complete every required T2I
  frame before one intentional model release, then complete every required LTX I2V clip before releasing LTX.
  Remake batches must preflight selected revisions, group image+video remakes by Anima/Pony family, render all of
  those frames, then run every eligible video (including video-only remakes) as one LTX phase. Never call
  ComfyUI's free-memory endpoint between scenes in the same phase.
- A continuation chunk attempt owns one bounded low-resolution handoff, deterministic x2 video upscale, and one
  sampled second-pass audio latent. Persist only plain LTX video latents in safetensors with `[1, 128, frames, height, width]`
  floating-point samples and the explicitly supported auxiliary keys. Flush and atomically rename the tensor,
  record identity/size/descriptors/SHA-256 in its JSON manifest, write attempt `COMPLETE.json` last, then select the
  attempt in SQLite. Retain only the bounded tail required by the planned window; never retain an accumulating
  full-scene latent. Every load must pass the plan's exact `expected_temporal_tokens` so identity, hash, descriptor,
  shape, and token-count validation all precede reuse. File existence alone is never proof of success.
- Continuation plans and attempt inputs are immutable. Store unsigned-64-bit seeds as text, bind each successor to
  the selected predecessor's artifact hash, and invalidate the changed chunk plus all descendants when upstream
  lineage or artifact verification changes. Recovery may resume stage two from a valid stage-one checkpoint and
  may reuse a valid completed stage-two checkpoint plus raw window; otherwise restart at the earliest invalid
  dependency.
- Persist each continuation stage prompt ID and workflow hash immediately after `/prompt` returns and before
  blocking. Every continuation generation/decode graph must use the always-reexecuted
  `10MinVideoMaker_FreshCheckpoint` before dynamic LoRAs: ComfyUI can reuse static checkpoint-loader outputs across
  graph/client boundaries despite unique node IDs and fresh wrapper/conditioning objects. Restart recovery must
  reclaim successful history or wait for the exact queued base-project prompt; an absent prompt may requeue only the
  same immutable workflow. Do not call `/free` between continuation stages. Persist Discord delivery ownership
  similarly, but fail closed on ambiguous queue/history and never automatically resend an uncertain side effect.
- Continuation scene assembly must validate every raw window at 768×1344 and exact 24/1 CFR with its planned frame
  count, audio, lossless FFV1 codec, and yuv444p pixel format. Store each raw attempt as unwatermarked
  `window.mkv`; never H.264-encode individual windows. Use the independently aligned video/audio slices above,
  apply 100 ms non-overlapping audio edge fades, and perform the route's one H.264
  High/yuv420p CRF-19 scene encode with closed GOP 48 and stereo 48 kHz AAC. Validate before atomically replacing
  the clean revision-facing `video.mp4`, then write `assembly\COMPLETE.json`.
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
- The legacy single-window LTX x2 spatial-upscale route uses an internal 384×672 first-pass latent and produces the
  fixed 768×1344 saved clip. Every first-pass and production axis is divisible by 32; route decoded frames directly
  to video combine. Continuation is the documented exception: deterministic `RealESRGAN_x2.pth` enlarges its
  style-stable first-pass decode. Do not expose or save another production size or post-upscale crop stage.
- I2V uses `VHS_VideoCombine` temporary output. The supervisor retrieves its exact history metadata through
  `/view` and writes the project clip into the matching versioned directory below
  `D:\LTX_Supervisor_Storage\jobs`; do not scan or move shared output folders.
- Controlled restart must verify the port-8188 owner is the expected Easy Install embedded Python executable before
  stopping it. Never weaken that path check. GUI startup must verify Save Scene Frame revision support, both chunk
  latent nodes' exact `stage1_handoff`/`stage2_video`/`stage2_audio` artifact options, and Load Chunk Latent's
  `expected_temporal_tokens`. A stale-node restart is permitted only with an empty ComfyUI queue, and startup must
  verify the contracts again afterward.
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
- Continuation worker graphs must contain no watermark or Discord sender. Raw FFV1/yuv444p `window.mkv` artifacts, assembled
  revision `video.mp4`, and project finals remain unwatermarked. Only after raw scene assembly succeeds may a
  separate delivery graph reload that scene, watermark it, and send it with `save_output=false`.
- Manual project finals are explicit, durable FFmpeg-only requests. Snapshot each included scene's latest successful
  revision at click time and concatenate that immutable selection only after active project work has ended. The
  per-scene inclusion flag is manual-final-only: automatic first-run assembly must retain its own all-successful
  routing and never be suppressed by GUI exclusions.

## Implementation and testing

- Use `apply_patch` for source and documentation edits.
- Add focused regression tests for every routing fix or bug fix.
- Prefer no-render validation before image, video, or audio generation. Do not generate media, download models, or install dependencies without explicit task authorization.
- `python scripts\validate_continuation_workflows.py` is the continuation graph gate: it builds representative
  initial/later/final stage-one and stage-two graphs plus delivery, checks only live `/object_info`, and must never
  queue a prompt. Passing it does not establish seam quality, anatomy quality, runtime, or peak VRAM.
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

### 2026-07-26 — unattended optional settings and mobile drill-down

- Changed files: `scripts/setup_and_start.py`, `web/index.html`, `web/styles.css`, `web/app.js`, focused setup/GUI
  tests, `README.md`, `docs/architecture.md`, `docs/user-guide.md`, and this file.
- Console routing: only **Change optional environment settings before starting? [y/N]** uses a ten-second Windows
  console timeout. On expiry or invalid input it safely chooses its existing **No** default; all credential,
  destructive/recovery, and other confirmation prompts remain explicit.
- Mobile routing: at `max-width: 760px`, the browser exposes exactly one view at a time—projects, selected project's
  scenes, or selected scene detail. Back controls reverse the hierarchy, while a sticky detail-level dropdown changes
  scenes without returning to the list. The desktop grid is unaffected outside that media query.
- Media routing: scene videos retain native `controls` and add `playsinline`/`webkit-playsinline`; the isolated,
  full-size video surface preserves browser-native mobile fullscreen controls above the media background.
- Reproduction: leave the optional-settings console question unanswered for ten seconds and verify startup continues.
  On a phone-width browser, select project → scene, switch scenes through the sticky dropdown, return through both
  back controls, and open a rendered video through its native fullscreen control.
- Verification commands:
  - Easy Install embedded-Python focused `test_setup_and_start.py` and `test_gui_app.py` suites with repository
    and `tests` paths inserted into `sys.path`
  - `node --check web/app.js`
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `git diff --check`

### 2026-07-26 — pin durable video downloads to the raw VHS output

- Changed files: `tenminvideomaker/comfy_http.py`, `supervisor.py`, `gui_service.py`, focused HTTP regression tests,
  `README.md`, `docs/architecture.md`, `docs/user-guide.md`, and this file.
- Cause: the completed ComfyUI history may contain both the raw `VHS_VideoCombine` MP4 and the watermarked
  `DiscordSendSaveVideo` MP4. The prior helper searched all output nodes for the first MP4 and could store the
  Discord version as the durable scene clip.
- Repair: `WorkflowBuild.output_node_id` is now mandatory when selecting video metadata. Normal jobs and GUI remake
  batches search only below that raw VHS node. Missing/invalid raw output metadata fails safely instead of falling
  back to any other MP4.
- Recovery: a historical watermarked `video.mp4` has no clean replacement in the project-owned revision folder.
  Re-run it as **Video Only** from its clean cached `frame.png` after this fix is active; do not regenerate T2I.
- Reproduction: construct history with a raw VHS MP4 and a watermarked Discord MP4 under different output node IDs.
  Selecting the raw node must return only the raw metadata regardless of mapping order.
- Verification commands:
  - Easy Install embedded-Python focused `test_comfy_http.py`, supervisor, and GUI-service suites with repository
    and `tests` paths inserted into `sys.path`
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `git diff --check`

### 2026-07-26 — revision-specific remake review values

- Changed files: `web/app.js`, `tests/test_gui_app.py`, `README.md`, `docs/architecture.md`,
  `docs/user-guide.md`, and this file.
- Cause: the scene-detail endpoint already returned each immutable revision's saved parameter document, but the
  browser changed only the selected revision's frame/video URLs. The form always remained bound to the original
  source-scene document, so reviewing a remake could display version 1 prompts and sampler settings.
- Repair: selecting **Result version** now changes preview and working form together. New unsent edits are held per
  selected source revision in the browser, while the remake tray contains only the scene/version currently chosen
  for submission. Turning off **Mark for remake** discards those temporary edits and restores the selected immutable
  revision record.
- Reproduction: create revisions 2 and 3 with distinct T2I prompt/I2V sampler values, select each item in the
  result-version dropdown, and verify its fields match its generation manifest before marking it for another remake.
- Verification commands:
  - Easy Install embedded-Python `tests/test_gui_app.py`
  - `python -m unittest discover -s tests -q`
  - `node --check web/app.js`
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `git diff --check`

### 2026-07-26 — explicit manual project finals after remakes

- Changed files: `tenminvideomaker/state_store.py`, `tenminvideomaker/gui_service.py`,
  `tenminvideomaker/gui_app.py`, `web/index.html`, `web/styles.css`, `web/app.js`, focused state/GUI-service/GUI-app
  tests, `README.md`, `docs/architecture.md`, `docs/user-guide.md`, and this file.
- Architecture: a manual-final request persists a snapshot of every included scene's latest successful immutable
  revision and clip path. A request runs in the existing single GUI worker only once normal project work is idle;
  it validates the fixed profile and uses the standard FFmpeg copy concat to overwrite the project final. It never
  queues a ComfyUI graph, loads a model, or changes pipeline state.
- Routing: `include_in_manual_final` is a durable per-scene GUI choice. It filters only manual-final request
  snapshots; the automatic first-run completion concat continues to use all successful original scene records.
- Reproduction: remake a scene, exclude an unwanted scene through its scene editor, then press **Render project
  final** beneath the selected project's scenes. The request must use the newest successful revision for every
  included scene, wait behind active work, and fail clearly if an included scene lacks a successful video.
- Verification commands:
  - `python -m unittest discover -s tests -p "test_state_store.py" -v`
  - `python -m unittest discover -s tests -p "test_gui_service.py" -v`
  - Easy Install embedded-Python `tests/test_gui_app.py`
  - `node --check web/app.js`
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `git diff --check`

### 2026-07-26 — live skip of error-paused job 20260726-1823

- Changed files: durable state only (D:\LTX_Supervisor_Storage\state\pipeline.sqlite3); no application code.
- Cause: job 20260726-1823 paused the singleton pipeline in `error` after every scene failed
  (`Scene frame must be 704x1248; received 768x1344`). While in `error`, the supervisor refuses new Gmail jobs.
- Existing skip path: `PipelineStateStore.abandon_job` already preserves the job payload/history, marks unfinished
  scenes cancelled, sets job status cancelled, and returns the pipeline to idle. Surfaces: launcher answer
  **no** to “Resume this saved job…”, or direct store call. There was no browser GUI button for this case.
- Live recovery: called `abandon_job('20260726-1823', reason='Skipped by operator after no successful scenes;
  preserved for later remake.')`. All 15 scenes became cancelled; payload remains loadable for later remake;
  pipeline is idle with no active job. The running GUI worker requests the next job on its next scheduled tick
  without restart.
- Note for later remake: no successful frames were cached, so only **image_and_video** remake is possible, and the
  704x1248 vs 768x1344 frame-size rejection should be fixed first.

### 2026-07-27 — GUI Cancel project advances to the next Gmail job

- Changed files: `tenminvideomaker/gui_service.py`, `gui_app.py`, `supervisor.py`, `web/index.html`,
  web/app.js, focused GUI/supervisor tests, README.md, docs/architecture.md, docs/user-guide.md,
  and this file.
- User surface: top-bar **Cancel project** appears while a job is held in active render, `error`, or
  `awaiting_review`. Confirmation abandons via the existing atomic `abandon_job` path (history preserved,
  unfinished scenes cancelled), cancels only project-owned ComfyUI prompts when rendering, and wakes the
  worker. `POST /api/pipeline/cancel-current` and status field `can_cancel_current_project` back the control.
- Idle routing: after cancel/abandon lands on idle, the next tick checks unread LTX_JOB_COMPLETE mail first
  and only sends a new Grok request when no valid handoff is waiting. Normal job completion still uses
  `_request_next_job` directly and does not pass through this idle poll-first path.
- Verification commands:
  - `python -m unittest discover -s tests -p "test_gui_service.py" -v`
  - `python -m unittest discover -s tests -p "test_gui_app.py" -v`
  - `python -m unittest discover -s tests -p "test_supervisor.py" -v`
  - `node --check web/app.js`
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `git diff --check`

### 2026-07-29 — normalize numeric Civitai IDs in Gmail handoffs

- Changed files: `tenminvideomaker/contracts.py`, `tests/test_contracts.py`, and this file.
- Contract compatibility: Civitai `model_id` and `version_id` accept either a JSON positive integer or an
  ASCII digit-only string from a Google Drive/Gmail handoff, then normalize it to the typed integer used by
  asset validation. Empty, signed, whitespace-padded, decimal, nonnumeric, boolean, and nonpositive values
  remain rejected.
- Reproduction: submit a job whose `character.lora.version_id` is `"3184055"`; it must validate exactly as
  numeric `3184055`, while `"3184055.0"` must still fail contract validation.
- Verification commands:
  - `python -m unittest discover -s tests -p "test_contracts.py" -v`
  - `python -m compileall -q tenminvideomaker tests`
  - `git diff --check`

### 2026-07-29 — collision-safe reused Grok job IDs

- Changed files: `tenminvideomaker/contracts.py`, `tenminvideomaker/state_store.py`,
  `tenminvideomaker/mail.py`, `tests/test_state_store.py`, and this file.
- Inbound routing: when Grok reuses an accepted `job_id`, compare a canonical SHA-256 fingerprint of the parsed
  job document with only `job_id` and `created_at` omitted. Identical content is marked seen and skipped; distinct
  content is accepted under a deterministic project-local ID such as `{source_job_id}-local-2`, while preserving
  `source_job_id` in the stored raw payload for audit. The suffix remains inside the 128-character job-ID limit.
- Verification commands:
  - `python -m unittest discover -s tests -p "test_state_store.py" -v`
  - `python -m unittest discover -s tests -p "test_mail.py" -v`
  - `python -m compileall -q tenminvideomaker tests`
  - `git diff --check`

### 2026-07-30 — implemented beta bounded LTX 2.3 continuation

- Changed files:
  - Configuration/docs/example: `.env.example`, `README.md`, `AI_DEVELOPMENT_RULES.md`, `TODO.md`,
    `docs/architecture.md`, `docs/research/ltx23_chunked_continuation_plan.md`, `docs/user-guide.md`,
    `examples/example_job.json`.
  - Entrypoints: `scripts/run_gui.py`, `scripts/run_supervisor.py`, `scripts/setup_and_start.py`,
    `scripts/validate_continuation_workflows.py`.
  - Runtime: `tenminvideomaker/chunk_artifacts.py`, `tenminvideomaker/chunk_assembly.py`,
    `tenminvideomaker/comfy_http.py`, `tenminvideomaker/configuration.py`,
    `tenminvideomaker/continuation.py`, `tenminvideomaker/continuation_renderer.py`,
    `tenminvideomaker/continuation_validation.py`, `tenminvideomaker/continuation_workflow.py`,
    `tenminvideomaker/contracts.py`, `tenminvideomaker/gui_app.py`, `tenminvideomaker/gui_service.py`,
    `tenminvideomaker/nodes.py`, `tenminvideomaker/review.py`, `tenminvideomaker/state_store.py`,
    `tenminvideomaker/storage.py`, `tenminvideomaker/supervisor.py`, and
    `tenminvideomaker/workflow_builder.py`.
  - GUI: `web/app.js`, `web/index.html`, `web/styles.css`.
  - Tests: `tests/test_chunk_artifacts.py`, `tests/test_chunk_assembly.py`, `tests/test_comfy_http.py`,
    `tests/test_continuation.py`, `tests/test_continuation_renderer.py`,
    `tests/test_continuation_validation.py`, `tests/test_continuation_workflow.py`,
    `tests/test_contracts.py`, `tests/test_gui_app.py`, `tests/test_gui_service.py`,
    `tests/test_nodes.py`, `tests/test_review.py`, `tests/test_run_supervisor.py`,
    `tests/test_setup_and_start.py`, `tests/test_state_store.py`, `tests/test_storage.py`, `tests/test_supervisor.py`,
    `tests/test_validate_continuation_workflows.py`, and `tests/test_workflow_builder.py`.
- Contract/planning: the safe schema-version-2 example demonstrates optional `i2v.continuation`,
  `i2v.continuity`, and ordered `i2v.segments`. The planner resolves a 30-second scene to transition
  contributions `120,96,96,96,96,96,96,24`, a 721-frame generation master, and an exact 720-frame presentation
  timeline. The 121-frame nominal window, 24-frame overlap, and delayed-commit ownership remain fixed.
- Routing: later first passes extend bounded latent tails. Every latent reload validates the exact expected temporal
  tokens. Later second passes keep the eight-frame causal preroll and use core `LTXVAddGuide` with the prior raw
  window's 25 provisional visible frames at frame eight, after the sacrificial causal-token preroll. Raw workers
  write clean lossless FFV1/yuv444p
  `window.mkv`; assembly trims/joins windows and performs one H.264 scene encode. No worker, raw window, assembled
  scene, or project final contains a watermark; only the separate Discord graph creates a transient watermarked
  send with local output disabled.
- Recovery/activation: automatic and remake prompt ownership distinguishes `t2i`, `i2v_legacy`,
  `i2v_continuation`, and delivery where applicable. A persisted continuation plan/started legacy attempt locks the
  selected I2V route across restart and configuration changes. Restart recovery reclaims queue/history without
  feeding one graph's prompt into another. An explicit saved-job retry preserves only a genuinely RUNNING owned
  prompt, grants exhausted T2I/legacy I2V a fresh bounded retry epoch, and grants failed continuation chunks their
  separate invalidated-attempt epoch. A missing cached-frame file resets both T2I and downstream I2V budgets.
  Stage two atomically checkpoints both video and audio latents; when they verify but `window.mkv` is missing,
  recovery uses a decode/mux-only graph instead of repeating diffusion. Generic intermediate `i2v` ownership
  migrates by exact chunk-attempt prompt ID, never by plan existence alone.
  Explicit I2V rerun invalidates accepted continuation chunks from zero plus revision assembly ownership; it must
  not silently reuse the prior chain. GUI remake recovery binds a post-I2V/pre-delivery checkpoint to the exact
  job/scene/revision, route, parameter hash, frame hash, raw-video hash/path, and probed production profile; only
  that checkpoint may skip a repeated diffusion run. GUI startup checks Save Scene Frame plus both chunk-latent
  contracts and performs the path-verified restart only with an empty queue.
- Rollout: `explicit` remains the beta/manual-opt-in default. `auto` fails closed without
  `<TENMIN_STORAGE_ROOT>\state\continuation-validation-v1.json` (default
  `D:\LTX_Supervisor_Storage\state\continuation-validation-v1.json`) bound to the current generation/routing/recovery
  implementation and every representative live-graph node contract, containing required external-asset hash
  provenance, four completed bounded generations, peak VRAM, and all accepted decisions.
  The broad rollout implementation identity covers continuation generation, routing, state/recovery, remakes, and
  entrypoints. Accepted chunk cache identity separately hashes only generation-affecting project code plus
  normalized structural node contracts; dynamic model/LoRA combo membership and GUI/setup changes cannot discard a
  valid chunk. Exact graph hashes remain part of every attempt. The live rollout identity covers every node class
  in the representative initial/later/final, checkpoint-only decode, and delivery graphs.
  `validate_against_object_info` must check literal scalar types, finite floats, and numeric min/max bounds as well
  as required inputs, combo values, routes, output slots, and routed types.
- Reproduction:
  1. Parse `examples\example_job.json`; build a 30-second `SceneFramePlan` and verify the eight contributions and
     exact 721/720 generation/presentation counts above.
  2. Build initial, later, and final stage-one/stage-two graphs; verify later stage two loads 25 prior frames into
     `LTXVAddGuide`, both latent loads carry the planned token count, and the raw combine selects
     `video/ffv1-mkv` with `yuv444p`.
  3. With ComfyUI healthy, run `python scripts\validate_continuation_workflows.py`; it must inspect live
     `/object_info` and must not queue a prompt. A stale live node contract must trigger the GUI's empty-queue-only
     restart guard or fail startup without touching active work.
  4. Start in `auto` without the D-drive approval file and verify supervisor construction fails closed. Do not
     manufacture approval until the bounded GPU matrix and human comparison are complete.
- Verification commands:
  - `python -c "import json; from pathlib import Path; from tenminvideomaker.contracts import parse_job_payload; parse_job_payload(json.loads(Path(r'examples/example_job.json').read_text(encoding='utf-8')))"`
  - `python -m unittest discover -s tests -p "test_continuation*.py" -v`
  - `python -m unittest discover -s tests -p "test_chunk*.py" -v`
  - `python -m unittest discover -s tests -p "test_run_supervisor.py" -v`
  - `python -m unittest discover -s tests -p "test_validate_continuation_workflows.py" -v`
  - `python -m unittest discover -s tests -p "test_gui_app.py" -v`
  - `python -m unittest discover -s tests -p "test_supervisor.py" -v`
  - `python -m unittest discover -s tests -p "test_workflow_builder.py" -v`
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `python scripts\validate_continuation_workflows.py`
  - `git diff --check`

### 2026-07-31 — continuation graph cache-isolation repair

- Changed files: `tenminvideomaker/workflow_builder.py`,
  `tenminvideomaker/continuation_workflow.py`,
  `tests/test_continuation_workflow.py`, `docs/architecture.md`,
  `docs/research/ltx23_chunked_continuation_plan.md`, and this file.
- Evidence: `gpu12` completed base, single-frame, and decoded-guide cases, then
  its 97-frame later `LTXVExtendSampler` failed at 5,040-versus-4,788 tokens.
  The exact failed graph, with the same client ID and only every node ID shifted
  by 1,000, completed at
  `D:\LTX_Supervisor_Storage\jobs\continuation-acceptance-20260731-gpu12-probe-unique-ids`.
  The clean-client 97-frame probe had also completed; only shared numeric IDs
  reproduced the mismatch.
- Rule: continuation graph IDs must be unique across job, scene, revision,
  chunk, attempt, and phase. Do not reuse `1..N` for all continuation prompts:
  locally cached mutable LTX conditioning can otherwise leak between stages.
- Reproduction: with an empty Comfy queue, clone the failed history prompt,
  change only all node IDs and internal links to a fresh numeric range, keep the
  same client ID, and queue the isolated raw stage-one saver. It must succeed
  before changing generation behavior.
- Verification commands:
  - `python -m unittest discover -s tests -p "test_continuation_workflow.py" -v`
  - `python -m unittest discover -s tests`
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `python scripts\validate_continuation_workflows.py`
  - `git diff --check`

### 2026-07-31 — mutable conditioning cache barrier

- Changed files: `tenminvideomaker/nodes.py`,
  `tenminvideomaker/continuation_workflow.py`,
  `tenminvideomaker/continuation_renderer.py`, `scripts/run_gui.py`,
  `tests/test_nodes.py`, `tests/test_continuation_workflow.py`,
  `tests/test_gui_app.py`, `docs/architecture.md`, and this file.
- Root cause: acceptance prompt `d135b4e5-005f-4525-82b3-42dabdd8130c` reused the static checkpoint,
  text-encoding, and `LTXVConditioning` chain, then `LTXVExtendSampler` failed with a 5,040-versus-4,788 token
  shape mismatch. Its exact graph, with only all node IDs shifted, completed as
  `cc15cee3-5dc9-465c-adf2-33827eb5e334` with no cached nodes. The issue is mutable LTX conditioning reuse, not
  the official 24-frame overlap: live `/object_info/LTXVExtendSampler` requires an overlap of at least 16 frames in
  steps of 8.
- Decision: preserve the documented 24-frame overlap and 97-frame full extension. Before `LTXVConditioning`, clone
  each positive/negative `CONDITIONING` value through `10MinVideoMaker_IsolateConditioning`; that node always
  reexecutes, so the downstream LTX conditioning chain receives fresh tensors and metadata on every prompt.
- Follow-up evidence: the conditioning barrier and `LTXVConditioning` both reexecuted in GPU15, but prompt
  `4cfb6821-2747-4342-9c5f-4e42e909dc46` still failed while its static checkpoint, LoRA, and
  `LTXVChunkFeedForward` chain was cached. Its all-fresh-ID counterpart
  `fa1055ab-aa47-4037-a3b5-c51400e18988` completed. Treat `MODEL` wrappers as mutable too: clone them immediately
  before and after chunk feed-forward with always-reexecuted `10MinVideoMaker_IsolateModel` nodes. `ModelPatcher`
  cloning creates wrapper copies, not model-weight copies.
- Reproduction: with an empty ComfyUI queue, reuse a failed later-stage graph. The unmodified graph may cache its
  static conditioning chain and fail. Renumbering every node must succeed. The production graph must instead expose
  two `10MinVideoMaker_IsolateConditioning` nodes feeding `LTXVConditioning`, two
  `10MinVideoMaker_IsolateModel` nodes bracketing `LTXVChunkFeedForward`, and retain `frame_overlap=24`.
- Verification commands:
  - `python -m unittest discover -s tests -p "test_nodes.py" -v`
  - `python -m unittest discover -s tests -p "test_continuation_workflow.py" -v`
  - `python -m unittest discover -s tests`
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `python scripts\validate_continuation_workflows.py`
  - `git diff --check`

### 2026-07-31 — `LTXVExtendSampler` endpoint-frame repair

- Changed files: `tenminvideomaker/continuation_workflow.py`,
  `tests/test_continuation_workflow.py`, `README.md`, `docs/architecture.md`,
  `docs/research/ltx23_chunked_continuation_plan.md`, `docs/user-guide.md`, and
  this file.
- Reproduction: GPU acceptance `continuation-acceptance-20260731-gpu11` completed
  base, single-frame, and decoded-guide cases, then failed the first later
  `LTXVExtendSampler` at 96 with a 4,788-versus-4,536 token shape mismatch.
  The isolated same-graph probe changed only `num_new_frames` to 97 and
  completed, saving a 17-token bounded handoff for chunk 1 at
  `D:\LTX_Supervisor_Storage\jobs\continuation-acceptance-20260731-gpu11-probe97`.
- Rule: plan fields count transitions; the sampler expects pixel-frame count.
  Pass `chunk.new_transition_frames + 1` so `frame_overlap + num_new_frames`
  remains `8n+1`. Full 96-transition extensions therefore pass 97; short final
  extensions follow the same rule. Do not regress to 96.
- Verification commands:
  - `python -m unittest discover -s tests -p "test_continuation_workflow.py" -v`
  - `python -m unittest discover -s tests`
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `python scripts\validate_continuation_workflows.py`
  - `git diff --check`
- Acceptance boundary: the unit/no-render checks do not validate image quality. No bounded GPU generation matrix,
  visual seam comparison, anatomy comparison, peak-VRAM acceptance, or runtime acceptance has been completed.
  Therefore continuation has no production-quality claim and `auto` must remain locked.

### 2026-07-31 — bounded continuation acceptance runner

- Changed files: `tenminvideomaker/continuation_acceptance.py`,
  `tenminvideomaker/continuation_workflow.py`, `tenminvideomaker/comfy_http.py`,
  `scripts/run_continuation_acceptance.py`, `scripts/validate_continuation_workflows.py`, focused continuation,
  HTTP-client, and runner tests, `README.md`, `docs/architecture.md`, `docs/user-guide.md`, and this file.
- Architecture: a unique `continuation-acceptance-*` job namespace is created only in
  `D:\LTX_Supervisor_Storage\acceptance` and project-owned job paths. The runner reads the selected saved payload
  through SQLite read-only mode and reuses only its exact D-drive cached T2I frame. It never starts the supervisor,
  edits `pipeline.sqlite3`, or changes production rollout mode.
- Routing: all cases share one production two-pass 121-frame base. `single_frame` consumes its exact final decoded
  frame; `decoded_17_frame` uses a diagnostic initial refinement branch which loads frames 96–112 through
  `LTXVAddGuide` at frame zero; `latent_overlap` uses the normal 24-frame `LTXVExtendSampler` route plus later
  full-resolution guide at causal-preroll frame eight. The diagnostic initial guide is opt-in and cannot alter
  standard production initial chunks.
- Evidence: every stage records prompt ID, runtime, sampled peak VRAM, SHA-256, lossless raw video, ffprobe data,
  RGB/luma/chroma boundary differences, optional Farneback flow discontinuity, and explicit pending human anatomy,
  contact, identity, camera, and motion-restart decisions. Completion always remains `awaiting_human_review`; it
  cannot manufacture `continuation-validation-v1.json` or unlock `auto`.
- Reproduction: with an empty ComfyUI queue, run
  `python scripts\run_continuation_acceptance.py --source-job-id <saved-id> --source-scene-id <scene-id> --dry-run`
  first. Remove `--dry-run` only for deliberate GPU acceptance work. Inspect
  `D:\LTX_Supervisor_Storage\acceptance\<run-id>\run.json`; review raw windows without exposing or using the
  supervisor's active state.
- Verification commands:
  - `python -m unittest discover -s tests -p "test_continuation*.py" -v`
  - `python -m unittest discover -s tests -p "test_comfy_http.py" -v`
  - `python -m unittest discover -s tests -p "test_run_continuation_acceptance.py" -v`
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `python scripts\validate_continuation_workflows.py`
  - `python scripts\run_continuation_acceptance.py --source-job-id <saved-id> --source-scene-id <scene-id> --dry-run`
  - `git diff --check`

### 2026-07-31 — LTX broadcast noise-mask checkpoint repair

- Changed files: `tenminvideomaker/chunk_artifacts.py`,
  `tests/test_chunk_artifacts.py`, and this file.
- Root cause: the first live acceptance prompt
  `4858eb2f-e3a5-4d0b-b5f3-3f6c5a80cda5` reached
  `10MinVideoMaker_SaveChunkLatent` with an LTX I2V video latent containing
  `noise_mask` shaped `[1, 1, 16, 1, 1]`. The checkpoint guard incorrectly
  required full latent spatial dimensions, although the one-by-one spatial axes
  are valid PyTorch broadcast axes and must remain compact when persisted.
- Decision: video checkpoint masks still require matching batch, a valid
  one-or-channel axis, and exact temporal-token count. Each spatial axis may be
  either `1` or the matching video-latent dimension. The node preserves the
  original mask shape; it must never expand a compact mask before serializing.
  Audio validation remains exact-shape-only. Second-pass split video also
  carries the opaque LTX `type: "audio"` marker. A direct LTX tiled-video-decode
  probe confirmed output slot zero is still the video latent despite that marker,
  matching the live node's declared output order. The checkpoint allowlist
  therefore preserves only the exact `audio` or `video` type markers without
  inferring artifact kind from them.
- Reproduction: with an empty queue, run the continuation acceptance runner
  against a cached source scene. Before this repair, its common base stage-one
  checkpoint failed with `LATENT noise_mask shape does not match video samples.`
  The focused regression uses the same `[1, 1, frames, 1, 1]` shape and
  verifies save/load round-trip integrity. The related type-marker regression
  verifies the marker is preserved through the manifest and checkpoint reload.
- Verification commands:
  - `python -m unittest discover -s tests -p "test_chunk_artifacts.py" -v`
  - `python -m compileall -q tenminvideomaker tests`
  - `python scripts\validate_continuation_workflows.py`
  - `git diff --check`

### 2026-07-31 — decoded-guide diagnostic temporal-slot repair

- Changed files: `tenminvideomaker/continuation_workflow.py`,
  `tests/test_continuation_workflow.py`, and this file.
- Evidence: the first corrected GPU matrix completed common base and
  single-frame cases, then `decoded_25_frame` failed inside `SamplerCustom`
  with a 21,168-by-128 guide tensor targeting 20,160-by-128 latent positions.
  Directly routing the guide into the upscaled latent removed its competing
  single-frame route, but a second GPU run retained exact 21-versus-20 token
  mismatch. A direct 17-frame guide probe completed on same graph.
- Root cause: initial refined latent exposes only 20 guide-token positions. A
  25-frame valid `8n+1` guide encodes 21 tokens and cannot fit. Largest valid
  `8n+1` guide that fits is 17 frames, encoding 20. A current-job GPU probe
  proved the valid span is frames 96–112; moving it to 104–120 fails with a
  20-versus-19 mismatch. Diagnostic is named `decoded_17_frame`; metrics use
  its exact span. Normal later-window 25-frame visible-overlap route remains
  separate production test.
- Routing repair: every `LTXVAddGuide` must feed `LTXVCropGuides` before
  `LTXVConcatAVLatent` and `SamplerCustom`; wire crop positive, negative, and
  latent outputs as the new sampling inputs. A same-graph GPU probe completed
  only after this crop. Without it, the guide's appended token positions cause
  `SamplerCustom` to fail with a 20-guide-token versus 19-target-slot mismatch.
- Reproduction: run the D-drive acceptance matrix with an empty queue. The
  failed matrix remains at
  `D:\LTX_Supervisor_Storage\acceptance\continuation-acceptance-20260731-gpu5`.
  The regression verifies that a decoded 17-frame guide has no preceding
  `LTXVImgToVideoInplaceKJ` or `LTXReferenceConditioning`, and that its guide
  latent is the direct output of `LTXVLatentUpsamplerTiled`. Acceptance output
  schema is version 3; old incomplete runs retain their original case name and
  original metric semantics.
- Verification commands:
  - `python -m unittest discover -s tests -p "test_continuation_workflow.py" -v`
  - `python -m compileall -q tenminvideomaker tests`
  - `python scripts\validate_continuation_workflows.py`
  - `git diff --check`

### 2026-07-31 — forced-fresh LTX continuation checkpoint

- Changed files: `tenminvideomaker/nodes.py`,
  `tenminvideomaker/continuation_workflow.py`,
  `tenminvideomaker/continuation_renderer.py`, `scripts/run_gui.py`, focused
  node/workflow/GUI tests, `README.md`, `docs/architecture.md`,
  `docs/user-guide.md`, `TODO.md`, and this file.
- Evidence: a fresh-client probe initially completed an unchanged failed
  later-stage graph, but a complete four-case matrix still failed later with
  `5040` versus `4788`; fresh client IDs are not a reliable execution-cache
  boundary. Histories showed `CheckpointLoaderSimple` plus the mandatory LoRA
  chain cached across different client IDs. `10MinVideoMaker_FreshCheckpoint`
  then replaced `CheckpointLoaderSimple` in every continuation stage/decode
  graph. Completed matrix
  `continuation-acceptance-20260731-065935` recorded all eight stage prompts as
  successful. Its fresh-checkpoint node was never `execution_cached`; only
  non-mutable encoder/prompt/selector inputs reused cache. Follow-up base-client
  later-window probe `a4165c9e-f40b-4683-8370-af4b92bf6d10` also completed in
  36.8 seconds with zero cached nodes, confirming the production ownership path
  rather than acceptance-only client scoping.
- Decision: the project node always delegates to ComfyUI's public
  `CheckpointLoaderSimple` surface with a phase scope and `IS_CHANGED=NaN`.
  This constructs fresh MODEL/CLIP/VAE wrappers before dynamic LoRAs and chunk
  feed-forward, without calling `/free` between chunks. Do not rely on unique
  graph IDs or ComfyUI client IDs to invalidate mutable LTX loader state.
- Reproduction: with an empty queue, run the four-case matrix against a cached
  project frame. Each graph must expose exactly one
  `10MinVideoMaker_FreshCheckpoint`; its history must not list that node under
  `execution_cached`. Verify all output streams are 768×1344, 24/1, FFV1, and
  yuv444p mechanically. Human visual review remains mandatory.
- Verification commands:
  - `python -m unittest discover -s tests -p "test_nodes.py" -v`
  - `python -m unittest discover -s tests -p "test_continuation_workflow.py" -v`
  - `python -m unittest discover -s tests -p "test_gui_app.py" -v`
  - `python -m unittest discover -s tests`
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `python scripts\validate_continuation_workflows.py`
  - `python scripts\run_continuation_acceptance.py --source-job-id 20260730-0217 --source-scene-id 1 --dry-run`
  - `git diff --check`

### 2026-07-31 — continuation acceptance review UI

- Changed files: `tenminvideomaker/acceptance_review.py`, `tenminvideomaker/gui_app.py`,
  `web/acceptance-review.html`, `web/acceptance-review.js`, `web/acceptance-review.css`, normal-GUI review link,
  focused review/GUI tests, and `docs/user-guide.md`.
- Decision: acceptance review is a read-only FastAPI surface. It accepts only a completed
  `continuation-acceptance-YYYYMMDD-HHMMSS` run document in `awaiting_human_review`; every raw source and still
  must resolve under project-owned D-drive storage. Browser sees semantic case names, exact boundary labels, and
  API URLs only, never raw filesystem paths or raw JSON.
- Media routing: review proxies are unwatermarked H.264/AAC MP4 files below the run's
  `acceptance/<run-id>/review/` directory. FFmpeg writes a same-directory temporary file, checks it is non-empty,
  then atomically replaces the proxy. Existing non-empty proxies are reused. Raw FFV1/FLAC windows remain untouched.
  Proxy failure returns HTTP 503; invalid/missing artifacts return 404. LAN Basic authentication applies globally.
- Review semantics: `common_base` is reference only. `single_frame` compares base 119/120 with case 0/1;
  `decoded_17_frame` compares base 111/112 with case 16/17; `latent_overlap` compares base 119/120 with case
  24/25. The mobile page stacks media while desktop keeps side-by-side native HTML5 videos.
- Verification commands:
  - `python -m unittest discover -s tests -p "test_acceptance_review.py" -v`
  - embedded Python `test_gui_app.py` suite (FastAPI supplied by ComfyUI environment)
  - `python -m unittest discover -s tests -v`
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `python scripts\validate_continuation_workflows.py`
  - `git diff --check`

### 2026-07-31 — locked-auto continuation review fallback

- Changed files: `scripts/run_gui.py`, `tenminvideomaker/gui_app.py`, `tests/test_gui_app.py`,
  `docs/user-guide.md`, and this file.
- Root cause: `TENMIN_LTX_CONTINUATION_MODE=auto` correctly raises `ContinuationRolloutError` before the normal
  launcher creates its FastAPI application. This made the evidence-review route unreachable at the precise point
  it was needed to complete human acceptance.
- Decision: retain the fail-closed supervisor gate, but serve a separate review-only FastAPI application on the
  normal GUI port. It has only acceptance-review routes and static assets; it does not construct a controller,
  start Gmail polling, start ComfyUI, submit prompts, modify job state, or approve rollout. LAN Basic authentication
  applies unchanged. The regular GUI still starts only after `auto` approval succeeds.
- Reproduction: set continuation mode to `auto` with a missing or stale
  `D:\LTX_Supervisor_Storage\state\continuation-validation-v1.json`, then run the normal launcher. It must serve
  `GET /api/acceptance-runs` with HTTP 200 and redirect `/` to `/acceptance-review.html`, while `/api/status`
  returns HTTP 404.
- Verification commands:
  - embedded Python focused `GuiAppTests.test_review_only_gui_serves_acceptance_page_without_supervisor`
  - embedded Python focused `GuiAppTests.test_launcher_uses_review_only_application_when_auto_rollout_is_locked`
  - embedded Python `test_gui_app.py` suite
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `git diff --check`

### 2026-07-31 — human continuation decision and exact assembled review

- Changed files: `tenminvideomaker/acceptance_review.py`,
  `tenminvideomaker/gui_app.py`, `web/acceptance-review.html`,
  `web/acceptance-review.js`, `web/acceptance-review.css`, focused tests,
  `experiments/ltx23-style-conversion/`, `docs/research/continuation_acceptance_20260731_human_review.md`,
  `docs/user-guide.md`, and this file. Project-owned evidence was added under
  `D:\LTX_Supervisor_Storage\acceptance\continuation-acceptance-20260731-065935\human-review*`.
- Human decision: approve no production method. `single_frame` is only the next
  baseline and remains rejected for blur/detail loss. `decoded_17_frame` is
  rejected for same-style continuation but preserved as a high-value
  anime-to-live-action research lead. `latent_overlap` is rejected and
  preserved as a weaker anime-to-semi-realistic-3D lead. Automatic rollout
  remains locked; do not infer approval from the separate `human-review.json`.
- Preservation: exact decoded-guide and latent-overlap stage-one/stage-two API
  workflows are frozen and SHA-256-bound under
  `experiments/ltx23-style-conversion/`. Large raw videos remain only on D:.
  New research must copy the frozen evidence instead of editing it.
- Assembly routing: the review service creates one unwatermarked, video-only
  H.264 proxy per case. Exact cuts are single-frame base 0–120 plus case 1+;
  decoded-guide base 0–112 plus case 17+; latent-overlap base 0–120 plus case
  25+. FFmpeg writes a same-directory partial file, validates non-empty output,
  atomically replaces the cache, and never modifies raw windows.
- Reproduction: open `/acceptance-review.html?run=continuation-acceptance-20260731-065935`,
  select each method, and play **True assembled result** below the existing
  side-by-side view. The review-only application must not expose supervisor
  routes, submit prompts, or create `continuation-validation-v1.json`.
- Verification commands:
  - embedded Python `test_experiment_preservation.py`
  - embedded Python `test_acceptance_review.py`
  - embedded Python `test_gui_app.py`
  - `python -m unittest discover -s tests -v`
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `git diff --check`

### 2026-07-31 — duplicate GUI lock recovery

- Changed files: `tenminvideomaker/ownership.py`, `scripts/run_gui.py`,
  `tests/test_ownership.py`, focused GUI tests, `docs/user-guide.md`, and this
  file.
- Root cause: Windows denies reads of byte zero while another process owns the
  `msvcrt` byte lock. `SupervisorInstanceLock.acquire()` read that byte before
  its protected nonblocking-lock call, so a legitimate duplicate launch leaked
  raw `PermissionError` instead of the intended `OwnershipError`.
- Decision: determine sentinel emptiness by seeking to end and reading the file
  position; never read the locked byte. `msvcrt.locking(..., LK_NBLCK, 1)`
  remains the sole ownership authority. Do not delete, steal, or bypass
  `supervisor.lock`.
- GUI behavior: only an initial instance-lock collision is treated as an
  already-running GUI. The duplicate process logs and optionally opens
  `http://127.0.0.1:<port>/`, then returns zero. Later ownership failures for a
  busy queue, legacy takeover, or stale node contracts remain fatal.
- Reproduction: keep one GUI process running and start `scripts/run_gui.py`
  again. Before the repair, Windows raises `PermissionError` from
  `handle.read(1)`; afterward the second launch prints the existing URL without
  constructing an app or supervisor.
- Verification commands:
  - embedded Python `test_ownership.py`
  - embedded Python duplicate-launch tests in `test_gui_app.py`
  - embedded Python full `test_gui_app.py`
  - `python -m unittest discover -s tests -v`
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `git diff --check`

### 2026-07-31 — production style-stable continuation rollout

- Changed files: `scripts/run_continuation_acceptance.py`,
  `scripts/prepare_safe_continuation_source.py`,
  `examples/safe_continuation_source.json`,
  `tenminvideomaker/acceptance_review.py`,
  `tenminvideomaker/chunk_assembly.py`, `tenminvideomaker/constants.py`,
  `tenminvideomaker/continuation_acceptance.py`,
  `tenminvideomaker/continuation_renderer.py`,
  `tenminvideomaker/continuation_validation.py`,
  `tenminvideomaker/continuation_workflow.py`, their focused tests,
  `README.md`, `docs/architecture.md`, `docs/user-guide.md`, and this file.
- Root cause: every tested stage-two diffusion-video route repainted the
  subject. Lower sigma and persistent-reference variants still changed style
  or identity, while direct stage-one handoff decode preserved both the look
  and continuous motion. The rejected diagnostic workflows remain frozen in
  `experiments/ltx23-style-conversion`; do not erase that research evidence.
- Production route: keep the official bounded `LTXVExtendSampler` chain as the
  visual source of truth. The stage-two LCM/spatial graph still samples AV so
  generated audio stays synchronized, but only its audio latent is retained.
  Save the stage-one handoff as the durable video checkpoint, tiled-decode at
  384×672, and apply `RealESRGAN_x2.pth` before lossless mux. Recovery performs
  the same decode/upscale/mux route.
- Timeline ownership: full later direct decodes have eight sacrificial video
  frames. Non-final video slices are initial `0:104` then later `8:104`; the
  final later slice is `8:model_window_frames`. Sampled audio is eight frames
  earlier, so later audio starts at 16 and retains the video slice's frame
  count. The prior raw overlap always begins at frame 96 for the stage-two
  guide, including after a later raw window.
- Acceptance evidence: the safe, fully clothed run
  `D:\LTX_Supervisor_Storage\acceptance\continuation-acceptance-safe-production-20260731`
  completed all four bounded methods at 768×1344/24 fps without OOM. Direct
  visual review of base 102/103 and continuation 8/9 approved identity, style,
  wardrobe/prop/environment continuity, anatomy, and forward motion. The
  production seam RGB MAE was 12.332. Latent-overlap stage-two peak VRAM was
  15,860,302,852 bytes. The exact two-window assembly validated as 216 frames,
  H.264/yuv420p, stereo 48 kHz AAC.
- Rollout: approval schema is version 2 at
  `D:\LTX_Supervisor_Storage\state\continuation-validation-v2.json`. It binds
  the complete continuation implementation and live structural node contracts,
  includes `RealESRGAN_x2.pth` in external-asset provenance, and requires the
  no-OOM, LCM-guider, production-seam motion, style/identity, anatomy, A/V
  profile, and runtime decisions.
- Reproduction and verification commands:
  - `python scripts\prepare_safe_continuation_source.py --help`
  - `python scripts\run_continuation_acceptance.py --source-payload-file <safe.json> --source-frame <frame.png> --source-scene-id 1 --dry-run`
  - `python -m unittest discover -s tests -v`
  - `python -m compileall -q tenminvideomaker scripts tests`
  - `python scripts\validate_continuation_workflows.py --comfy-url http://127.0.0.1:8188`
  - `git diff --check`
