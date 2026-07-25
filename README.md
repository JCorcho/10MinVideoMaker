# 10MinVideoMaker

An independent ComfyUI custom-node project for building a guided long-form video workflow.

## Current status

The durable job contract, SQLite state machine, Gmail/Google Drive transport, LoRA resolver, FFmpeg assembly service, eight
interactive ComfyUI nodes, scene-specific Anima/Pony/LTX workflow builders, and unattended supervisor are
implemented. A project-local one-click launcher now configures Gmail securely, validates SMTP/IMAP and OAuth Drive
access, starts ComfyUI
when needed, configures Civitai downloads, and launches the supervisor. The first production job rendered all eight
scenes successfully; its geometry recovery and final assembly are documented below.

The first completed master is `D:\output\10minfinals\20260724-2249_final.mp4`: 704×1248, 24 fps, stereo AAC,
3,848 frames, and 160.35 seconds.

T2I routing is model-specific: Anima uses its 30-step `er_sde`/`beta57` reference path without a detailer. Pony
uses 30-step `res_3m_ode` followed by 30-step `res_5s_ode`, then the reference YOLO face bbox detector and
`FaceDetailer` before caching the scene frame.

LoRA routing is stage-safe. Anima/Pony character and scene LoRAs are T2I-only and can never enter the LTX model
chain. Dynamic I2V LoRAs must be verified from Civitai metadata as part of the LTX 2.x family (`LTXV2`,
`LTXV 2.0`–`LTXV 2.3`, and equivalent labels) before workflow construction; an unverifiable, LTX 1.x, or
image-model LoRA is rejected even when its file is already installed. DMD 1.0 and JoyAI 0.5 remain the two
mandatory local LTX LoRAs.

Every newly generated scene also has a metadata-free Patreon delivery branch. The exact approved `wm.png`
watermark is applied only to Discord media: images are sent as lossless PNG at quality 100, and videos are sent as
H.264 with generated audio at quality 65 and 24 fps. Both DiscordSendSave nodes have `save_output`, prompt inclusion,
workflow JSON, CDN logging, and GitHub updates disabled. Clean cached frames and deterministic scene clips remain
unwatermarked for I2V reuse and master assembly.

## One-click start

Double-click `Start 10MinVideoMaker.bat` in the repository root. On first run it:

1. Detects missing Gmail settings.
2. Offers Google App Password or OAuth2 browser authorization. OAuth includes read-only Google Drive access for
   private job-file links.
3. Opens Civitai Account Settings and securely collects an API token when missing.
4. Saves non-secrets in the ignored `.env` file and secrets encrypted with Windows DPAPI in ignored `runtime/`.
5. Securely collects the Discord Patreon-delivery webhook when missing.
6. Shows the optional settings editor when requested, including a Discord webhook replacement action.
7. Validates Gmail and configured Drive access without sending a message, performs a ComfyUI health check, and
   offers to resume or abandon any active saved job before starting the supervisor. Declining cancels only this
   project's queued/running ComfyUI prompts, marks unfinished scenes cancelled, preserves the saved audit history,
   and releases the pipeline to accept a new email.

On later runs, valid required settings are reused and the launcher asks whether to change optional settings before
starting. See `docs/user-guide.md` for OAuth setup details and safe setup-only commands.

The visible supervisor console prints a redacted `STATUS` heartbeat every 15 seconds by default. It reports the
durable state, active job/scene, and only ComfyUI's running/pending queue counts. It also announces Gmail checks,
LoRA resolution, cached-artifact reuse, T2I/I2V attempts, and stitching. Change the interval through the optional
**Console heartbeat seconds** setting (`TENMIN_STATUS_INTERVAL_SECONDS`); use the existing log-level option for
additional `DEBUG` messages.

Run `powershell -ExecutionPolicy Bypass -File scripts\install_windows_shortcuts.ps1` to install the project icon
and launcher shortcut on the current user's Desktop and Start Menu. The shortcut can then be pinned through
Windows' context menu.

## Available nodes

- **Validate Job** — validates and normalizes the exact Grok JSON contract.
- **Pipeline Status** — reads the durable pipeline state.
- **Request Grok Job** — sends `Run the LTX video pipeline` and instructs Grok to return a new message rather than
  replying to it.
- **Poll Gmail Once** — searches unread mail for the exact completion subject `LTX_JOB_COMPLETE`, then performs
  attachment-first extraction and accepts body JSON or a Google Drive file link; scheduling belongs to the
  supervisor. Candidate bodies are fetched with IMAP `BODY.PEEK[]`, so inspecting several messages does not mark
  unclaimed jobs read. Private-file sign-in redirects fall back to the authenticated Drive API. A Grok-style
  leading-zero integer is normalized only for `seed` and `original_seed` fields before strict contract validation.
- **Resolve LoRAs** — uses the live ComfyUI process's active LoRA roots, resolves dynamic assets, verifies mandatory
  I2V LoRAs, accepts verified LTX 2.x dynamic LoRAs for LTX 2.3, and blocks LTX 1.x and image-model LoRAs from I2V.
- **Release Memory** — runs Python and CUDA cache cleanup.
- **Save Scene Frame** — atomically caches a deterministic 704×1248 PNG for the matching scene.
- **Stitch Clips** — verifies every clip is 704×1248 at 24 fps before FFmpeg concat.

Nodes that access Gmail, download assets, or stitch video have side effects and should only be queued deliberately.

## Installation

The repository is intended to live at:

`ComfyUI/custom_nodes/10MinVideoMaker`

Restart ComfyUI after adding or changing a Python node module. Once nodes are introduced, ComfyUI will discover the
pack through `NODE_CLASS_MAPPINGS` in `__init__.py`.

This installation currently discovers this package through the legacy V1 mappings. The node implementations remain
thin wrappers over framework-independent services so the automation supervisor and ComfyUI surface share behavior.

## Repository layout

- `__init__.py` — ComfyUI custom-node package entry point.
- `workflows/` — versioned ComfyUI workflow JSON files.
- `examples/` — a safe example of the exact incoming JSON contract.
- `tests/` — focused regression tests for node and routing behavior.
- `Start 10MinVideoMaker.bat` — interactive setup, validation, ComfyUI health check, and supervisor launch.
- `AI_DEVELOPMENT_RULES.md` — persistent implementation, validation, and documentation rules.

## Development baseline

Before implementing a node or workflow, inspect its live contract through the local ComfyUI API. Validate workflow
routing without rendering before running any expensive generation. See `AI_DEVELOPMENT_RULES.md` for the detailed
project rules and verification commands.

See `docs/user-guide.md` for workflow locations, Gmail environment variables, and the current no-render boundary.
