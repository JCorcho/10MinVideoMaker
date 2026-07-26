# 10MinVideoMaker

An independent ComfyUI custom-node project for building a guided long-form video workflow.

## Current status

The durable job contract, SQLite state machine, Gmail/Google Drive transport, LoRA resolver, FFmpeg assembly
service, interactive ComfyUI nodes, scene-specific Anima/Pony/LTX workflow builders, and supervisor are
implemented. The supervisor normally runs behind a loopback FastAPI browser UI at
`http://127.0.0.1:8765/`. Optional password-protected private-LAN access supports phone review without exposing
ComfyUI. New Grok jobs start automatically for unattended operation, while historical complete,
partial, failed, and cancelled jobs can be reviewed without displaying raw JSON.

All new persistent runtime data lives under `D:\LTX_Supervisor_Storage`: settings and encrypted secrets, SQLite
state, source payloads, versioned frames and clips, generation manifests, finals, logs, and temporary assembly
files. The first GUI launch performs a non-destructive import of the former project runtime and recorded
`D:\output\10minfinals` artifacts. Legacy source files are left untouched.

Legacy first completed master: `D:\output\10minfinals\20260724-2249_final.mp4` at 704×1248, 24 fps, stereo AAC,
3,848 frames, and 160.35 seconds. New jobs use 768×1344.

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

To avoid model-reload thrash on the 16 GB GPU, an automatic job has two sequential model-residency phases: it creates
every required T2I frame first, unloads the image model once, then creates every required LTX clip. A multi-scene
remake batch follows the same policy—its image+video edits create frames first (grouped by Anima/Pony family when
mixed), then every successful remake, including video-only edits, runs through one LTX video pass.

## One-click start

Double-click `Start 10MinVideoMaker.bat` in the repository root. On first run it:

1. Detects missing Gmail settings.
2. Offers Google App Password or OAuth2 browser authorization. OAuth includes read-only Google Drive access for
   private job-file links.
3. Opens Civitai Account Settings and securely collects an API token when missing.
4. Saves non-secrets and current-user DPAPI-encrypted secrets under
   `D:\LTX_Supervisor_Storage\config`.
5. Securely collects the Discord Patreon-delivery webhook when missing.
6. Shows the optional settings editor when requested, including a Discord webhook replacement action.
7. Validates Gmail and configured Drive access without sending a message. If the authorized local ComfyUI API is
   down, it starts the unchanged Easy Install `Start ComfyUI.bat` launcher (which retains Sage Attention), waits for
   HTTP health, then launches the edit-and-review GUI plus its single supervisor worker.

On later runs, valid required settings are reused and the launcher asks whether to change optional settings before
starting. See `docs/user-guide.md` for OAuth setup details and safe setup-only commands.

The browser UI shows pipeline/ComfyUI status, a job library, scene previews, every generation parameter used by the
workflow, and version history. Mark any number of scenes across jobs, choose **Video Only** or
**Image + Video**, edit parameters, then use **Save & Remake**. Image-only remakes are deliberately impossible.
If an automated render is active, the UI asks whether to queue edits afterward or cancel only this project's
current prompts and run the edits immediately. Project cards use the readable
`Character · MM/DD/YYYY` label, and the project and scene columns scroll independently.
It is mobile responsive and its T2I/I2V LoRA pickers query the running ComfyUI loader contracts, so each picker
shows locally selectable files for its own model route.

For a testing or review-only launch, run `Start 10MinVideoMaker.bat --hold-new-jobs-for-review`. In that session,
new Gmail handoffs enter **Awaiting review** and require **Approve & Queue Job**. The normal launcher always
auto-starts incoming jobs.

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
- **Save Scene Frame** — atomically caches a deterministic 768×1344 PNG for the matching scene.
- **Stitch Clips** — verifies every clip is 768×1344 at 24 fps before FFmpeg concat.

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
