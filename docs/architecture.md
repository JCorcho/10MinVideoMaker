# 10MinVideoMaker architecture

The pipeline has three layers that call the same pure-Python services:

- **GUI**: a FastAPI/vanilla-JavaScript frontend at `127.0.0.1:8765` by default. Optional private-LAN binding uses
  HTTP Basic credentials stored with Windows DPAPI, while loopback access remains credential-free. It maps stored jobs to
  human-readable controls, previews versioned media, accepts approvals and revision batches, and never exposes raw
  filesystem paths or raw JSON. At the 760px breakpoint, a frontend-only state machine presents the project list,
  selected-project scene list, and selected-scene editor as mutually exclusive drill-down views; desktop retains its
  simultaneous three-panel layout.
- **Single-owner supervisor controller**: one worker owns Gmail polling, automatic jobs, remake batches, ComfyUI API
  job submission, retries, assembly, and controlled restart requests. A cross-process lock prevents the legacy
  console supervisor and GUI worker from running together.
- **ComfyUI nodes**: interactive/status controls built on the same state, payload, mail, asset, and assembly services. They expose no independent routing or persistence rules.

The running ComfyUI 0.27.1 installation did not expose the package's initial V3 entrypoint through `/object_info`.
The package therefore registers its compatibility surface through `NODE_CLASS_MAPPINGS`. Business logic remains
outside the node classes, so this registration choice does not fork the automation implementation.

`D:\LTX_Supervisor_Storage\state\pipeline.sqlite3` is the durable state store. It records the singleton pipeline,
job history, scene state, immutable source payloads, editable scene revisions, cross-job remake batches, and each
revision's media paths. New Gmail jobs auto-queue and render in the normal GUI session. A GUI launched with
`--hold-new-jobs-for-review` instead claims them as `awaiting_review` until explicitly approved. Completed artifacts
remain intact when unfinished scenes are re-queued.

Automatic jobs are stage-batched for model residency: asset preparation, every necessary T2I frame, one model
release, every necessary LTX I2V clip, then final assembly. The scene state and exact frame path remain durable after
each individual prompt, so a restart can resume unfinished work without rebuilding completed frames. Remake batches
preflight all selected revisions, render image+video frames first (grouped by T2I family), and only then render the
entire eligible set of LTX clips. The ComfyUI free-memory endpoint is called at phase boundaries, never between
scenes in the same model family.

## Chunked LTX continuation

The optional long-scene route is identified by feature flag `ltx_chunked_continuation_v1` and resolved strategy
`ltx23_latent_overlap_v1`. The rollout setting `TENMIN_LTX_CONTINUATION_MODE` accepts `disabled`,
`explicit`, or `auto`; `explicit` is the current default. In explicit mode, only a scene whose generation master is
longer than 121 frames and whose `i2v.continuation.enabled` value is true uses continuation. This remains the
beta/manual opt-in route. Scenes at or below the threshold stay on the legacy path, and a previously completed
legacy revision is never converted in place.

`auto` does not merely change the routing predicate. Supervisor construction fails closed unless
`<TENMIN_STORAGE_ROOT>\state\continuation-validation-v1.json` (default
`D:\LTX_Supervisor_Storage\state\continuation-validation-v1.json`) has schema version 1, the current strategy,
`approved` status, reviewer/timestamp, a hash covering the current continuation generation/routing/recovery
implementation, and hashes covering every node contract used by the representative live continuation graphs. It
must also record SHA-256/source/license evidence for the checkpoint, text encoder, spatial upscaler, DMD, and JoyAI
assets; completed `common_base`, `single_frame`, `decoded_17_frame`, and `latent_overlap` generations (including
positive peak VRAM for latent overlap); and all six safety/quality/runtime decisions as accepted. Missing, stale,
malformed, or incomplete evidence blocks `auto` before work starts. The project does not read or modify shared
model files to manufacture this evidence.

Temporal accounting distinguishes the exact presentation timeline from the LTX generation master. The requested
timeline is `round(seconds × 24)` frames. The planner rounds its transition count up to a multiple of eight and adds
one frame to form the `8n + 1` generation master. The initial model invocation contributes at most 120 transitions.
Every full continuation contributes 96 new transitions while regenerating a 24-frame overlap. The exact reference
cases are:

| Requested duration | Timeline frames | Generation master | Transition contributions |
| --- | ---: | ---: | --- |
| 5 seconds | 120 | 121 | `120` |
| 10 seconds | 240 | 241 | `120, 96, 24` |
| 20 seconds | 480 | 481 | `120, 96, 96, 96, 72` |
| 30 seconds | 720 | 721 | `120, 96, 96, 96, 96, 96, 96, 24` |
| 32 seconds | 768 | 769 | `120, 96, 96, 96, 96, 96, 96, 72` |

Each chunk attempt owns both passes. Chunk zero performs the normal cached-frame-conditioned 384×672 video-only
first pass, then atomically checkpoints a bounded plain-LTX handoff. Later handoffs contain only the bounded latent
tail needed for the current 121-frame-or-shorter model window, including the causal predecessor token; the complete
scene latent is never retained. Both stage-one and stage-two checkpoint reuse supplies the planned
`expected_temporal_tokens` to `10MinVideoMaker_LoadChunkLatent`. The node validates that count with the checkpoint
identity, manifest SHA-256, tensor descriptors, and LTX shape, and its cache fingerprint includes the hash and
expected token count. Later first passes use the official `LTXVExtendSampler` with the fixed 24-frame overlap, up to
96 new transitions plus their endpoint frame (`num_new_frames=97` for a full window), the existing first-pass LCM
sigmas, and a deterministic unsigned-64-bit derived seed. Chunk
prompts prepend stable identity, wardrobe, environment, camera-axis, and screen-direction anchors when supplied.
Explicit beat segments are mapped to the model windows they overlap; old payloads reuse the scene prompt with a
deterministic “continue seamlessly” instruction without rewriting the source JSON.

Continuation worker node IDs are deterministically scoped by job, scene, revision, chunk, attempt, and phase.
The graph also places an always-reexecuted `10MinVideoMaker_IsolateConditioning` node between every
`CLIPTextEncode` and `LTXVConditioning` node. It clones prompt tensors and metadata before LTX guide state can be
attached. This is required because ComfyUI may reuse static prompt-node outputs after a failed prompt, even when a
later continuation graph has a separately scoped node-ID range. The barrier changes no prompt, sampler, sigma, or
model setting; it only supplies a fresh conditioning object for each prompt.

The full-resolution second pass uses the existing tiled x2 spatial upscaler and second LCM schedule. The initial
121-frame window selects 16 temporal latent tokens. A later 121-frame visible window selects 17 tokens: a nonzero
LTX temporal token cannot safely become the special first token after slicing, so the extra token acts as an
eight-frame causal preroll. The prior raw window's 25 provisional final-resolution visible frames are loaded after
its own preroll/commit offset and passed to core `LTXVAddGuide` at frame eight with strength 1.0. Frame eight skips
the sacrificial causal-token preroll and aligns the guide with the first visible frame retained by assembly. This is the
second-pass seam guide; it is not a regenerated T2I frame or a single decoded last frame. The raw continuation
window is therefore 129 frames for a full later chunk, and assembly discards its first eight frames. Short final
chunks retain the same accounting with a shorter visible window.

Every `LTXVAddGuide` output runs through `LTXVCropGuides` before `LTXVConcatAVLatent` and `SamplerCustom`.
`LTXVAddGuide` appends guide tokens for conditioning; sampling those appended tokens produces a temporal-grid shape
mismatch on this local LTX build. The crop node returns matched positive/negative conditioning and the sampled latent.

Assembly uses a one-overlap delayed-commit policy. Every non-final refined window contributes the frames before the
following window begins—normally 96—and the final window contributes its entire overlap-plus-new range. This makes
the later fused window own the overlap exactly once. The generation master is then trimmed to the exact presentation
timeline; a 30-second request therefore uses eight model invocations, assembles 721 unique generation frames, and
emits exactly 720 frames.

Every scene revision is stored below
`D:\LTX_Supervisor_Storage\jobs\{job_id}\scenes\scene_{id}\revisions\{revision}` with `frame.png`,
`video.mp4`, and a human-readable `generation-manifest.json`. Source Grok payloads live in each job's `source`
folder, and finals live in `D:\LTX_Supervisor_Storage\finals`. The first GUI launch copies the prior SQLite
database, configured secrets/settings, payloads, and only media recorded by this project into the new layout.
Migration never deletes or rewrites the legacy source.

A continuation revision adds this project-owned structure:

```text
D:\LTX_Supervisor_Storage\jobs\{job_id}\scenes\scene_{id}\revisions\{revision}\
  frame.png
  video.mp4
  generation-manifest.json
  chunks\
    chunk_0000\
      attempts\
        0001\
          stage1_handoff.safetensors
          stage1_handoff.json
          stage2_video.safetensors
          stage2_video.json
          stage2_audio.safetensors
          stage2_audio.json
          window.mkv
          COMPLETE.json
  assembly\
    continuation-plan.json
    COMPLETE.json
    discord-delivery.json
```

`video.mp4` remains the revision-facing, raw scene artifact used by the GUI, remakes, project concat, and later
upscaling. `assembly\COMPLETE.json` binds that path and hash to the exact continuation plan and selected chunk
artifact hashes. `discord-delivery.json` persists delivery ownership/status, prompt ID, and the scene/plan/workflow
hashes needed to reclaim a queued or completed send safely; it never stores watermarked media.

The SQLite store adds immutable `continuation_plans`, `scene_chunks`, and `chunk_attempts`. Attempt seeds are stored
as text to preserve all unsigned 64-bit values. A selected attempt's `COMPLETE.json` hash becomes the required
upstream artifact hash for its successor. Selecting or invalidating a different upstream attempt marks descendants
stale/invalidated, so a superficially similar downstream MP4 can never bypass latent lineage.

Video latent checkpoints accept only `[1, 128, frames, height, width]` floating-point samples. The second pass also
persists its non-empty batch-one LTX audio latent. Both allow only the narrow supported auxiliary fields.
Safetensors data is moved to CPU, flushed, atomically renamed, then described by a schema-versioned JSON manifest
containing identity, byte size, tensor descriptors, and SHA-256. The per-attempt `COMPLETE.json` is written last and
records the plan, upstream hash, prompts, seed, expected raw and committed frame counts, all three checkpoint hashes,
raw-window hash, and workflow results. File existence alone never means that an attempt succeeded.

On restart the renderer verifies selected manifests, hashes, exact expected video temporal-token counts, latent
descriptors, lossless raw-window geometry/codec/pixel format, frame count, audio presence, and lineage. A valid
stage-one checkpoint resumes at stage two without repeating the first pass. Valid completed stage-two video and
audio checkpoints can recreate a missing FFV1/yuv444p `window.mkv` through decode/mux only, without another
diffusion pass. A valid window is reused directly. If an accepted artifact fails verification, that chunk and every
descendant are invalidated and recovery starts at the earliest bad dependency. The narrow crash window after
writing `COMPLETE.json` but before selecting the attempt is also reconciled. A valid scene `video.mp4` plus assembly
manifest is reused without another FFmpeg encode.

Stage-one, stage-two, and Discord delivery prompt IDs are persisted immediately after `/prompt` returns and before
the blocking wait. After a supervisor or controlled ComfyUI restart, the worker first reclaims successful history
or waits for the exact still-queued project prompt. If that owned prompt is absent from both queue and history, the
same immutable chunk workflow can be requeued; a workflow/input hash mismatch invalidates it. The queue-before-SQLite
window cancels only that project prompt when durable ownership cannot be committed. Discord delivery is stricter:
ambiguous queue/history or an ownership mismatch remains failed/reclaimable and blocks automatic resend to avoid a
duplicate Patreon post.

GUI remake I2V completion and Discord delivery are separate durable phases. Before release of I2V ownership, the
controller probes the raw clip and atomically records its job/scene/revision, selected route, parameter hash,
starting-frame hash, raw-video hash/path, and fixed production profile in `generation-manifest.json`. Recovery skips
diffusion only when that complete identity still verifies; it then resumes the idempotent Discord-delivery phase.
Corrupt media, changed parameters/frame content, a mismatched route, or an unprobed profile forces regeneration.

The GUI revision picker uses the selected revision's stored human-readable parameter document—not the source scene
document—for both preview and editor state. A selected revision can therefore be inspected or used as the basis for
a new remake with its exact prior prompts, seeds, LoRAs, sampler choices, sigmas, and routing values. Temporary
browser edits are kept per source revision while the remake tray retains only the currently selected scene edit.

Manual master-final requests are durable SQLite records. Clicking the project-level control snapshots the included
scene IDs, their latest successful revision numbers, and immutable raw clip paths at request time. The controller's
single worker runs queued manual finals only after active project rendering has ended, validates the fixed video
profile, then uses the existing FFmpeg copy-concat into the normal final path. This is FFmpeg-only work: it does not
call ComfyUI or alter pipeline state. Scene inclusion is a persistent scene attribute used solely by the manual
request; automatic first-run assembly still concatenates its own successful scene records unchanged.

Edits are validated by reconstructing the typed job contract, then passed as explicit workflow overrides. This
keeps the global character LoRA, stage LoRA separation, mandatory DMD/Joy chain, Pony detailer, samplers, schedulers,
sigmas, CFG, denoise, chunking, upscaler, seeds, prompts, and fixed production profile in one routing
implementation. Browser-facing seeds are strings so JavaScript cannot truncate unsigned 64-bit values. A
video-only revision requires an existing cached frame; image-only is not representable.

Each scene stores independent T2I/I2V attempt counts and its last ComfyUI prompt ID. On a transient prompt failure,
only the unfinished stage is retried. A timed-out prompt is deleted if pending or interrupted only when that exact
prompt ID is the running project prompt. On process recovery, succeeded scenes and their deterministic artifacts are
left untouched.

`run_forever` owns a daemon status reporter with a configurable default interval of 15 seconds. Because it runs
independently of the synchronous pipeline tick, it remains active while Gmail polling sleeps, the live ComfyUI route
resolves/downloads an asset, or a long T2I/I2V prompt executes. Each line reads only the durable state snapshot and
redacted queue totals (`running`/`pending`); workflow bodies, prompts, download URLs, and secrets are never logged.
Phase-level INFO messages identify asset names/results, scene stage attempts, cache reuse, and assembly boundaries.

LoRA resolution runs through a loopback-only custom ComfyUI route so the standalone supervisor uses the active
server process's exact `folder_paths` roots and selectable filenames. Dynamic assets are de-duplicated by Civitai
version ID, falling back to normalized download URL, rather than display name. This prevents one version repeated
under different JSON names from being downloaded or injected twice.

T2I and I2V use separate eligibility sets. The global character LoRA and every scene T2I LoRA are excluded from
I2V by stable asset identity, including aliases with different display names. Every remaining dynamic I2V candidate
is resolved for the LTX 2.3 target; Civitai `baseModel` metadata must identify the asset as LTX 2.x. This includes
Civitai's compact `LTXV2` label and versioned 2.x labels, while rejecting LTX 1.x, Anima, Pony, SDXL, Flux, and
unverifiable assets before manifest or local-file acceptance. The supervisor stores validated I2V filenames under
an I2V-specific key, and the workflow builder fails closed if that key is absent. Mandatory local DMD and JoyAI are
the only exceptions because their exact filenames and weights are project constants.

For a missing Civitai asset, public model-version metadata confirms that it is a LoRA, selects a primary
virus-scanned SafeTensor, obtains its canonical filename/size/hash, and performs a second local lookup before any
transfer. Downloads use the DPAPI-protected project Civitai token, redirect-following retries, an atomic partial file,
free-space preflight, and SHA-256 verification when supplied. The token is attached only to Civitai download URLs
and is never returned to the supervisor or logs. Each failed asset is reported independently so a scene can fail
without cancelling unrelated scenes. Mandatory DMD and JoyAI I2V LoRAs remain local-only because no trusted download
URL was supplied for them.

Continuation windows are never H.264-encoded individually and stream-copied together. Each worker result is a
lossless FFV1/yuv444p `window.mkv`. FFprobe first requires that codec/pixel format, 768×1344, exact 24/1 CFR, the
planned decoded-frame count, and an audio stream for every accepted raw window. FFmpeg then trims exact frame
ranges, removes each later window's eight-frame causal preroll, resets timestamps, applies 100 ms quarter-sine
audio edge fades without overlapping samples or changing segment duration, and performs the route's one lossy scene
encode. That encode is H.264 High profile, yuv420p, CRF 19 with the slow preset, closed GOP 48, and stereo
48 kHz/192 kbps AAC. The temporary result must pass exact decoded-frame, codec, pixel-format, and audio-profile
validation before atomically replacing the revision's clean `video.mp4`.

Before stitching, FFmpeg preflight verifies every successful clip is 768×1344 at 24 fps. The concat operation uses
stream copy and emits `D:\LTX_Supervisor_Storage\finals\{job_id}_final.mp4`.

VHS writes scene video to its temporary ComfyUI output and returns metadata through prompt history. The supervisor
downloads only the `WorkflowBuild.output_node_id` VHS output through the local HTTP API into the matching versioned
scene directory under `D:\LTX_Supervisor_Storage\jobs`; it never scans other output nodes, including Discord
delivery, or moves unrelated shared output files.

Output delivery is never a replacement for durable artifacts. The clean T2I result feeds the deterministic frame
saver and the I2V cache; a second edge passes through `DaSiWa_Watermark` into `DiscordSendSaveImage`. The legacy
single-window I2V graph similarly keeps its raw VHS edge separate from its watermarked Discord edge. Continuation
worker graphs contain no watermark or sender at all: after assembly, a separate graph reloads the verified raw
scene, watermarks only that frame stream, and sends it with the decoded audio. Discord nodes are the only media
senders, strip workflow metadata, use `save_output=false`, and do not retain a second local copy. The webhook is
loaded from the project DPAPI secret store at graph-build time. Versioned templates contain a nonfunctional
placeholder; only the approved shared GUI copies and runtime-generated API graphs receive the encrypted runtime
value.

The controlled Windows restart script resolves the expected Easy Install paths, verifies that the process listening
on port 8188 is the expected embedded Python executable, stops only that process, launches the unchanged
`Start ComfyUI.bat` hidden, and waits for HTTP health. The console and GUI launchers both use this same guard when
the local API is unavailable, so a post-reboot GUI launch starts ComfyUI with its existing Sage Attention flags.
It is also called for fatal ComfyUI availability failures and once at GUI takeover when any required project node
contract is stale. The guard requires Save Scene Frame revision support, both Save/Load Chunk Latent artifact-kind
options (`stage1_handoff`, `stage2_video`, `stage2_audio`), and Load Chunk Latent's
`expected_temporal_tokens`. A stale-contract
reload is refused while any ComfyUI prompt is running or pending; after restart the same contracts must be present
or GUI startup fails.

The one-click launcher is also project-local; it does not edit shared ComfyUI startup scripts or global environment
configuration. Non-secret settings live in `D:\LTX_Supervisor_Storage\config\settings.env`. App Passwords, OAuth
client secrets, refresh tokens, Civitai tokens, and the Discord webhook live in
`D:\LTX_Supervisor_Storage\config\secrets.json` after encryption with Windows DPAPI for the current Windows user.
Explicit process environment variables take precedence over saved project values. OAuth uses a loopback desktop
callback, PKCE, state validation, offline access, the full Gmail IMAP/SMTP scope, and read-only Google Drive scope.
`GmailClient` exchanges the stored refresh token for short-lived access tokens and caches them only in memory.

LAN mode is explicit (`TENMIN_GUI_LAN_ENABLED`) and requires the DPAPI-protected
`TENMIN_GUI_LAN_PASSWORD`; startup refuses a LAN bind without a 12+ character password. The GUI binds `0.0.0.0` only
when enabled, applies Basic authentication to every non-loopback request (static files, APIs, media, and events), and
uses fixed username `10min`. It never exposes ComfyUI `:8188`; Windows Firewall remains user-controlled and should be
limited to Private networks. The mobile stylesheet changes the three-column desktop workspace into independently
scrollable history/scene sections followed by a one-column detail/editor view. LoRA pickers derive options live from
`LoraLoader` and `LoraLoaderModelOnly` contracts, preserving stage separation and server-visible filenames.

Outbound requests use `Run the LTX video pipeline`; inbound jobs use a separate exact subject,
`LTX_JOB_COMPLETE`. The completion must be a new unread message. IMAP narrows candidates by that subject, then the
client applies an exact decoded-header comparison because Gmail's `SUBJECT` search is substring-based. The durable
poller repeats the exact-subject gate before sender and payload validation, preventing the outbound trigger, replies,
forwards, and similarly named messages from entering the job state machine. Candidate messages are retrieved with
`BODY.PEEK[]`; fetching a batch cannot set `\Seen` on messages that the current single-job state machine has not
accepted. The console reports candidate count, parser rejection, and duplicate-job decisions without logging mail
content, links, or credentials.

Incoming job precedence is `.json` attachment, valid JSON in the plain-text body, then a supported
`drive.google.com/file/d/...` (or `open?id=`/`uc?id=`) file link found in plain text or HTML. Drive folder links and
arbitrary URLs are rejected. Downloads are capped at 5 MiB and may redirect only to approved Google download hosts.
Public files are attempted anonymously. A sign-in redirect outside the approved content-host allowlist is classified
as an access failure without consuming its response body, so OAuth mode falls back to the Drive API for files shared
with the configured Gmail account. An authenticated 404 is reported as missing/not-shared rather than as an OAuth
failure. Authentication/network failures leave the email unread for retry, while a successfully downloaded malformed
payload is claimed as invalid under the existing mailbox rules. A claimed invalid message with `job_id=NULL` may
later be atomically upgraded to a real job if its Drive file is corrected or a compatible parser repair makes it
valid; accepted message/job IDs remain immutable duplicates. Before strict contract validation, the mail parser
performs one narrow Grok compatibility normalization: redundant leading zeroes are removed only from unquoted
integer values whose JSON key is `seed` or `original_seed`. It does not rewrite string content or relax any other
JSON syntax.

If every scene fails asset preparation, the supervisor transitions to `error` and stops polling for replacement
jobs. The one-click launcher detects any active saved state, including asset resolution, T2I, I2V, stitching, and
error, and offers an atomic resume or abandonment. Resume clears unfinished scene errors and prompt IDs while
leaving succeeded scenes and attempt counters durable. A saved stitching job with no unfinished scenes can resume
final assembly. A partial asset failure still allows successful scenes to render and stitch.

Declining performs a separate atomic abandonment transition. Before changing durable state, the launcher asks
ComfyUI to delete pending prompts and interrupt a running prompt only when its queue metadata carries the
`10MinVideoMaker-supervisor` client ID. Unfinished scenes are then marked `cancelled`, their attempt history and the
original job payload remain available for diagnosis, and the singleton pipeline state is cleared to `idle` with no
active job. This prevents the supervisor from reopening the rejected job without touching other ComfyUI clients.

The GUI exposes the same abandonment path as **Cancel project** / `POST /api/pipeline/cancel-current` for held
states (`downloading_assets`, `running_t2i`, `running_i2v`, `stitching`, `error`, and `awaiting_review`). Active
renders cancel only project-owned ComfyUI prompts first. Status includes `can_cancel_current_project` so the button
stays hidden when the pipeline is already `idle` or `waiting_for_grok`. After cancel, an `idle` tick polls unread
handoffs before sending a new request email, so a job already sitting in Gmail starts without an extra Grok request.

## Production profile

- Image/video size: 768×1344.
- Frame rate: 24 fps.
- LTX frame count: `8n + 1`, derived by rounding up to cover a scene's requested duration.
- Maximum LTX scene duration: 32 seconds.
- T2I Anima: one 30-step `er_sde`/`beta57` pass at CFG 4.5; no face detailer.
- T2I Pony: 30-step `res_3m_ode` followed by 30-step `res_5s_ode` at CFG 6, then
  `UltralyticsDetectorProvider` (`bbox/face_yolov8m.pt`) and the reference `FaceDetailer` settings.
- I2V: two LCM sampling passes, with separate verified sigma schedules and the LTX spatial upscaler.

The workflow templates will be rebuilt independently from live node contracts. The approved reference workflows are never written to or copied into this repository.

Scene workflows are built dynamically from the validated job rather than mutating user-owned workflow JSON. The T2I
builder selects Anima or Pony from `character.lora.base`, applies the character LoRA once, adds any scene LoRAs, and
uses the exact family sampler route. Pony's decoded image is face-detailed before it reaches the deterministic frame
saver; Anima bypasses the detector/detailer entirely. The I2V builder consumes the deterministic PNG produced by the matching T2I
scene, adds DMD and JoyAI before dynamic model-only LoRAs, enables feed-forward chunking, and uses separate LCM
samplers and sigma schedules around the tiled spatial upscaler.

GUI workflow export inserts ComfyUI's separate `fixed` seed-control widget after sampler and face-detailer seeds.
That widget is not part of the API input contract, but it is required in `widgets_values`; omitting it shifts every
later canvas field and can display CFG as the step count.

The x2 spatial upscaler uses a 384×672 first-pass latent and emits the fixed 768×1344 production clip. Every axis is
divisible by 32 (`384=12×32`, `672=21×32`, `768=24×32`, `1344=42×32`), so `EmptyLTXVLatentVideo` does not quantize
either side. The decoded video connects directly to `VHS_VideoCombine`; no final crop or resize exists in the route.

Assembly profile failures are caught explicitly. The supervisor transitions the saved job to `error`, preserves
completed clips, and stops requesting replacement jobs instead of repeating the same failing stitch every polling
interval.

## No-render validation

- `python -m unittest discover -s tests -v`
- `python -m compileall -q tenminvideomaker scripts __init__.py`
- `python scripts/setup_and_start.py --help`
- `python scripts\validate_continuation_workflows.py`
- `git diff --check`
- The continuation validator builds initial, decoded-guide, later, and final graphs for both passes plus post-assembly delivery,
  fetches only their live `/object_info/<node type>` contracts, and never posts to `/prompt`.
- Restart ComfyUI only with an empty queue, then query `/object_info/<node type>` for the project types, including
  `10MinVideoMaker_SaveChunkLatent` and `10MinVideoMaker_LoadChunkLatent`.
- POST a local-only mandatory-LoRA lookup to `/10minvideomaker/assets/resolve`; this checks the live roots without
  downloading or loading a model.
- Run `python scripts/export_workflows.py --install-approved-shared-copies` while ComfyUI is healthy. Export refuses
  unavailable classes or mismatched routes, lays out nodes by dependency depth, checks node overlaps and group bounds,
  and writes both API and GUI forms.

These checks prove graph/schema consistency and deterministic accounting only. The separate
`scripts\run_continuation_acceptance.py` creates a unique D-drive-only test job from one existing cached frame,
runs common-base, single-frame, decoded-17-frame-guide, and latent-overlap cases without touching the supervisor
database state, captures ComfyUI peak-VRAM/runtime telemetry plus FFmpeg/Pillow/OpenCV seam evidence, and ends in
human review. It cannot enable `auto` or create the approval manifest. Until GPU evidence and human visual review
exist, continuation has no production-quality claim and `explicit` remains the default.

The diagnostic guide is frames 96–112, not the final 17 frames: live LTX testing shows that its initial refined
latent accepts this 20-token span but rejects the later 104–120 range. This does not change the normal later-window
25-frame production overlap, which remains frames 96–120.
