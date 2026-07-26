# 10MinVideoMaker architecture

The pipeline has three layers that call the same pure-Python services:

- **Loopback GUI**: a FastAPI/vanilla-JavaScript frontend at `127.0.0.1:8765`. It maps stored jobs to
  human-readable controls, previews versioned media, accepts approvals and revision batches, and never exposes raw
  filesystem paths or raw JSON.
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

Every scene revision is stored below
`D:\LTX_Supervisor_Storage\jobs\{job_id}\scenes\scene_{id}\revisions\{revision}` with `frame.png`,
`video.mp4`, and a human-readable `generation-manifest.json`. Source Grok payloads live in each job's `source`
folder, and finals live in `D:\LTX_Supervisor_Storage\finals`. The first GUI launch copies the prior SQLite
database, configured secrets/settings, payloads, and only media recorded by this project into the new layout.
Migration never deletes or rewrites the legacy source.

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

Before stitching, FFmpeg preflight verifies every successful clip is 768×1344 at 24 fps. The concat operation uses
stream copy and emits `D:\LTX_Supervisor_Storage\finals\{job_id}_final.mp4`.

VHS writes scene video to its temporary ComfyUI output and returns metadata through prompt history. The supervisor
downloads that exact output through the local HTTP API into the matching versioned scene directory under
`D:\LTX_Supervisor_Storage\jobs`; it does not scan or move unrelated shared output files.

Output delivery is a parallel branch, not a replacement for durable artifacts. The clean T2I result feeds the
deterministic frame saver and the I2V cache; a second edge passes through `DaSiWa_Watermark` into
`DiscordSendSaveImage`. Likewise, normalized decoded I2V frames feed the temporary VHS clip unchanged while a
parallel watermark edge feeds `DiscordSendSaveVideo` with the same decoded audio. Discord nodes are the only media
senders, strip workflow metadata, and do not retain a second local copy. The webhook is loaded from the project
DPAPI secret store at graph-build time. Versioned templates contain a nonfunctional placeholder; only the approved
shared GUI copies and runtime-generated API graphs receive the encrypted runtime value.

The controlled Windows restart script resolves the expected Easy Install paths, verifies that the process listening
on port 8188 is the expected embedded Python executable, stops only that process, launches the unchanged
`Start ComfyUI.bat` hidden, and waits for HTTP health. It is called for fatal ComfyUI availability failures and once
at GUI takeover when the live Save Scene Frame contract lacks revision support. The takeover reload is refused
while any ComfyUI prompt is running or pending.

The one-click launcher is also project-local; it does not edit shared ComfyUI startup scripts or global environment
configuration. Non-secret settings live in `D:\LTX_Supervisor_Storage\config\settings.env`. App Passwords, OAuth
client secrets, refresh tokens, Civitai tokens, and the Discord webhook live in
`D:\LTX_Supervisor_Storage\config\secrets.json` after encryption with Windows DPAPI for the current Windows user.
Explicit process environment variables take precedence over saved project values. OAuth uses a loopback desktop
callback, PKCE, state validation, offline access, the full Gmail IMAP/SMTP scope, and read-only Google Drive scope.
`GmailClient` exchanges the stored refresh token for short-lived access tokens and caches them only in memory.

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
- `git diff --check`
- Restart ComfyUI, then query `/object_info/<node type>` for all eight `10MinVideoMaker_*` types.
- POST a local-only mandatory-LoRA lookup to `/10minvideomaker/assets/resolve`; this checks the live roots without
  downloading or loading a model.
- Queue `10MinVideoMaker_ReleaseMemory` alone as a harmless API smoke test. This verifies real execution without
  loading a model or generating media.
- Run `python scripts/export_workflows.py --install-approved-shared-copies` while ComfyUI is healthy. Export refuses
  unavailable classes or mismatched routes, lays out nodes by dependency depth, checks node overlaps and group bounds,
  and writes both API and GUI forms.
