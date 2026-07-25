# 10MinVideoMaker architecture

The automated pipeline has two layers that call the same pure-Python services:

- **Supervisor**: the 24/7 owner of Gmail polling, ComfyUI API job submission, scene retries, FFmpeg assembly, and controlled restart requests. It polls once every five minutes; it never blocks ComfyUI's execution queue by sleeping inside a node.
- **ComfyUI nodes**: interactive/status controls built on the same state, payload, mail, asset, and assembly services. They expose no independent routing or persistence rules.

The running ComfyUI 0.27.1 installation did not expose the package's initial V3 entrypoint through `/object_info`.
The package therefore registers its compatibility surface through `NODE_CLASS_MAPPINGS`. Business logic remains
outside the node classes, so this registration choice does not fork the automation implementation.

`runtime/pipeline.sqlite3` is the local durable state store. It is intentionally ignored by Git and records one global pipeline state plus per-scene states. A job is accepted only from `idle` or `waiting_for_grok`; completed scene artifacts remain intact when unfinished scenes are re-queued.

Each scene stores independent T2I/I2V attempt counts and its last ComfyUI prompt ID. On a transient prompt failure,
only the unfinished stage is retried. A timed-out prompt is deleted if pending or interrupted only when that exact
prompt ID is the running project prompt. On process recovery, succeeded scenes and their deterministic artifacts are
left untouched.

LoRA resolution runs through a loopback-only custom ComfyUI route so the standalone supervisor uses the active
server process's exact `folder_paths` roots and selectable filenames. Dynamic assets are de-duplicated by Civitai
version ID, falling back to normalized download URL, rather than display name. This prevents one version repeated
under different JSON names from being downloaded or injected twice.

For a missing Civitai asset, public model-version metadata confirms that it is a LoRA, selects a primary
virus-scanned SafeTensor, obtains its canonical filename/size/hash, and performs a second local lookup before any
transfer. Downloads use the DPAPI-protected project Civitai token, redirect-following retries, an atomic partial file,
free-space preflight, and SHA-256 verification when supplied. The token is attached only to Civitai download URLs
and is never returned to the supervisor or logs. Each failed asset is reported independently so a scene can fail
without cancelling unrelated scenes. Mandatory DMD and JoyAI I2V LoRAs remain local-only because no trusted download
URL was supplied for them.

Before stitching, FFmpeg preflight verifies every successful clip is 704×1248 at 24 fps. The concat operation uses stream copy and emits `D:\output\10minfinals\{job_id}_final.mp4`; the folder is created only when a completed job is actually assembled.

VHS writes scene video to its temporary ComfyUI output and returns metadata through prompt history. The supervisor
downloads that exact output through the local HTTP API into
`D:\output\10minfinals\.work\{job_id}\clips\scene_{id}.mp4`; it does not scan or move unrelated shared output files.

The controlled Windows restart script resolves the expected Easy Install paths, verifies that the process listening
on port 8188 is the expected embedded Python executable, stops only that process, launches the unchanged
`Start ComfyUI.bat` hidden, and waits for HTTP health. It is called only for fatal ComfyUI availability failures.

The one-click launcher is also project-local; it does not edit shared ComfyUI startup scripts or global environment
configuration. Non-secret settings live in ignored `.env`. App Passwords, OAuth client secrets, and refresh tokens
live in ignored `runtime/secrets.json` after encryption with Windows DPAPI for the current Windows user. Explicit
process environment variables take precedence over saved project values. OAuth uses a loopback desktop callback,
PKCE, state validation, offline access, the full Gmail IMAP/SMTP scope, and read-only Google Drive scope.
`GmailClient` exchanges the stored refresh token for short-lived access tokens and caches them only in memory.

Outbound requests use `Run the LTX video pipeline`; inbound jobs use a separate exact subject,
`LTX_JOB_COMPLETE`. The completion must be a new unread message. IMAP narrows candidates by that subject, then the
client applies an exact decoded-header comparison because Gmail's `SUBJECT` search is substring-based. The durable
poller repeats the exact-subject gate before sender and payload validation, preventing the outbound trigger, replies,
forwards, and similarly named messages from entering the job state machine.

Incoming job precedence is `.json` attachment, valid JSON in the plain-text body, then a supported
`drive.google.com/file/d/...` (or `open?id=`/`uc?id=`) file link found in plain text or HTML. Drive folder links and
arbitrary URLs are rejected. Downloads are capped at 5 MiB and may redirect only to approved Google download hosts.
Public files are attempted anonymously. A sign-in redirect outside the approved content-host allowlist is classified
as an access failure without consuming its response body, so OAuth mode falls back to the Drive API for files shared
with the configured Gmail account. An authenticated 404 is reported as missing/not-shared rather than as an OAuth
failure. Authentication/network failures leave the email unread for retry, while a successfully downloaded malformed
payload is claimed as invalid under the existing mailbox rules.

If every scene fails asset preparation, the supervisor transitions to `error` and stops polling for replacement
jobs. The one-click launcher detects that saved job and offers an atomic retry: unfinished scene errors and prompt IDs
are cleared, succeeded scenes remain untouched, and attempt counters remain durable. A partial asset failure still
allows successful scenes to render and stitch.

Declining that retry performs a separate atomic abandonment transition. Unfinished scenes are marked `cancelled`,
their attempt history and the original job payload remain available for diagnosis, and the singleton pipeline state
is cleared to `idle` with no active job. This prevents the supervisor from reopening the rejected job while allowing
the next idle tick to request and accept a new payload.

## Production profile

- Image/video size: 704×1248.
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

The x2 spatial upscaler requires a half-resolution first-pass latent. Its internal dimensions are 352×624 so the
requested production dimensions are represented at half scale. However, `EmptyLTXVLatentVideo` quantizes each
half-resolution side to a 32-pixel grid. Although 1248 is divisible by 32, its half-height 624 is not; the live x2
route therefore decodes at 704×1216. A final core `ImageScale` performs Lanczos scale-to-fill and a centered crop to
the only saved production size, 704×1248. This removes about nine pixels from each horizontal edge, preserves
subject proportions, and does not expose an alternate output resolution.

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
