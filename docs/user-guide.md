# 10MinVideoMaker user guide

## Current boundary

The project can validate incoming jobs, poll/send Gmail, resolve LoRAs, build and queue per-scene generation graphs,
cache the exact T2I frame, download the matching I2V clip, validate/stitch completed clips, request the next job, and
recover unfinished scenes. Its browser GUI adds a historical job library, human-readable parameter editing,
versioned previews, and multi-scene remake batches. Incoming jobs auto-start by default; manual approval is an
optional test/review launch mode.

Persistent data is rooted at `D:\LTX_Supervisor_Storage`. Nothing new is written to the C: drive except the
version-controlled custom-node source already in this repository.

## Human-review GUI

Double-click `Start 10MinVideoMaker.bat`. After the existing credential checks, the launcher opens
`http://127.0.0.1:8765/`. Default binding is only this computer.

You do not need to start ComfyUI manually after a PC reboot. The GUI launch uses the same local health guard as the
console launcher: if `http://127.0.0.1:8188` is down, it starts the verified Easy Install `Start ComfyUI.bat`, which
preserves its Sage Attention startup configuration, and waits for the API before starting the supervisor.

If you leave **Change optional environment settings before starting? [y/N]** unanswered, the Windows console waits
ten seconds, prints that it is using **No**, and continues automatically. This timeout applies only to that optional
settings question; credential and recovery decisions still wait for your explicit input.

For phone use, choose **Change optional environment settings** then **Configure mobile LAN access**. Enable it and
set a 12+ character password. The next GUI start binds to the private LAN and logs a phone URL such as
`http://192.168.x.x:8765/`. Sign in with username `10min` and that password. This is HTTP Basic on a trusted private
LAN, not HTTPS: do not port-forward it, do not use public/untrusted Wi-Fi, and never expose ComfyUI port 8188. If
Windows Firewall asks, allow this GUI only on **Private** networks.

On a phone the interface intentionally uses one screen at a time:

1. Start with **Projects** only.
2. Selecting a project replaces that list with its **Scenes** and a **← Projects** button.
3. Selecting a scene replaces the list with its details. A sticky **Scenes** button and compact scene dropdown stay
   at the top, so you can switch scenes or return to the list without losing the project.

The generated video uses the device's native HTML5 controls. Tap its fullscreen control for true device fullscreen;
the video remains inline while you review parameters, then returns to the same scene editor when closed.

1. Select a job in **Project library**, then select a scene.
   Project names use `Character · MM/DD/YYYY`; the internal job ID remains hidden as the routing key. The project
   library and scene list each scroll independently when they contain more rows than fit on screen.
2. Review the source frame, generated video, prompts, seeds, character and stage LoRAs, T2I passes, Pony face
   detailer, I2V samplers and sigma schedules, chunking, upscaler, and fixed production profile.
   **Result version** selects one immutable generation record: its preview and every form field change together to
   the exact settings used for that version. This makes an older remake safe to inspect or use as the starting point
   for a new remake without silently reverting to version 1.
3. New Gmail handoffs start automatically. To test an incoming payload before it starts, launch with
   `Start 10MinVideoMaker.bat --hold-new-jobs-for-review`; that session shows **Approve & Queue Job** for each
   newly claimed handoff.
4. To revise a scene, enable **Mark for remake** and choose:
   - **Video Only** to reuse that revision's existing cached frame and run only I2V.
   - **Image + Video** to generate a new frame and force the matching I2V workflow to use it.
5. Edit fields and add/remove LoRAs. Each T2I/I2V editor has an **Installed local file** picker populated live from
   its matching ComfyUI LoRA loader. Selecting it fills the LoRA name; retain a valid HTTPS download URL for durable
   asset identity. I2V choices remain subject to the LTX 2.x compatibility gate. Mandatory I2V DMD 1.0 and JoyAI 0.5 and the 768×1344/24 fps profile remain
   locked safety invariants.
6. Continue selecting scenes from this or other jobs. The tray keeps the edits until **Save & Remake** is clicked.
7. If standard work is rendering, choose **Queue edits to run after current job finishes** or
   **Interrupt/Cancel current job and run edits immediately**. Interrupt targets only prompts owned by this
   project, preserves the interrupted job's history, and does not restart healthy ComfyUI.

There is intentionally no image-only choice: a changed starting image always requires a new video. Every submitted
edit creates a new numbered revision; prior frames, clips, parameters, and manifests remain available in history.

### Efficient render ordering

The supervisor avoids swapping the image and LTX models between individual scenes. For a normal job, it makes all
uncached T2I starting frames, releases the image model once, then renders all remaining I2V clips from those exact
frames. For a remake batch, all **Image + Video** frame remakes happen first (Anima and Pony frames are grouped when
needed), followed by every batch item's I2V render, including **Video Only** remakes. This is deliberate for the
16 GB GPU: it avoids repeatedly unloading and reloading the multi-gigabyte models.

## Workflow templates

The repository and the approved ComfyUI workflow folder contain three independently rebuilt GUI workflows:

- `10MinVideoMaker_T2I_Anima.json`
- `10MinVideoMaker_T2I_Pony.json`
- `10MinVideoMaker_I2V_LTX23_TwoPass.json`

Matching `.api.json` files are versioned in the repository for headless generation and testing. Static templates use
the safe payload in `examples/example_job.json`; production jobs are turned into fresh API graphs so seeds, prompts,
LoRAs, scene IDs, frame counts, and cached-frame paths cannot leak between scenes.

Do not queue the example workflows for a real render without replacing the example LoRA. Its
`https://example.invalid/` URL is intentionally non-functional.

## Fixed generation profile

- Final image and video: 768×1344.
- Frame rate: 24 fps.
- Frame count: smallest `8n + 1` value covering the requested duration.
- Maximum scene length: 32 seconds.
- Anima T2I: 30 steps, CFG 4.5, `er_sde`, `beta57`.
- Pony T2I: 30 steps per pass, CFG 6, `res_3m_ode` then `res_5s_ode`, `karras`.
- Pony post-process: `bbox/face_yolov8m.pt` bbox detection followed by the reference `FaceDetailer` settings
  (20 detailer steps, CFG 5, `dpmpp_2m_sde`, `karras`, denoise 0.38). Anima does not use this detailer.
- LTX I2V: LCM on both passes, verified distinct sigma lists, x2 tiled spatial upscaler, DMD 1.0, JoyAI 0.5.

### LoRA stage boundary

- `character.lora` and `scenes[].t2i.loras[]` are Anima/Pony T2I assets only.
- The same asset is ignored if Grok repeats or aliases it in `scenes[].i2v.loras[]`.
- `ltxv_character_lora` and every remaining `scenes[].i2v.loras[]` item must report a Civitai LTX 2.x base model.
  Compact `LTXV2` and versioned labels such as `LTXV 2.0`, `LTXV 2.2`, and `LTXV 2.3` are accepted for LTX 2.3.
  LTX 1.x, image-model, and unverifiable bases are rejected before the workflow is queued, even if the file exists.
- Mandatory DMD 1.0 and JoyAI 0.5 are applied separately and are not sourced from scene JSON.

This means Grok may list genuine LTX 2.x motion/character LoRAs in I2V fields. It should never repeat an Anima,
Pony, SDXL, Flux, or other image-model LoRA there.

### Patreon Discord delivery

Each production scene sends two metadata-free, watermarked Patreon previews through the installed
DiscordSendSave nodes:

- Image: `wm.png`, bottom-right, scale 0.70, transparency 0.40, 20-pixel padding; lossless PNG, quality 100.
- Video: the same watermark settings on the final 768×1344 frame batch; H.264 MP4 with generated audio, 24 fps,
  quality 65.

The Discord nodes send to the webhook only. `save_output`, previews, prompt inclusion, workflow JSON, CDN URL
storage, and GitHub updates are disabled. The clean deterministic T2I PNG remains the I2V input, and the clean
temporary VHS clip remains the master-assembly input; watermarking cannot feed back into generation. The supervisor
stores a durable clip only from the designated raw VHS node output, never from the Discord delivery node.

The webhook is encrypted in `D:\LTX_Supervisor_Storage\config\secrets.json` with Windows DPAPI. The versioned workflow files contain only a
nonfunctional placeholder, while the approved shared workflow copies receive the configured webhook during export.
To replace it later, run the launcher, choose optional settings, then choose **Configure/change Discord webhook**.

The GUI templates store a separate `fixed` seed-control value after each sampler/detailer seed. This keeps the
canvas widgets aligned, so the Pony samplers visibly show 30 steps rather than incorrectly displaying CFG 6 in the
steps field.

The first LTX pass uses an internal 384×672 latent because the mandated spatial model is x2. Both first-pass axes
and both final 768×1344 axes are divisible by 32, so the spatial pass decodes directly to the saved production
clip. No post-decode crop or resize is applied.

## One-click setup and start

Double-click `Start 10MinVideoMaker.bat` in the project root. Do not use the shared ComfyUI start scripts for this
project setup.

### Windows shortcut

The repository includes a project icon and a repeatable shortcut installer:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_windows_shortcuts.ps1
```

It creates `10MinVideoMaker.lnk` on the current user's Desktop and in Start Menu Programs. Both shortcuts launch
the existing project batch file through `cmd.exe`, use the repository as their working directory, and retain the
custom icon. To pin it, right-click the Desktop or Start Menu shortcut and choose **Pin to Start** or
**Pin to taskbar**; on Windows 11, the taskbar command may be under **Show more options**. Running the installer
again refreshes these two project-owned shortcuts.

On the first run, the launcher detects missing Gmail settings and offers:

- **Google App Password**: opens Google's App Password page when requested, then securely prompts for the
  16-character value. The Google account must have 2-Step Verification enabled.
- **OAuth2 browser login**: opens the Google Cloud credentials and Drive API pages, asks for a **Desktop app** OAuth
  client ID and client secret, then prints and opens a Google authorization URL. After consent, Google redirects to
  a temporary loopback listener on this computer and the launcher stores the refresh token.

OAuth requests `https://mail.google.com/` for Gmail SMTP/IMAP and
`https://www.googleapis.com/auth/drive.readonly` for private Drive job links. The Google Drive API must be enabled in
the same Cloud project as the Desktop OAuth client, and the OAuth consent screen's **Data Access** list must include
that Drive scope. For uninterrupted operation, an external OAuth consent screen must be published to **In
production**; refresh tokens from an external project left in **Testing** expire after seven days. A personal app may
still show Google's unverified-app warning because these scopes are sensitive or restricted.

Existing mail-only OAuth grants are detected on the next launcher run. The launcher reuses the encrypted client ID
and secret, opens the Drive API page, and performs a one-time browser reauthorization for the additional read-only
scope.

Secrets are not written to versioned workflow JSON or Git. The launcher encrypts App Passwords, OAuth client secrets,
OAuth refresh tokens, the Civitai API token, and the Discord webhook with Windows DPAPI for the current Windows user and stores the
ciphertext in `D:\LTX_Supervisor_Storage\config\secrets.json`. Non-secret values are stored in
`D:\LTX_Supervisor_Storage\config\settings.env`.
Existing process environment variables override saved project values.

If all required values already exist, the launcher asks whether to change optional settings. Choosing yes displays
the editable values and a Gmail reconfiguration option. Choosing no proceeds directly to validation and startup.
OAuth validation authenticates to SMTP/IMAP and calls the read-only Drive `about` endpoint, but sends no message and
downloads no file. App Password mode validates SMTP/IMAP; it can consume only Drive files shared as
**Anyone with the link**.

### Civitai API token

Civitai's public API can describe a LoRA without a key, including adult/NSFW metadata, but the model file endpoint
requires authenticated downloads on this machine. The launcher therefore asks for a Civitai API token when one is
not already saved.

1. Double-click `Start 10MinVideoMaker.bat`.
2. Answer **yes** when asked to configure a Civitai token. The launcher opens
   `https://civitai.com/user/account`.
3. Sign in, open **Account Settings**, locate **API Keys**, and create a key.
4. Paste the key only into the local hidden prompt. Do not paste it into chat.
5. When the launcher reports the saved unfinished job, accept the default **yes** to retry it.

The token is used only for `civitai.com` model-download URLs. Public metadata is checked first to confirm the version
is a LoRA, select a virus-scanned SafeTensor, obtain its canonical filename and SHA-256 hash, and detect an existing
local copy. The downloader follows redirects, verifies the hash when supplied, and never logs the token. There is no
separate NSFW toggle in this project: adult assets remain governed by the signed-in Civitai account's own settings
and permissions.

To configure and validate without starting ComfyUI or the supervisor:

```powershell
python scripts\setup_and_start.py --setup-only
```

For an offline UI-only diagnostic that saves settings without contacting Gmail or Google Drive:

```powershell
python scripts\setup_and_start.py --setup-only --skip-gmail-check
```

The offline switch does not prove the credentials work.

## Gmail environment variables

The launcher manages these automatically. They can also be supplied directly in the environment that launches the
supervisor:

- `TENMIN_GMAIL_USERNAME`
- `TENMIN_GMAIL_RECIPIENT` (defaults to the username)
- `TENMIN_GMAIL_ALLOWED_SENDERS` (comma-separated; defaults to the username)
- `TENMIN_GMAIL_AUTH_MODE` (`app_password` or `oauth2`)
- `TENMIN_GMAIL_APP_PASSWORD` when using an App Password
- `TENMIN_GMAIL_OAUTH_CLIENT_ID`, `TENMIN_GMAIL_OAUTH_CLIENT_SECRET`, and
  `TENMIN_GMAIL_OAUTH_REFRESH_TOKEN` for persistent OAuth2
- `TENMIN_GMAIL_OAUTH_SCOPES` is a non-secret launcher marker written automatically after mail-plus-Drive consent
- `TENMIN_GMAIL_OAUTH2_TOKEN` remains supported only as a legacy short-lived access-token override
- `TENMIN_CIVITAI_TOKEN` for authenticated Civitai file downloads; the launcher stores it with DPAPI

The supervisor sends its request with the exact subject `Run the LTX video pipeline`. Grok must deliver the job as a
**new email**, not a reply, using the exact subject `LTX_JOB_COMPLETE`. The poller considers only unread messages
whose decoded subject is exactly `LTX_JOB_COMPLETE`; `Re: LTX_JOB_COMPLETE`, the outbound request subject, and
similar partial matches are ignored. Opening a completion email before the supervisor claims it marks it read in
Gmail, so mark that message unread again before starting the supervisor.

The small metadata envelope Grok places in the email may contain `job_id`, `drive_web_view_link`, and related
delivery fields without the full `scenes` array. That envelope is intentionally not accepted as the job itself. Its
Drive file link is followed and the downloaded full JSON must contain both `job_id` and `scenes`.

The supervisor retrieves matching messages with IMAP `BODY.PEEK[]`. Looking at several candidates during one poll
does not mark the unclaimed messages read; only an accepted job or a deliberately rejected malformed handoff is
marked handled. The live console reports how many exact-subject messages were found and whether a parsed job was
accepted, rejected, or skipped as a duplicate. A repeated email containing a `job_id` already present in the
durable job history is correctly skipped even if its attachment or Drive file is otherwise valid.

Payload precedence is:

1. First `.json` attachment.
2. Valid job JSON in the plain-text body.
3. A Google Drive **file** link in the plain-text or HTML body.

A malformed attachment is not silently replaced by body content. Drive folder links and non-Google URLs are ignored.
The downloaded file must be UTF-8 JSON, no larger than 5 MiB, and must pass the same `job_id`/`scenes` contract. Ask
Grok to share the file with the configured Gmail account; alternatively, **Anyone with the link** works without
Drive OAuth. Merely including a Drive view URL does not grant access. In Google Drive, open **Share** and either add
the configured Gmail address as a Viewer or set **General access** to **Anyone with the link — Viewer**. A private
file's anonymous sign-in redirect automatically falls back to the authenticated Drive API. If that API returns 404,
the file does not exist for—or has not been shared with—the authorized account; the completion email remains unread
and will retry after its sharing is corrected.

Strict JSON normally rejects integer literals with redundant leading zeroes. Because Grok may emit values such as
`"seed": 012345678`, the handoff parser removes leading zeroes only from unquoted `seed` and `original_seed`
integers, producing the equivalent value `12345678`. Prompt text is untouched, and no other malformed JSON is
silently repaired.

## Supervisor settings

Optional environment variables:

- `TENMIN_COMFY_URL` (default `http://127.0.0.1:8188`)
- `TENMIN_POLL_SECONDS` (default `300`)
- `TENMIN_T2I_TIMEOUT_SECONDS` (default `3600`)
- `TENMIN_I2V_TIMEOUT_SECONDS` (default `21600`)
- `TENMIN_MAX_STAGE_ATTEMPTS` (default `2`)
- `TENMIN_FFMPEG` and `TENMIN_FFPROBE` (default to commands on `PATH`)
- `TENMIN_LOG_LEVEL` (default `INFO`)
- `TENMIN_STORAGE_ROOT` (default and required drive:
  `D:\LTX_Supervisor_Storage`; alternate project-owned folders must still be on D:)
- `TENMIN_REQUIRE_HUMAN_REVIEW` (legacy console supervisor only; default `false`)

The GUI auto-starts incoming jobs even when a legacy review setting remains in the saved environment. Use
`Start 10MinVideoMaker.bat --hold-new-jobs-for-review` for a launch where new jobs must wait for explicit approval.

This machine currently exposes both FFmpeg and FFprobe on `PATH`.

The one-click launcher is the preferred start path. For diagnostics, the loop can still be started directly from the
repository root after configuration:

```powershell
python scripts\run_supervisor.py
```

The launcher first verifies that ComfyUI is healthy. If the configured URL is local and unavailable, it invokes the
project's path-verified restart helper. The first supervisor tick sends the initial request email, then IMAP is checked
every five minutes. Use `--once` for one durable state-machine step or `--no-restart` to disable controlled ComfyUI
restart while diagnosing configuration.

### Reading the live console

The supervisor prints a heartbeat every 15 seconds even while a render or asset request is blocking the main loop:

```text
STATUS | state=running_i2v | job=20260725-1234 | scene=3 | ComfyUI queue=running=1 pending=0
```

Common states are `waiting_for_grok`, `downloading_assets`, `running_t2i`, `running_i2v`, `stitching`, and `error`.
`running=1` means ComfyUI is actively executing a prompt; `running=0 pending=0` is normal during Gmail waits, asset
metadata/download work, FFmpeg, or between jobs. Separate lines announce the exact safe phase, including each LoRA
check/result and each scene attempt, so a zero GPU reading during non-render work is not mistaken for a hang.

Change **Console heartbeat seconds** in the launcher's optional settings, or set
`TENMIN_STATUS_INTERVAL_SECONDS`. The default is 15 seconds. `TENMIN_LOG_LEVEL=DEBUG` adds tick/cache details.
Status output never includes prompts, workflow bodies, OAuth/App Passwords, Civitai tokens, or Discord webhooks.

Do not start the supervisor merely to test installation: its first tick intentionally sends email, and a received job
can download LoRAs and begin generation. Use the no-render checks below instead.

## Recovery behavior

- Completed scenes are never regenerated.
- A transient failed stage retries up to `TENMIN_MAX_STAGE_ATTEMPTS`.
- A timed-out project prompt is removed from the pending queue or interrupted if it is the current running prompt.
- A scene-specific LoRA failure marks only that scene failed; other scenes continue and can be stitched.
- If an I2V routing defect invalidates completed clips, the project can requeue I2V for the whole saved job while
  retaining each deterministic cached T2I frame; invalid clips must be quarantined before resuming.
- If asset preparation fails for every scene, the supervisor pauses in `error`, prints each cause in the console,
  preserves the job, and does not request a replacement email.
- Relaunching offers to resume or abandon any active saved job, including one stopped during asset resolution,
  T2I, I2V, or stitching. Successful scenes and attempt counters remain intact. Answering **no** first cancels only
  queued/running prompts owned by the `10MinVideoMaker-supervisor` ComfyUI client, then marks unfinished scenes
  `cancelled`; the payload and diagnostic history remain in SQLite, and the pipeline returns to `idle`.
- A clip geometry or FFmpeg assembly failure pauses in `error`, preserves every completed clip, and prints the exact
  mismatch rather than retrying the stitch forever.
- A server availability failure records `error`, runs the path-verified restart script, and requeues unfinished scenes.
- VRAM/system cleanup runs after T2I, after each scene attempt, and after assembly.

## Safe manual checks

From the repository root:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q tenminvideomaker scripts __init__.py
python scripts\setup_and_start.py --help
python scripts\export_workflows.py --install-approved-shared-copies
python scripts\run_supervisor.py --help
git diff --check
```

The export command is no-render validation. It queries the running ComfyUI `/object_info` contracts but does not load
models, download assets, or generate media.
