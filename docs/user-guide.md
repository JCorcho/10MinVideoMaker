# 10MinVideoMaker user guide

## Current boundary

The project can validate incoming jobs, poll/send Gmail, resolve LoRAs, build and queue per-scene generation graphs,
cache the exact T2I frame, download the matching I2V clip, validate/stitch completed clips, request the next job, and
recover unfinished scenes. Gmail has been authenticated and the first received job remains durably saved for retry.
No media has been rendered during implementation.

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

- Final image and video: 704×1248.
- Frame rate: 24 fps.
- Frame count: smallest `8n + 1` value covering the requested duration.
- Maximum scene length: 32 seconds.
- Anima T2I: 30 steps, CFG 4.5, `er_sde`, `beta57`.
- Pony T2I: 30 steps, CFG 6, `res_5s_ode` then `res_3m_ode`, `karras`.
- LTX I2V: LCM on both passes, verified distinct sigma lists, x2 tiled spatial upscaler, DMD 1.0, JoyAI 0.5.

The first LTX pass uses an internal 352×624 latent only because the mandated spatial model is x2. The decoded and
saved production clip is always 704×1248.

## One-click setup and start

Double-click `Start 10MinVideoMaker.bat` in the project root. Do not use the shared ComfyUI start scripts for this
project setup.

On the first run, the launcher detects missing Gmail settings and offers:

- **Google App Password**: opens Google's App Password page when requested, then securely prompts for the
  16-character value. The Google account must have 2-Step Verification enabled.
- **OAuth2 browser login**: opens the Google Cloud credentials page, asks for a **Desktop app** OAuth client ID and
  client secret, then prints and opens a Google authorization URL. After consent, Google redirects to a temporary
  loopback listener on this computer and the launcher stores the refresh token.

OAuth requests the full `https://mail.google.com/` scope because Gmail's SMTP/IMAP XOAUTH2 protocol requires it. For
uninterrupted operation, an external OAuth consent screen must be published to **In production**; refresh tokens from
an external project left in **Testing** expire after seven days. A personal app may still show Google's unverified-app
warning because this scope is restricted.

Secrets are not written to `.env`, workflow JSON, or Git. The launcher encrypts App Passwords, OAuth client secrets,
OAuth refresh tokens, and the Civitai API token with Windows DPAPI for the current Windows user and stores the
ciphertext in the ignored `runtime/secrets.json`. Non-secret values are stored in the ignored project `.env`.
Existing process environment variables override saved project values.

If all required values already exist, the launcher asks whether to change optional settings. Choosing yes displays
the editable values and a Gmail reconfiguration option. Choosing no proceeds directly to validation and startup.
Gmail validation authenticates to both SMTP and IMAP but sends no message.

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

For an offline UI-only diagnostic that saves settings without contacting Gmail:

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
- `TENMIN_GMAIL_OAUTH2_TOKEN` remains supported only as a legacy short-lived access-token override
- `TENMIN_CIVITAI_TOKEN` for authenticated Civitai file downloads; the launcher stores it with DPAPI

The exact request subject is `Run the LTX video pipeline`. A `.json` attachment takes precedence over the plain-text
body. A malformed attachment is not silently replaced by body content.

## Supervisor settings

Optional environment variables:

- `TENMIN_COMFY_URL` (default `http://127.0.0.1:8188`)
- `TENMIN_POLL_SECONDS` (default `300`)
- `TENMIN_T2I_TIMEOUT_SECONDS` (default `3600`)
- `TENMIN_I2V_TIMEOUT_SECONDS` (default `21600`)
- `TENMIN_MAX_STAGE_ATTEMPTS` (default `2`)
- `TENMIN_FFMPEG` and `TENMIN_FFPROBE` (default to commands on `PATH`)
- `TENMIN_LOG_LEVEL` (default `INFO`)

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

Do not start the supervisor merely to test installation: its first tick intentionally sends email, and a received job
can download LoRAs and begin generation. Use the no-render checks below instead.

## Recovery behavior

- Completed scenes are never regenerated.
- A transient failed stage retries up to `TENMIN_MAX_STAGE_ATTEMPTS`.
- A timed-out project prompt is removed from the pending queue or interrupted if it is the current running prompt.
- A scene-specific LoRA failure marks only that scene failed; other scenes continue and can be stitched.
- If asset preparation fails for every scene, the supervisor pauses in `error`, prints each cause in the console,
  preserves the job, and does not request a replacement email.
- Relaunching offers to retry unfinished scenes from the saved job; successful scenes and attempt counters remain
  intact.
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
