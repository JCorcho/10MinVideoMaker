# 10MinVideoMaker user guide

## Current boundary

The project can validate incoming jobs, poll/send Gmail, resolve LoRAs, build and queue per-scene generation graphs,
cache the exact T2I frame, download the matching I2V clip, validate/stitch completed clips, request the next job, and
recover unfinished scenes. The supervisor is implemented but has not been started because Gmail credentials are not
configured. No media has been rendered during implementation.

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

## Gmail credentials

Credentials are never stored in workflow JSON or the repository. Set them in the environment that launches the
supervisor or ComfyUI:

- `TENMIN_GMAIL_USERNAME`
- `TENMIN_GMAIL_RECIPIENT` (defaults to the username)
- `TENMIN_GMAIL_ALLOWED_SENDERS` (comma-separated; defaults to the username)
- `TENMIN_GMAIL_AUTH_MODE` (`app_password` or `oauth2`)
- `TENMIN_GMAIL_APP_PASSWORD` when using an App Password
- `TENMIN_GMAIL_OAUTH2_TOKEN` when using OAuth2

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

After setting Gmail credentials, start the loop from the repository root:

```powershell
python scripts\run_supervisor.py
```

The first tick sends the initial request email, then IMAP is checked every five minutes. Use `--once` for one durable
state-machine step or `--no-restart` to disable controlled ComfyUI restart while diagnosing configuration.

Do not start the supervisor merely to test installation: its first tick intentionally sends email, and a received job
can download LoRAs and begin generation. Use the no-render checks below instead.

## Recovery behavior

- Completed scenes are never regenerated.
- A transient failed stage retries up to `TENMIN_MAX_STAGE_ATTEMPTS`.
- A timed-out project prompt is removed from the pending queue or interrupted if it is the current running prompt.
- A scene-specific LoRA failure marks only that scene failed; other scenes continue and can be stitched.
- A server availability failure records `error`, runs the path-verified restart script, and requeues unfinished scenes.
- VRAM/system cleanup runs after T2I, after each scene attempt, and after assembly.

## Safe manual checks

From the repository root:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q tenminvideomaker scripts __init__.py
python scripts\export_workflows.py --install-approved-shared-copies
python scripts\run_supervisor.py --help
git diff --check
```

The export command is no-render validation. It queries the running ComfyUI `/object_info` contracts but does not load
models, download assets, or generate media.
