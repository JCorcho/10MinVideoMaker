# 10MinVideoMaker user guide

## Current boundary

The project can validate incoming jobs, poll/send Gmail on demand, resolve LoRAs, build per-scene generation graphs,
cache the exact T2I frame, and validate/stitch completed clips. The unattended five-minute supervisor is the remaining
automation layer. No media has been rendered during implementation.

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

## Safe manual checks

From the repository root:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q tenminvideomaker scripts __init__.py
python scripts\export_workflows.py --install-approved-shared-copies
git diff --check
```

The export command is no-render validation. It queries the running ComfyUI `/object_info` contracts but does not load
models, download assets, or generate media.
