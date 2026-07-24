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

LoRA files are resolved independently and mapped to predictable safe `.safetensors` filenames. The manager first checks only ComfyUI-provided LoRA roots, then downloads a missing payload-provided HTTPS asset with redirect-following retries into the authorized LoRA destination. Each failed asset is reported independently so its scene can fail without cancelling the rest of the job. Mandatory DMD and JoyAI I2V LoRAs must already be installed because no trusted download URL was supplied for them.

Before stitching, FFmpeg preflight verifies every successful clip is 704×1248 at 24 fps. The concat operation uses stream copy and emits `D:\output\10minfinals\{job_id}_final.mp4`; the folder is created only when a completed job is actually assembled.

VHS writes scene video to its temporary ComfyUI output and returns metadata through prompt history. The supervisor
downloads that exact output through the local HTTP API into
`D:\output\10minfinals\.work\{job_id}\clips\scene_{id}.mp4`; it does not scan or move unrelated shared output files.

The controlled Windows restart script resolves the expected Easy Install paths, verifies that the process listening
on port 8188 is the expected embedded Python executable, stops only that process, launches the unchanged
`Start ComfyUI.bat` hidden, and waits for HTTP health. It is called only for fatal ComfyUI availability failures.

## Production profile

- Image/video size: 704×1248.
- Frame rate: 24 fps.
- LTX frame count: `8n + 1`, derived by rounding up to cover a scene's requested duration.
- Maximum LTX scene duration: 32 seconds.
- T2I: Anima and Pony each keep their verified reference sampler path.
- I2V: two LCM sampling passes, with separate verified sigma schedules and the LTX spatial upscaler.

The workflow templates will be rebuilt independently from live node contracts. The approved reference workflows are never written to or copied into this repository.

Scene workflows are built dynamically from the validated job rather than mutating user-owned workflow JSON. The T2I
builder selects Anima or Pony from `character.lora.base`, applies the character LoRA once, adds any scene LoRAs, and
uses the exact family sampler route. The I2V builder consumes the deterministic PNG produced by the matching T2I
scene, adds DMD and JoyAI before dynamic model-only LoRAs, enables feed-forward chunking, and uses separate LCM
samplers and sigma schedules around the tiled spatial upscaler.

The x2 spatial upscaler requires a half-resolution first-pass latent. Its internal dimensions are 352×624 so the
second pass lands exactly at the only production output size, 704×1248. No alternate output resolution is exposed.

## No-render validation

- `python -m unittest discover -s tests -v`
- `python -m compileall -q tenminvideomaker __init__.py`
- `git diff --check`
- Restart ComfyUI, then query `/object_info/<node type>` for all seven `10MinVideoMaker_*` types.
- Queue `10MinVideoMaker_ReleaseMemory` alone as a harmless API smoke test. This verifies real execution without
  loading a model or generating media.
- Run `python scripts/export_workflows.py --install-approved-shared-copies` while ComfyUI is healthy. Export refuses
  unavailable classes or mismatched routes, lays out nodes by dependency depth, checks node overlaps and group bounds,
  and writes both API and GUI forms.
