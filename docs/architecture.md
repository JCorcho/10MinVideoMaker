# 10MinVideoMaker architecture

The automated pipeline has two layers that call the same pure-Python services:

- **Supervisor**: the 24/7 owner of Gmail polling, ComfyUI API job submission, scene retries, FFmpeg assembly, and controlled restart requests. It polls once every five minutes; it never blocks ComfyUI's execution queue by sleeping inside a node.
- **ComfyUI nodes**: interactive/status controls built on the same state, payload, mail, asset, and assembly services. They expose no independent routing or persistence rules.

The running ComfyUI 0.27.1 installation did not expose the package's initial V3 entrypoint through `/object_info`.
The package therefore registers its compatibility surface through `NODE_CLASS_MAPPINGS`. Business logic remains
outside the node classes, so this registration choice does not fork the automation implementation.

`runtime/pipeline.sqlite3` is the local durable state store. It is intentionally ignored by Git and records one global pipeline state plus per-scene states. A job is accepted only from `idle` or `waiting_for_grok`; completed scene artifacts remain intact when unfinished scenes are re-queued.

LoRA files are resolved independently and mapped to predictable safe `.safetensors` filenames. The manager first checks only ComfyUI-provided LoRA roots, then downloads a missing payload-provided HTTPS asset with redirect-following retries into the authorized LoRA destination. Each failed asset is reported independently so its scene can fail without cancelling the rest of the job. Mandatory DMD and JoyAI I2V LoRAs must already be installed because no trusted download URL was supplied for them.

Before stitching, FFmpeg preflight verifies every successful clip is 704×1248 at 24 fps. The concat operation uses stream copy and emits `D:\output\10minfinals\{job_id}_final.mp4`; the folder is created only when a completed job is actually assembled.

## Production profile

- Image/video size: 704×1248.
- Frame rate: 24 fps.
- LTX frame count: `8n + 1`, derived by rounding up to cover a scene's requested duration.
- Maximum LTX scene duration: 32 seconds.
- T2I: Anima and Pony each keep their verified reference sampler path.
- I2V: two LCM sampling passes, with separate verified sigma schedules and the LTX spatial upscaler.

The workflow templates will be rebuilt independently from live node contracts. The approved reference workflows are never written to or copied into this repository.

## No-render validation

- `python -m unittest discover -s tests -v`
- `python -m compileall -q tenminvideomaker __init__.py`
- `git diff --check`
- Restart ComfyUI, then query `/object_info/<node type>` for all seven `10MinVideoMaker_*` types.
- Queue `10MinVideoMaker_ReleaseMemory` alone as a harmless API smoke test. This verifies real execution without
  loading a model or generating media.
