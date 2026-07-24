# 10MinVideoMaker architecture

The automated pipeline has two layers that call the same pure-Python services:

- **Supervisor**: the 24/7 owner of Gmail polling, ComfyUI API job submission, scene retries, FFmpeg assembly, and controlled restart requests. It polls once every five minutes; it never blocks ComfyUI's execution queue by sleeping inside a node.
- **ComfyUI nodes**: interactive/status controls built on the same state, payload, mail, asset, and assembly services. They expose no independent routing or persistence rules.

`runtime/pipeline.sqlite3` is the local durable state store. It is intentionally ignored by Git and records one global pipeline state plus per-scene states. A job is accepted only from `idle` or `waiting_for_grok`; completed scene artifacts remain intact when unfinished scenes are re-queued.

## Production profile

- Image/video size: 704×1248.
- Frame rate: 24 fps.
- LTX frame count: `8n + 1`, derived by rounding up to cover a scene's requested duration.
- Maximum LTX scene duration: 32 seconds.
- T2I: Anima and Pony each keep their verified reference sampler path.
- I2V: two LCM sampling passes, with separate verified sigma schedules and the LTX spatial upscaler.

The workflow templates will be rebuilt independently from live node contracts. The approved reference workflows are never written to or copied into this repository.
