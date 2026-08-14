# Production VLM QC Phase 1 Design

## Purpose

Add a bounded, opt-in production quality-control lane for paying-subscriber scene videos while the protected LTX 2.3 latent-overlap v3 work continues independently. Phase 1 uses the validated local Qwen3.6 27B vision configuration, one deterministic A1 retry, one constrained B1 prompt-repair retry, and a human canary approval gate. It does not train a model, introduce a multi-tier autonomous repair ladder, or change the LTX continuation algorithm.

The emergency branch is safe by default:

- `quality_control_enabled = false`
- `auto_advance_pass = false`

When QC is disabled, the existing T2I, I2V, delivery, partial-job, and assembly control flow remains the baseline. Enabling QC is explicit. Enabling automatic PASS advancement later is a configuration change, not a schema or code change.

## Repository and isolation boundary

This design applies only to `hotfix/production-vlm-qc-phase1`, based on production main commit `a7dabc39cc25c73a88aa9739219514416fa922d4`.

The following remain read-only inputs:

- Protected v3 worktree: `C:\AI\GitWorktrees\10MinVideoMaker-recovery-ltx23-latent-overlap-v3`
- Standalone lab: `C:\AI\LTX23-VLM-Video-QC-Lab`

The lab supplies validated behavior and extraction references, not a runtime dependency. Production code must not import from the lab, write into it, launch it, or store production results there. Model weights and the matching projector remain external configured assets; the production repository does not copy them.

## Current production architecture

### State and immutability

`tenminvideomaker/state_store.py` owns one SQLite database at `StorageLayout.database_path`, normally `D:\LTX_Supervisor_Storage\state\pipeline.sqlite3`.

Current durable concepts are:

- `jobs`: immutable source payload plus mutable job status/final path.
- `scenes`: the current scene pointer and automatic T2I/I2V attempt counts.
- `scene_revisions`: immutable parameter documents and versioned frame/video paths. Revision 1 is the original automatic result; GUI remakes create later revisions.
- `remake_batches` and `remake_items`: bounded GUI-requested revision generation.
- `continuation_plans`, `scene_chunks`, and `chunk_attempts`: immutable chunk plans, attempt lineage, seeds, checkpoint hashes, raw windows, and recovery state.
- `manual_final_requests`: an immutable snapshot of selected successful revisions for an explicit FFmpeg-only final.

Existing continuation attempts already demonstrate the right reliability pattern: allocate an identity, persist immutable inputs and artifacts, write completion evidence last, then select/promote the result. Phase 1 extends that pattern. It does not put QC evidence into a separate database or overwrite historical candidate artifacts.

### Generation and artifact reality

`PipelineSupervisor.process_job()` currently performs:

1. asset preparation;
2. `_process_t2i_batch()` for all scenes;
3. one `/free` model release;
4. `_process_i2v_batch()` for all scenes;
5. one `/free` model release;
6. `_deliver_i2v_batch()`;
7. profile validation and `FfmpegAssembler.stitch()`.

T2I uses Anima or Pony through `build_t2i_api_workflow()`. I2V uses either `build_i2v_api_workflow()` for a single window or `ContinuationRenderer.render_scene()` for bounded exact-frame continuation. `PipelineSupervisor.render_i2v_scene()` is the shared I2V service suitable for original, A1, and B1 candidates.

The continuation route has real persisted LTX pass boundaries per chunk:

- `stage1_handoff.safetensors`: bounded 384x672 LTX video latent;
- `stage2_video.safetensors`: native 768x1344 LTX video latent;
- `stage2_audio.safetensors`: second-pass audio latent;
- `window.mkv`: lossless FFV1/yuv444p raw window;
- revision-facing `video.mp4`: assembled 768x1344 scene video.

These are recovery boundaries inside one candidate render, not a job-wide low-quality/final-quality staging system. The legacy route also performs both LCM passes in one graph.

Both LTX passes load `10Eros_v1.4_fp8mixed_learned.safetensors`. Current main has no NVFP4 generation stage, no NVFP4-to-FP8 handoff, and no job-wide deferred FP8 final pass. Creating a decoded stage-one candidate, changing model precision between passes, or deferring all stage-two work would require workflow, renderer, persistence, recovery, and acceptance changes. It is not a Phase-1 dependency.

For clarity, this design uses **QC candidate** to mean the revision-facing `video.mp4` after the existing LTX two-pass render. It does not mean the internal `stage1_handoff` latent.

### Model and process lifecycle

The browser GUI, FastAPI routes, and `SupervisorController` run in one Python process. The controller owns one worker thread. ComfyUI is a separate Easy Install Python process on loopback port 8188. The supervisor owns only project prompts through the `10MinVideoMaker-supervisor` client ID.

ComfyUI controls model residency inside its process. `ComfyHttpClient.free_memory()` posts `{"unload_models": true, "free_memory": true}` to `/free`; the Python supervisor also calls `gc.collect()`. Current code intentionally keeps a model family resident across a batch and releases it only at a phase boundary.

A llama.cpp QC server can safely be a separate owned process only when:

- the project ComfyUI queue has no running or pending prompt;
- ComfyUI `/free` has completed;
- the configured physical QC GPU identity has been verified;
- the controller retains the exact child-process handle and dedicated loopback port;
- generation cannot resume until that child has exited and the port is closed.

Phase 1 never runs 5070 generation concurrently with 4080 QC. Correctness uses serialized epochs.

### Storage and test isolation

`StorageLayout` owns all production data under `D:\LTX_Supervisor_Storage`. Candidate media remains under the existing revision path:

```text
jobs/{job_id}/scenes/scene_{scene_id}/revisions/{revision}/
  frame.png
  video.mp4
  generation-manifest.json
  qc/
    evaluations/{evaluation_id}/
      result.json
      raw-response.txt
      frame-accounting.json
    repairs/{repair_id}.json
```

SQLite stores queryable identities, states, hashes, decisions, and manifest pointers. Immutable JSON/text evidence stores the full evaluator and planner records. Tests construct `StorageLayout` with temporary roots and mock FFmpeg, HTTP, process, GPU, and ComfyUI boundaries. No test may call `StorageLayout.configured().ensure()` or write to the real production root.

### UI and service boundaries

`tenminvideomaker/gui_app.py` serves a FastAPI API and the vanilla JavaScript files in `web/`. The existing scene detail already exposes revision history, candidate video URLs, and generation parameters. The smallest canary UI is an added QC panel tied to the selected revision; no new web framework or separate application is needed.

`server_api.py` is a loopback ComfyUI route only for live LoRA resolution. It is not a general local production API. The authoritative application boundary is direct Python service/controller integration, with narrow FastAPI routes only for displaying QC evidence and recording human decisions.

## Validated standalone lab findings

The reusable validated path is:

- Qwen3.6 27B Uncensored HauhauCS Balanced;
- `Qwen3.6-27B-Uncensored-HauhauCS-Balanced-IQ3_M.gguf`;
- matching FP16 `mmproj`;
- physical RTX 4080 SUPER;
- 2 FPS;
- four chronological frames per request;
- llama.cpp `--image-min-tokens 1024`;
- automatic early FAIL only after two strong evidence windows;
- strong evidence requires severity at least 3, error confidence at least 0.85, and non-empty concrete visible evidence.

The inspected local assets are:

```text
llama.cpp backend:
C:\Users\Elijah\.lmstudio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.28.2\llama-server.exe

CUDA vendor directory:
C:\Users\Elijah\.lmstudio\extensions\backends\vendor\win-llama-cuda12-vendor-v2

model (12,580,873,632 bytes):
C:\Users\Elijah\.lmstudio\models\HauhauCS\Qwen3.6-27B-Uncensored-HauhauCS-Balanced\Qwen3.6-27B-Uncensored-HauhauCS-Balanced-IQ3_M.gguf

projector (927,606,944 bytes):
C:\Users\Elijah\.lmstudio\models\HauhauCS\Qwen3.6-27B-Uncensored-HauhauCS-Balanced\mmproj-Qwen3.6-27B-Uncensored-HauhauCS-Balanced-f16.gguf
```

The lab launches loopback llama.cpp with context 16384, all GPU layers, one parallel slot, flash attention, Jinja chat templates, no Web UI, and image-min-tokens 1024. Production must rediscover and validate these configured paths rather than silently substituting a newer backend or another quantization.

The four-clip smoke set caught 3/3 known-bad and accepted 1/1 known-good. This is evidence for a canary, not a statistically sufficient autonomous rollout.

Rejected configurations remain rejected in Phase 1:

- 1 FPS missed the nuanced defect;
- six/eight-image windows reduced recall;
- Qwen3VL 8B falsely rejected the good clip;
- the 8B-to-27B cascade missed the nuanced bad clip;
- the 5070-only run was weaker on the nuanced clip;
- splitting the 27B model across GPUs was slower;
- image-token response was non-monotonic.

Reusable headless lab pieces and their production treatment are:

| Lab component | Production treatment |
| --- | --- |
| `Run-VLM-QC-Lab.ps1` llama.cpp discovery/launch arguments | Reimplement as an owned Python process manager with list arguments, hidden window, health polling, bounded shutdown, dedicated port, and immutable launch evidence. |
| `config.json` model/projector paths and validated settings | Move to `QualityControlSettings`; verify files, hashes, backend version, and physical GPU identity at epoch start. |
| `sampling.select_frame_indices()` 2-FPS logic | Port the deterministic algorithm. Use existing FFmpeg/FFprobe rather than adding PyAV/OpenCV as production dependencies. |
| `LMStudioRuntime` four-frame chronological windows | Extract the window behavior and timestamped OpenAI-compatible image request into a backend-neutral judge service. |
| `prompts/ltx_video_qc_v1.txt` | Add a versioned production prompt in this repository; persist version and SHA-256 on every evaluation. |
| `schema.parse_model_json()` | Port and strengthen it: strict top-level schema, decision enum, numeric bounds, timestamp bounds, and fail-closed malformed/refusal handling. |
| `has_confirmed_defect()` | Preserve the severity/confidence/evidence rule. |
| two-window early fail | Preserve exactly; never let a bare model FAIL trigger it. |
| lone-suspect shifted confirmation | Preserve: after full coverage with one strong suspect, evaluate one shifted overlapping four-frame window from scratch. One unconfirmed strong window normalizes to UNCERTAIN. |
| frame accounting | Preserve selected source indices/timestamps, window bounds, unique inspected frames, confirmation exposures, planned/processed windows, and early-exit reason. |
| `ResultsStore` evidence/history | Adapt to revision-local immutable manifests plus SQLite pointers; do not use the lab `results/` tree. |
| blind benchmark logic | Port only as a test/readiness helper. Ground truth never enters a production judge request. |
| `scripts/run_video_benchmark.py` | Reference for an offline benchmark command, not the production controller. |

The lab does not already provide a complete production-ready headless unit. `LMStudioRuntime` is usable library code, but the validated direct llama.cpp launch is PowerShell-owned, its result store is lab-specific, and the benchmark CLI assumes a server lifecycle outside the production state machine. Phase 1 extracts behavior; it does not import the lab package.

One pre-canary blocker must be resolved explicitly: the lab environment report lists the RTX 5070 Ti as physical index 0 and the RTX 4080 SUPER as physical index 1, while the launcher config records `CUDA_VISIBLE_DEVICES=0`. Production must bind by the verified 4080 UUID/name and confirm the resulting llama.cpp device from startup telemetry. It must not trust a mutable ordinal.

## Trust boundaries

### Vision judge

The vision judge is stateless and blind for every window request.

Model-visible input contains only:

- the versioned QC system/rubric prompt;
- four chronological candidate frames;
- their timestamps and window number.

It does not contain candidate tier, revision label, prior result, repair proposal, previous human expectation, job outcome, or benchmark ground truth. The request contains no tools, MCP definitions, URLs to project services, shell, SQL, or filesystem operations. Each HTTP chat request contains only its own system/user messages and uses a fresh server request context. The process manager uses one parallel slot, disables session/prompt-cache persistence, and verifies that completed-request KV state is cleared before the next request. Model weights may remain loaded; request KV/history may not.

The judge returns structured visual evidence only. It cannot choose retries or mutate project state.

### Repair planner

The repair planner is a separate, fresh, text-only chat request with its own versioned system prompt and cleared KV state. It may reuse the already-loaded Qwen weights during the same QC epoch, but it receives no images and no vision-judge role prompt.

Its model-visible input is limited to:

- immutable scene/job facts already available in the current production contract;
- original and current I2V prompts;
- the locked negative/safety prompt;
- current seed and fixed generation configuration;
- normalized QC JSON;
- previous repair summaries;
- explicit mutable fields (`i2v.prompt`, `i2v.seed`);
- explicit locked fields.

The request carries no tools. Output is one constrained JSON patch. The planner cannot write a file, call an API, execute code, grade a regenerated video, or promote a candidate.

The B1 candidate is always evaluated later by a new blind vision-judge request. The planner never sees that evaluation in the context that created the candidate.

### Deterministic controller

The Python controller is the only authority that:

- validates both schemas;
- derives retry seeds;
- enforces one A1 and one B1 maximum;
- deep-compares locked fields;
- rejects stale or duplicate proposals;
- creates normal scene revisions;
- calls the existing I2V render service;
- persists immutable evidence;
- promotes an accepted revision;
- decides whether final delivery/assembly can proceed.

Model strings are data. They are never interpolated into shell commands, SQL, paths, Python, workflow node types, or arbitrary configuration. SQLite uses bound parameters. Filesystem paths are derived only by `StorageLayout`. Subprocesses receive fixed argument lists.

## Configuration

Add `QualityControlSettings` in `tenminvideomaker/qc_config.py` with a strict `from_environment()` parser. The public policy fields are exactly:

```text
quality_control_enabled = false
auto_advance_pass = false
```

Environment keys are `TENMIN_QUALITY_CONTROL_ENABLED` and `TENMIN_QC_AUTO_ADVANCE_PASS`. Other settings cover the dedicated loopback port, llama.cpp executable/vendor root, model/projector paths, expected GPU UUID/name, context length, startup/request/shutdown timeouts, 2 FPS, four frames, image-min-tokens 1024, prompt versions, severity 3, confidence 0.85, and two strong windows.

Every setting is validated before QC starts. Unsafe threshold/window changes are not exposed in the canary UI. The effective settings document and its SHA-256 are persisted per evaluation.

`auto_advance_pass=true` changes only the PASS transition. It does not affect FAIL, UNCERTAIN, evidence thresholds, retry budgets, or human decisions.

## Persistent QC model

### Candidate identity

Each candidate is one existing `scene_revisions` row plus a `qc_candidates` row. The revision owns the frame, video, parameters, and generation manifest. The QC row adds:

- stable `candidate_id`;
- job, scene, and revision;
- tier `ORIGINAL`, `A1`, or `B1`;
- parent candidate;
- source video path and SHA-256;
- original/current prompt and seed;
- negative/safety prompt value and identity hash;
- candidate state and next action;
- bounded infrastructure-failure count and last failure evidence;
- created/updated timestamps.

Uniqueness on `(job_id, scene_id, tier)` enforces at most one candidate per tier. Original maps to revision 1. A1 and B1 are normal `VIDEO_ONLY` revisions with the original accepted T2I frame and full validated parameter documents.

### Evaluations

`qc_evaluations` is append-only and records:

- evaluation ID and candidate ID;
- source artifact path/hash;
- evaluator/backend/model/projector/executable identities and hashes;
- GPU identity;
- effective launch/request config and prompt version/hash;
- sampling/window config;
- raw model results;
- normalized PASS/FAIL/UNCERTAIN;
- suspect times/windows and strong-window count;
- frame accounting;
- evidence manifest path/hash;
- next action and timestamps.

An idempotency key derived from candidate video hash plus evaluator/config/prompt hashes prevents duplicate completed evaluations after restart.

### Repairs

`qc_repairs` is append-only and records planner backend/version, source candidate/evaluation, canonical repair input hash, raw output, proposed patch, accepted/rejected status and reason, prior repair summaries, and timestamps. A proposal is accepted only if:

- it was built from the current candidate/evaluation hashes;
- it contains exactly `i2v.prompt`, `i2v.seed`, and a summary field;
- its prompt is non-empty and differs from the current prompt;
- its seed equals the controller-derived B1 seed and differs from all prior candidate seeds;
- a deep comparison proves every field except I2V prompt and seed is unchanged;
- `validate_scene_edit()` accepts the resulting full document.

Rejected proposals are evidence and do not consume another planner loop. The scene enters `HOLD_FOR_REVIEW`.

### Human decisions

`qc_human_decisions` is append-only and records candidate, local action `APPROVE`, `REJECT`, or `HOLD`, optional note, actor label, result/evidence hashes, and timestamp. A terminal human decision cannot be overwritten by a late browser request.

Only `APPROVE` promotes the candidate. `REJECT` and `HOLD` both enter `HOLD_FOR_REVIEW`; neither creates another automatic tier.

## Candidate and job states

Candidate states are:

- `PENDING_GENERATION`
- `GENERATING`
- `PENDING_QC`
- `QC_RUNNING`
- `PASS_PENDING_HUMAN`
- `ACCEPTED`
- `HOLD_FOR_REVIEW`
- `SUPERSEDED`

The normalized evaluator decision remains separately stored as PASS, FAIL, or UNCERTAIN. FAIL is not itself a durable candidate state because the deterministic next action is immediately persisted: schedule the next allowed tier or hold.

Add job-level pipeline states `RUNNING_QC` and `AWAITING_QC_REVIEW`. Existing `RUNNING_I2V`, delivery, and `STITCHING` semantics remain.

The exact transition table is:

| Current tier/result | Next action |
| --- | --- |
| ORIGINAL PASS, `auto_advance_pass=false` | `PASS_PENDING_HUMAN`; job waits in `AWAITING_QC_REVIEW`. |
| ORIGINAL PASS, `auto_advance_pass=true` | `ACCEPTED`. |
| ORIGINAL FAIL | Create exactly one A1 `PENDING_GENERATION`. |
| ORIGINAL UNCERTAIN | `HOLD_FOR_REVIEW`. |
| A1 PASS, canary | `PASS_PENDING_HUMAN`. |
| A1 PASS, auto-advance | `ACCEPTED`. |
| A1 FAIL | Run one text-only B1 repair plan while Qwen remains loaded when practical; validated B1 becomes `PENDING_GENERATION`. |
| A1 UNCERTAIN | `HOLD_FOR_REVIEW`. |
| B1 PASS, canary | `PASS_PENDING_HUMAN`. |
| B1 PASS, auto-advance | `ACCEPTED`. |
| B1 FAIL or UNCERTAIN | `HOLD_FOR_REVIEW`. |
| Human APPROVE | `ACCEPTED`; promote that revision. |
| Human REJECT or HOLD | `HOLD_FOR_REVIEW`; no new automatic tier. |

Final delivery and assembly require exactly one accepted candidate with a valid video for every scene. QC-enabled assembly never omits a held or missing scene. This prevents silent scene deletion and partial “success” caused by the QC lane. Existing partial-job behavior remains unchanged when the kill switch is off.

## A1 and B1 mutation rules

### A1

A1 is exactly one deterministic video-only retry:

- reuse the same T2I frame;
- reuse the exact I2V prompt;
- preserve negative/safety prompt and all fixed scene/generation fields;
- derive a new unsigned-64-bit seed from job ID, scene ID, source revision, original seed, and literal tier `A1` using SHA-256;
- require the derived seed to differ from original, retrying the derivation with a fixed counter only on the theoretical collision;
- persist the original/current prompt and both seeds;
- render through `validate_scene_edit()` plus `PipelineSupervisor.render_i2v_scene()`.

### B1

B1 is exactly one constrained video-only retry:

- reuse the same T2I frame;
- invoke the fresh text-only repair planner after A1 FAIL;
- compute the new B1 seed deterministically before accepting a patch;
- permit only the I2V prompt and seed to differ;
- preserve the negative/safety prompt, LoRAs, continuation, segments, duration, T2I data, model settings, samplers, sigmas, spatial upscaler, character/job facts, and production profile;
- reject stale, identical, repeated, or schema-invalid proposals;
- persist original, A1/current, and repaired prompts plus proposal status;
- judge the regenerated B1 video blindly in a later fresh vision context.

After a B1 candidate is not accepted, the scene holds. There is no loop, C/D tier, scene deletion, or whole-job abandonment.

## Serialized epoch integration

Phase 1 uses these coarse epochs:

1. **Original generation epoch:** run the current T2I batch and current full two-pass I2V batch; persist all original revision-facing videos; call `/free`.
2. **QC epoch:** verify empty ComfyUI queue; launch the owned Qwen llama.cpp process on the verified 4080; evaluate every `PENDING_QC` candidate; perform B1 text-only planning for A1 failures while weights remain loaded when practical; persist results; terminate Qwen and prove exit.
3. **Repair generation epoch:** render all `PENDING_GENERATION` A1/B1 revisions through existing production I2V paths; call `/free` after the batch.
4. Repeat QC and repair-generation epochs until no automatic action remains.
5. If all scenes are accepted, run the existing separate Discord delivery and final assembly. Otherwise enter `AWAITING_QC_REVIEW`.

The controller advances from durable database state, not a memory-only loop. A process restart resumes the earliest incomplete action. A completed immutable evaluation or candidate artifact is reused only when all identity hashes match.

If the llama worker crashes, the controller terminates/reaps the owned process, leaves the candidate intact, and retries the same pending evaluation once in a new QC epoch. A second infrastructure failure holds the candidate with explicit error evidence. It never converts a crash into PASS or drops the candidate. Generation-worker failure continues to use existing prompt/checkpoint recovery and bounded retry behavior.

## Kill switch behavior

With `quality_control_enabled=false`:

- new jobs follow the exact pre-feature `process_job()` call order;
- no llama process starts;
- no QC candidate is required for delivery or assembly;
- existing retry counts and partial-job behavior are unchanged.

If the switch is turned off while a job is held in Phase-1 QC, the controller preserves all QC evidence, selects the original revision-1 video as the pre-feature result, and resumes the existing delivery/assembly path. It never silently promotes an unapproved A1/B1 revision. This is the explicit rollback behavior and is covered by an integration test.

## Human canary UI

Extend the existing selected-scene view with:

- current candidate/revision and tier;
- the existing HTML5 candidate video;
- normalized VLM decision, confidence, summary, and evaluator/config versions;
- suspect windows with timestamps, severity, confidence, description, and visible evidence;
- process/parse errors when present;
- `Approve`, `Reject`, and `Hold` buttons.

Add one read route for scene QC history and one job/scene/candidate-scoped decision route. The route validates identity and current state, writes the immutable human decision, promotes only on approval, wakes `SupervisorController`, and returns the new state. The browser never sends artifact paths, prompts, seeds, or arbitrary patches in a decision request.

## MCP and API decision

Choose **A: direct service-function/controller integration**.

The controller and evaluator are on the same host and already share a single-owner state machine. Direct typed Python calls minimize failure modes and make schema validation, budgets, process ownership, and transactions testable. The existing loopback FastAPI API remains appropriate for the human GUI only. The llama.cpp OpenAI-compatible endpoint is an internal backend transport, not a project automation API.

Do not add MCP in Phase 1. A narrow loopback MCP server would duplicate authentication, lifecycle, serialization, and tool-surface work without enabling a required external client. No shell, SQL, or filesystem MCP tools are permitted. A future backend can implement the same `QcBackend` interface without changing the controller.

## Stage-one and stage-two residual-risk audit

Internal LTX stage one contains broad composition and motion information, so many gross defects would likely be visible after an added decode. It is not the final artifact and current main does not persist a stage-one video.

Stage two can introduce defects absent from stage one. It spatially upsamples and resamples the video, changes fine anatomy/identity/detail, produces the final audio/video latent, and is the source of the 768x1344 raw windows. Repository history explicitly records prior stage-two identity/style drift and a later correction to use native full-resolution stage-two output.

Therefore a VLM gate on only an added stage-one preview would leave material residual risk. It would require at least a lightweight post-stage-two human canary review. A second automated post-stage-two VLM pass would also require another serialized Qwen epoch after final generation and would increase model-lifecycle cost; that is impractical for this emergency scope unless later evidence justifies it.

Phase 1 avoids that gap by judging the existing post-stage-two revision-facing scene video. The default human approval is consequently also a post-stage-two canary review. There is no second automated VLM pass because the one automated pass already sees the final per-scene pixels. Final Discord watermarking/encoding and project concat may introduce transport/encode issues, but not new generative anatomy or motion; existing profile validation remains responsible for those outputs.

## Baseline and acceptance boundary

The established CPU-only baseline is:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s tests -v
```

At base commit `a7dabc39cc25c73a88aa9739219514416fa922d4`, it ran 355 tests in 9.599 seconds: PASS with 12 skipped. Test output includes mocked setup/ComfyUI messages; no ComfyUI or supervisor process was started. A preliminary run with an artificial `TENMIN_STORAGE_ROOT` override produced one expected path-string assertion failure, created no directory, and was discarded; the established command above is clean.

Implementation acceptance requires focused tests plus the full CPU suite without production writes. A real canary is separate and bounded: one explicitly selected safe scene, empty ComfyUI queue, verified 4080 identity, owned llama process, QC enabled and auto-advance disabled, evidence review, explicit human decision, and post-run proof that both ComfyUI and llama resources were released.

## Excluded work

Phase 1 does not include temporal-model training, SigLIP, VideoMAE, V-JEPA, Tier C/D, a full A-D ladder, Grok repair, Hivemind orchestration, final autonomous whole-video QC, self-improving prompts, simultaneous dual-GPU generation/QC, NVFP4-to-FP8 staging, or changes to protected v3.
