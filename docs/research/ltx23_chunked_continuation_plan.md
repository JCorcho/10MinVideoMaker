# 10MinVideoMaker: Chunked LTX-2.3 Continuation Architecture

**Status:** Implementation design
**Target:** Local ComfyUI on Windows, RTX 4080 SUPER 16 GB, 32 GB RAM
**Primary objective:** Improve temporal anatomical stability and shot continuity for long, adult-only scenes without repeatedly loading a T2I model
**Recommended strategy:** Rolling first-pass latent-overlap continuation using the official Lightricks `LTXVExtendSampler`, followed by overlap-aware second-pass spatial refinement and deterministic scene assembly

> **Runtime correction, 2026-07-31:** Preserve production later-window 25-frame final-resolution overlap. Initial
> refinement diagnostic uses only a 17-frame decoded guide (frames 96–112). Live LTX testing proved initial
> refined latent has 20 guide-token positions: a 25-frame `8n+1` guide encodes 21 and is rejected; 17 frames
> encode 20 and completed. This supersedes only initial diagnostic described below.

---

## 1. Executive recommendation

Implement long scenes as a sequence of **rolling LTX first-pass windows**, using:

- A nominal **121-frame generation window**
- A **24-transition-frame overlap**, equivalent to 25 stored image samples
- **96 new transitions per full continuation**
- The previous chunk’s **first-pass denoised video latent tail** as the primary continuation state
- A bounded rolling latent, not the complete previous graph or an ever-growing GPU-resident latent
- A one-overlap-window commit delay so an overlap can be fused before its frames become immutable
- Final-resolution overlap guides during the second pass to protect visible identity, clothing, lighting, and seam appearance
- Hard cuts at validated points inside the overlap, rather than ordinary pixel-space crossfades over human anatomy
- Deterministic, derived seeds and explicit approximately five-second prompts supplied by Grok

Do **not** use the literal final decoded frame as the normal continuation mechanism. A single frame preserves appearance and pose at one instant but contains no explicit velocity or recent motion trajectory. It is therefore prone to visible motion restarts.

Do **not** send every boundary through Anima, Pony, or another T2I/I2I model. That would add model-loading overhead and introduce a second model’s interpretation of identity, pose, anatomy, lighting, and background. Use T2I boundary repair only as an exceptional manual remediation path.

The resulting scene pipeline is:

1. Generate all T2I scene-start images.
2. Release the T2I model completely.
3. Load and patch the LTX model once.
4. Process one scene at a time.
5. Generate that scene’s first-pass continuation chunks sequentially.
6. Spatially upscale and refine stabilized chunk windows.
7. Validate and assemble the chunks into the scene’s ordinary raw video artifact.
8. Move to the next scene while LTX remains resident.
9. Release LTX after the entire job.
10. Keep watermarking exclusively in the Discord delivery branch.

### Audio recommendation

The official local `LTXVLoopingSampler` explicitly rejects combined audio-video latents. The official hosted LTX extension API can extend motion and audio together, but that capability is not presently exposed as an equivalent documented local ComfyUI node. citeturn504112view2turn504112view5

Therefore:

- Treat video continuation and audio seam management as separate orchestration concerns.
- Preserve each chunk’s generated audio when the existing graph provides it.
- Trim duplicated or overlapped audio according to the committed video range.
- Apply short, equal-power audio crossfades at suitable ambience boundaries.
- Do not place a chunk boundary through dialogue, a syllable, an impact transient, or a one-shot sound.
- For dialogue-critical scenes, use a scene-level audio track or move boundaries into silence rather than attempting to blend two independently generated utterances.

---

## 2. Primary-source findings with links

### 2.1 LTX-2.3 constraints and production pipeline

**Documented:** LTX-2.3 requires width and height divisible by 32 and frame counts following `8n + 1`. The official API documentation explicitly lists 121 as a valid frame count and describes it as approximately five seconds at 24 fps. citeturn504112view4turn377877search0

**Documented:** Lightricks recommends its two-stage text/image-to-video pipeline for production-quality output. The current official repository lists the LTX-2.3 x2 spatial upscaler as required by its current two-stage implementations. citeturn206669view0

**Documented:** The official Lightricks ComfyUI two-stage workflow currently uses the same first-pass sigma list supplied in the question:

`1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0`

Its published second pass currently begins at `0.85`, whereas this system uses `0.909375`. That is a real workflow difference, not a transcription error. citeturn613545view2turn613545view3

**Decision:** Preserve the system’s current second-pass schedule in architecture version 1. Do not silently change it while also changing continuation behavior. Record the official `0.85` schedule as a separately testable tuning option after the continuation architecture is stable.

### 2.2 Is 121 frames an official quality optimum?

**Documented:** 121 frames is an officially supported frame count and an official approximately-five-second example.

**Not documented:** I found no Lightricks statement that 121 is a uniquely optimal quality point for LTX-2.3 or that quality necessarily deteriorates immediately above 121. The official duration table also lists 161 and 257 frames. citeturn504112view4

**Engineering inference:** 121 is a sound operational window because it:

- Satisfies `8n + 1`
- Corresponds to approximately five seconds
- Matches official examples
- Keeps temporal attention and VRAM bounded
- Encourages prompts to describe one coherent action beat
- Limits the time over which anatomical and identity errors can compound

It should be described internally as the **default continuation window**, not as a model-guaranteed quality optimum.

### 2.3 Official conditioning and continuation mechanisms

LTX-2.3 and its current ecosystem support the following:

| Mechanism | Official support | Relevant implementation |
|---|---|---|
| One starting image | Yes | Native ComfyUI I2V and `LTXVImgToVideoConditionOnly` |
| Multiple image/video frames as a guide | Yes | Core `LTXVAddGuide` |
| Existing latent sequence as context | Yes, through Lightricks custom nodes | `LTXVExtendSampler`, `LTXVAddLatentGuide` |
| First and last keyframes | Yes | Native FLF2V and `KeyframeInterpolationPipeline` |
| Multiple keyframes | Yes | `LTXVAddGuide`; keyframe interpolation |
| Video-to-video control | Yes | IC-LoRA pipelines |
| Retaking part of an existing video | Yes | `RetakePipeline` |
| Official video extension with motion and audio context | Yes, hosted API | `/v1/extend` |
| Equivalent documented local AV extension node | No | Local looping node rejects AV latents |

ComfyUI’s core `LTXVAddGuide` accepts either an image or a video. Multi-frame guides must be `8n + 1`; longer guide inputs are cropped to that pattern. For guides of nine or more frames, the starting frame index is aligned to a multiple of eight. citeturn613545view0

The Lightricks `LTXVImgToVideoConditionOnly` encodes one or more supplied image frames, writes the encoded result into the first temporal latent positions, and applies a noise mask. This is strong prefix replacement rather than merely a reference hint. citeturn613545view1

### 2.4 Official local extension nodes

**Documented:** `LTXVExtendSampler`:

- Accepts a video latent to extend
- Selects the ending overlap from that latent
- Creates a new overlap-plus-new-frames latent
- Adds the previous tail as a latent guide
- Samples the new temporal window
- Accounts for LTX’s asymmetric first temporal latent
- Blends the generated overlap with the previous latent
- Concatenates the result to the supplied latent
- Accepts an arbitrary ComfyUI sampler and sigma sequence
- Requires a `STGGuiderAdvanced`
- Supports overlaps from 16 to 128 frames in steps of eight
- Defaults to a 16-frame overlap and conditioning strength 0.5 citeturn504112view0turn504112view1turn134011view0

The source performs latent-space overlap fusion with `LinearOverlapLatentTransition`; this is materially different from decoding a frame and starting a fresh I2V generation. citeturn134011view0turn134011view1

**Documented:** `LTXVLoopingSampler` automates temporal tiling, per-tile prompts, long-term reference latents, overlap conditioning, and optional latent normalization. Its documentation says each subsequent tile adds `temporal_tile_size - temporal_overlap` new frames. It also exposes AdaIN-based normalization intended to mitigate accumulated oversaturation. citeturn324864search0turn134011view2

**Important limitation:** `LTXVLoopingSampler` explicitly throws an error when passed combined audio-video latents. citeturn504112view2

### 2.5 Official hosted extension

Lightricks’ hosted `/v1/extend` endpoint uses context frames from an input video to continue motion and audio. It accepts 1–20 seconds of context, and context plus generated duration may total no more than 505 frames. This is strong evidence that multi-frame temporal context is the intended continuation mechanism. It is not evidence that the same hidden context or audio behavior is exposed by the local open-source ComfyUI nodes. citeturn504112view5

### 2.6 JoyAI Echo and the DMD adapter

**Documented:** The official JoyAI Echo release is a standalone approximately 46 GB LTX-2.3-based model and inference system. Its long-video consistency depends on a paired cross-modal memory bank and its own inference code. It is not officially documented as a conventional ComfyUI LoRA applied at strength 0.5. citeturn377877search1turn377877search3

**Community source:** The TenStrip `LTX2.3_DMD_Lora` model card describes that adapter as an extraction of DMD deltas from JoyAI Echo and recommends strength 1.0 for ordinary 8/4-step LTX sampling. citeturn377877search2

**Required check:** The exact local artifact called “JoyAI Echo” must be identified by:

- Source repository
- Exact filename
- SHA-256 or BLAKE3 hash
- Whether it is a full-model merge, transformer patch, or LoRA
- Rank and target modules
- Whether it contains DMD-derived deltas already
- License
- Its declared compatibility with LTX-2.3 and the TenStrip DMD extraction

Stacking two adapters derived from the same JoyAI checkpoint could double-count some learned deltas. That cannot be determined safely from the names alone.

No community anecdotes are used to justify the central continuation architecture.

---

## 3. Verified 121-frame and 24-fps timing math

### 3.1 One 121-frame clip

At 24 fps:

- Stored-frame duration: `121 / 24 = 5.0416667 seconds`
- Timestamp from frame 1 to frame 121: `(121 - 1) / 24 = 5.0000000 seconds`

Both statements are correct. They answer different questions.

A conventional CFR stream assigns each stored frame a duration of `1/24` second. Therefore its muxed stream duration is normally 121/24 seconds even though the first and last presentation timestamps are exactly five seconds apart.

### 3.2 Six chunks with one shared boundary frame

For six 121-frame chunks where each later chunk shares exactly one identical first/last boundary frame and that duplicate is removed:

`unique_frames = 121 + 5 × 120 = 721`

Equivalent form:

`unique_frames = 1 + 6 × 120 = 721`

Therefore:

- First-to-last-frame span: `(721 - 1) / 24 = 30.0000 seconds`
- Ordinary CFR stream duration: `721 / 24 = 30.0416667 seconds`

The proposed equation is mathematically correct.

### 3.3 Exact 30-second timeline output

A conventional exactly 30.000-second, 24-fps scene contains:

`30 × 24 = 720 stored frames`

Once generation is finished, the assembled scene does not need to retain `8n + 1`; that restriction applies to LTX generation inputs.

Recommended distinction:

- `generation_master_frames`: LTX-compatible frame count, such as 721
- `timeline_output_frames`: ordinary production frame count, such as 720

For a requested 30.000-second scene:

1. Generate and assemble 721 frames.
2. Retain all 721 in the generation master or frame archive.
3. Trim the final timeline artifact to 720 frames.
4. Use the 720-frame artifact as the ordinary `scene_raw.mp4`.

This prevents one-frame-per-scene duration accumulation in a long project.

### 3.4 Frame math with the recommended 24-frame overlap

Use:

- Window transitions `W = 120`
- Stored frames per full window `W + 1 = 121`
- Overlap transitions `O = 24`
- Overlap stored samples `O + 1 = 25`
- New transitions per full continuation `S = W - O = 96`

For a 30-second generation master:

- Target transitions: 720
- First window contributes: 120
- Remaining: 600
- Six full continuations contribute: `6 × 96 = 576`
- Final continuation contributes: 24
- Total generation windows: 8

Thus a 24-frame-overlap design does **not** generate a 30-second shot with six 121-frame model runs. It uses eight generation windows, although the final window is much shorter.

That extra cost buys approximately one second of recent motion context at every handoff.

---

## 4. Comparison of continuation methods

### 4.1 Continuity and quality

| Method | Identity | Anatomy | Pose | Velocity and motion | Camera continuity | Drift |
|---|---|---|---|---|---|---|
| **A. Exact final decoded frame** | Medium initially | Medium/low over many chunks | Strong at seam instant | **Weak**; velocity is absent | Weak/medium | High cumulative VAE/pixel drift |
| **B. Several ending frames or latent overlap** | **High** | **Best available local option** | High | **High** | **High** | Medium; controllable with anchors and overlap |
| **C. Independently generated overlapping clips** | Medium/high | Medium | Medium/high | Medium | Medium | Medium |
| **D. T2I/I2I boundary repair** | Medium | Potentially improves one frame, but can reinterpret anatomy | Medium/low | Weak | Low/medium | High identity-style drift risk |
| **E. First plus target/end frame** | High when a valid target exists | High at endpoints | High endpoints | Medium; motion is interpolation-constrained | Medium/high | Low between known endpoints |
| **F. Official extension mechanism** | Hosted API: high; local latent extension: high | High | High | High | High | Lowest supported design risk |

### 4.2 Operational characteristics

| Method | VRAM | Runtime | Complexity | Crash recovery | T2I reload | Suitability |
|---|---:|---:|---:|---:|---:|---|
| A | Low | Low | Low | Excellent | No | Emergency fallback |
| B, decoded multi-frame guide | Moderate | Moderate | Moderate | Excellent | No | Good AV-compatible fallback |
| B, first-pass latent overlap | Moderate | Moderate/high | Moderate/high | Excellent with saved latent tail | No | **Recommended** |
| C, pixel overlap plus trim | Moderate | High | Moderate | Good | No | Supplementary seam selection |
| C, pixel crossfade | Low | Low | Low | Good | No | Avoid over bodies/faces |
| D | High overall | High plus load thrash | High | Good | **Yes** | Manual exception only |
| E | Moderate | Moderate | Moderate | Good | Only if target generated externally | Planned endpoint shots |
| Hosted `/v1/extend` | Remote | API-dependent | Low locally | API-dependent | No | Not suitable for the required local architecture |
| Local `LTXVLoopingSampler` | Bounded per tile | Moderate/high | Low graph complexity, poor per-chunk durability | Weaker without orchestration | No | Useful reference, not the top-level job runner |

### 4.3 Conclusions by criterion

- **Character identity:** Latent overlap plus stable textual anchors is strongest.
- **Anatomy stability:** Short windows reduce exposure to long-horizon compounding; latent overlap avoids resetting motion from one isolated image.
- **Pose continuity:** Multiple recent frames or their latent representation are materially better than one frame.
- **Motion direction and velocity:** Requires temporal context; one image cannot encode it explicitly.
- **Camera continuity:** Preserve a stable camera-axis contract and recent latent motion.
- **Lighting/background continuity:** Use persistent environment anchors and final-resolution overlap guidance.
- **Cumulative degradation:** Do not continually decode and re-encode the entire chain. Keep the original start image as a long-term reference candidate and save first-pass latent context.
- **VRAM:** A rolling 121-frame window with a 24-frame tail is bounded and feasible to test on 16 GB; an ever-growing latent is not.
- **Crash recovery:** Per-chunk immutable attempts plus saved latent tails are more recoverable than one monolithic LoopingSampler execution.
- **Adult-only anatomical stability:** The same method is recommended. Prompts and metadata must explicitly identify every depicted person as an adult; continuity checks must not introduce age ambiguity.
- **T2I residency:** No boundary operation requires a T2I model.

---

## 5. Recommended ComfyUI graph and conditioning approach

### 5.1 Top-level design

Use the Lightricks custom nodes as a **worker graph**, but keep chunk dependency management in 10MinVideoMaker.

Do not place the full scene inside one monolithic `LTXVLoopingSampler` graph because:

- It returns a cumulative result rather than durable per-chunk job artifacts.
- Fine-grained restart and remake semantics become harder.
- It does not support combined AV latents.
- It can allow an increasingly large output latent to remain live.
- The application needs database-visible chunk states and attempts.

Use `LTXVExtendSampler` directly under application orchestration.

### 5.2 First chunk: low-resolution first pass

At 384×672:

1. Create an LTX video latent for the calculated first-window frame count.
2. Apply the exact cached T2I image at frame zero.
3. Use the official I2V conditioning path already validated in the current workflow.
4. Apply only approved LTX-model LoRAs:
   - LTX2.3 DMD at 1.0
   - Local JoyAI-derived adapter at 0.5, pending exact compatibility verification
   - Any segment LoRA explicitly declared compatible with LTX 2.x
5. Reject image-model LoRAs before graph submission.
6. Sample with LCM and the current first-pass sigma sequence.
7. Save:
   - Full first-pass denoised window latent
   - Bounded handoff latent tail
   - Preview decode
   - Exact conditioning, seed, model, LoRA, and workflow hashes

The exact cached T2I image must remain the first displayed frame after second-pass refinement. Validate it perceptually against the cached PNG.

### 5.3 Continuation chunks: first-pass latent overlap

For every later chunk:

1. Load only the previous accepted chunk’s bounded first-pass latent tail.
2. Create a continuation latent whose temporal span is:
   - 24 overlap transitions
   - Plus up to 96 new transitions
3. Run `LTXVExtendSampler` with:
   - `frame_overlap = 24`
   - Existing LCM sampler
   - Existing first-pass sigma list
   - Deterministic derived noise seed
   - `STGGuiderAdvanced`
4. Use the node’s official latent-overlap mechanism rather than a decoded-frame restart.
5. Save the resulting rolling window and its new tail.
6. Move saved tails to CPU memory or disk before the next generation.
7. Release all prior nonessential tensors and graph outputs.

`LTXVExtendSampler` requires `STGGuiderAdvanced`, even though the current workflow uses LCM. The node accepts arbitrary samplers and sigmas, but the exact zero-CFG/zero-STG configuration needed to preserve the current distilled DMD behavior must be checked locally. The official LTX API defines CFG 1.0 and STG 0.0 as disabled guidance settings. citeturn504112view4turn504112view0

### 5.4 Bounded rolling state

Do not pass the complete scene latent into every extension.

After an extension:

- Persist the newly stabilized latent range.
- Retain only the final overlap/context tail as the next chunk’s live input.
- Keep the rest of the scene latent on disk as immutable segments.
- Reconstruct the scene from segment manifests rather than one giant in-memory tensor.

The extension node itself can concatenate to its supplied latent. Therefore the application should supply a bounded rolling latent, not the complete accumulated scene.

### 5.5 One-overlap delayed commit

The overlap used to condition chunk `N+1` is blended with the prior tail. Consequently, the final overlap of chunk `N` should remain provisional until chunk `N+1` succeeds.

For each chunk:

- **Committed prefix:** Frames that cannot be changed by a later extension
- **Pending tail:** The final 25 stored overlap samples
- **New window:** Pending tail plus new generation
- **After success:** Replace the old pending tail with the fused overlap, commit the portion now outside the next handoff window, and retain a new pending tail
- **Final chunk:** Flush the last pending tail

This prevents the application from permanently encoding one version of an overlap and then generating a different latent-space version in the next step.

### 5.6 Second pass

Use the existing x2 latent upscaler to move 384×672 first-pass latents to 768×1344. The official example uses `LTXVLatentUpsampler` followed by `LTXVImgToVideoConditionOnly` and the second sampler. citeturn613545view2

Recommended second-pass behavior:

#### First window

- Upscale the stabilized first-pass latent.
- Reapply the original cached T2I start image at strength 1.0 through the currently validated official I2V path.
- Run the existing second-pass sigmas:
  - `0.909375, 0.725, 0.421875, 0.0`
- Verify that frame zero remains the exact intended starting composition.

#### Later windows

- Upscale the stabilized first-pass overlap-plus-new window.
- Supply the prior accepted final-resolution overlap as a **25-frame video guide** at frame index eight through core `LTXVAddGuide`. The bounded noninitial handoff carries one sacrificial causal predecessor token, decoded as an eight-frame preroll; index eight aligns the guide with the first visible frame retained by assembly.
- Twenty-five stored samples satisfy `8n + 1`.
- Start with the node’s documented default guide strength of 1.0.
- Retain the first-pass latent as the primary motion source; the final-resolution guide exists to protect visible detail and seam appearance.
- Sample with the second-pass sigma list.
- Decode only the range required for the provisional overlap and newly generated frames.

**Required check:** Compare `LTXVAddGuide` against prefix replacement through `LTXVImgToVideoConditionOnly` for the second-pass overlap. The former is the architectural default because it supplies guidance without blindly overwriting the whole overlap. Prefix replacement is the fallback if the guide fails to preserve the exact boundary appearance.

### 5.7 Which representation should continue the next chunk?

Use:

- **Primary continuation state:** First-pass denoised video latent tail
- **Visible overlap guide:** Final upscaled decoded overlap frames
- **Assembly source:** Final upscaled decoded frames
- **Not primary continuation state:** Literal final decoded frame
- **Not primary continuation state:** Spatially upscaled latent alone

The first-pass latent is generated before spatial detail synthesis and retains temporal information in the representation the extension node was designed to consume. The final decoded overlap preserves the appearance that viewers actually saw.

### 5.8 Long-term identity anchor

`LTXVLoopingSampler` exposes negative-index latents for long-term context and normalization latents for combating drift. citeturn134011view2turn134011view3

Do not enable these in the first production version. First validate the simpler recent-overlap design.

Reserve a phase-two option to encode the original scene-start image or a selected clean identity frame as a low-strength negative-index reference for every continuation. That must receive its own bounded check because overstrong global reference conditioning could resist legitimate pose and camera changes.

---

## 6. Exact chunk duration and rounding algorithm

### 6.1 Two duration domains

Maintain separate values:

- `requested_duration_seconds`
- `timeline_output_frames`
- `generation_transition_frames`
- `generation_master_frames`

At 24 fps:

`timeline_output_frames = max(1, round(requested_duration_seconds × 24))`

Choose enough LTX transitions to cover the requested output:

`generation_transition_frames = 8 × ceil(max(timeline_output_frames - 1, 0) / 8)`

`generation_master_frames = generation_transition_frames + 1`

This generates no more than seven extra frames and never generates fewer frames than needed.

After assembly:

- Retain the LTX-compatible generation master.
- Trim the normal scene artifact to exactly `timeline_output_frames`.

### 6.2 Chunk construction

Constants:

- `base_window_transitions = 120`
- `overlap_transitions = 24`
- `full_extension_new_transitions = 96`

Algorithm:

1. If target transitions are zero, produce a single image-frame scene only if the pipeline supports it.
2. If target transitions are 120 or fewer, generate one `target + 1` frame LTX window.
3. Otherwise:
   - First chunk contributes 120 transitions.
   - Each full continuation contributes 96 new transitions.
   - The final continuation contributes the remaining multiple of eight.
4. Each continuation model window spans:
   - 24 overlap transitions
   - Plus its new transitions
   - Plus one stored initial sample
5. A short final continuation is valid because:
   - Overlap is a multiple of eight.
   - The remaining new-transition count is a multiple of eight.
   - Their sum plus one satisfies `8n + 1`.

### 6.3 Example durations

| Requested duration | Timeline frames | LTX master frames | Window plan |
|---:|---:|---:|---|
| 5.0 s | 120 | 121 | 120 transitions |
| 10.0 s | 240 | 241 | 120 + 96 + 24 |
| 20.0 s | 480 | 481 | 120 + 96 + 96 + 96 + 72 |
| 30.0 s | 720 | 721 | 120 + 96 × 6 + 24 |
| 32.0 s | 768 | 769 | 120 + 96 × 6 + 72 |

### 6.4 Explicit frame-count requests

When Grok or the job explicitly supplies an LTX frame count:

- Accept it directly only when it is `8n + 1`.
- Treat it as `generation_master_frames`.
- Derive its transition count as `frames - 1`.
- Do not reinterpret it as an ordinary timeline frame count.
- Let a separate optional `timeline_output_frames` determine final trimming.

### 6.5 Duplicate first frames

Under a one-frame-chain fallback, remove frame zero of every chunk after the first.

Under the recommended latent-overlap method, do **not** apply a blanket “drop first frame” rule. The assembler must use each chunk’s committed range from its manifest because the overlap contains more than one frame and may have been latent-blended.

---

## 7. Prompt, beat, continuity-anchor, and seed strategy

### 7.1 Prompt division

Grok should provide explicit ordered beats going forward.

Each approximately-five-second beat should describe:

1. The adult subject’s state at the beginning
2. The action over the interval
3. The state at the end
4. Camera movement
5. Motion direction and approximate speed
6. Persistent identity and wardrobe anchors
7. Persistent environment and lighting
8. Relevant audio
9. Any required hand, object, or body contact continuity

This follows Lightricks’ official advice to describe action chronologically, with movements, appearances, camera behavior, lighting, and changes in a single flowing description. citeturn206669view0

### 7.2 Global prompt prefix

Create an immutable scene continuity prefix containing:

- Explicit adult ages or unmistakable adulthood
- Stable facial and body identifiers
- Hair
- Wardrobe and accessories
- Environment
- Time of day
- Key and fill lighting
- Lens and shot scale
- Camera side of the action axis
- Character screen positions
- Dominant motion direction
- Objects held and the relevant hand
- Negative constraints that must apply to every beat

The exact phrasing of these anchors should remain constant across chunks.

### 7.3 Beat prompt

Append a segment-specific action paragraph describing only what should occur in that temporal window.

Avoid:

- Reintroducing the scene as though it has restarted
- Repeating “begins to” at every chunk
- Contradicting the prior end pose
- Changing camera direction without an explicit transition
- Adding multiple unrelated actions to one five-second window
- Asking a single window to cover an entire 30-second narrative

Use continuity language such as:

- “Continuing the same uninterrupted movement…”
- “Her momentum carries toward screen right…”
- “The camera maintains the same leftward dolly speed…”
- “His right hand remains in contact with the object…”

### 7.4 Negative prompts

Use:

- A stable scene-level negative prompt
- An optional segment override that adds constraints
- No segment override may remove mandatory adult-safety or anatomical-quality constraints
- Store the final resolved negative prompt in each attempt manifest

### 7.5 Seeds

Use deterministic derived seeds.

Do not use the same raw seed for every chunk. Reusing the same seed with similar prompts can encourage repeated motion motifs and does not reproduce a continuous noise field.

Do not use unrelated random seeds because that weakens reproducibility and remake semantics.

Recommended policy:

`chunk_seed = deterministic_hash(seed_policy_version, job_id, scene_id, scene_revision, base_seed, chunk_index, beat_prompt_hash, variation_index)`

Requirements:

- Fixed 64-bit mapping
- Versioned derivation algorithm
- Stored resolved seed
- Stable across crash recovery
- `variation_index = 0` for the first attempt
- Increment only when intentionally requesting a new variation

The official LoopingSampler similarly supports a base seed plus per-tile offsets, which supports the principle of deterministic but distinct temporal-tile noise. citeturn134011view3

### 7.6 LoRA policy

Resolve LoRAs at scene or beat level.

Validation must reject:

- Any T2I/image-model LoRA on the LTX loader path
- Any LoRA without a declared model family
- Any dynamic LoRA not explicitly compatible with LTX 2.x
- Any unresolved file or hash

Changing a dynamic LTX LoRA between adjacent chunks increases seam risk. Prefer applying it for a complete scene. Segment-specific LoRA changes must be reserved for deliberate, visibly motivated transitions.

---

## 8. Proposed backward-compatible JSON schema

The existing `scene.i2v.prompt` remains valid.

New jobs should add `schema_version`, `continuation`, and `segments`.

```json
{
  "scene_id": "scene_0001",
  "requested_duration_seconds": 30.0,
  "i2v": {
    "prompt": "Backward-compatible scene-level fallback prompt.",
    "negative_prompt": "Scene-level negative prompt.",
    "base_seed": 43857291,
    "continuation": {
      "enabled": true,
      "strategy": "ltx23_latent_overlap_v1",
      "fps": 24,
      "base_window_transition_frames": 120,
      "overlap_transition_frames": 24,
      "timeline_duration_policy": "exact_cfr_trim",
      "seed_policy": "derived_v1",
      "audio_policy": "chunk_audio_crossfade_v1",
      "boundary_validation_profile": "ltx23_human_motion_v1"
    },
    "continuity": {
      "all_characters_unambiguously_adult": true,
      "identity_anchors": [
        "A 28-year-old adult woman with ...",
        "A 34-year-old adult man with ..."
      ],
      "wardrobe_anchors": [
        "The adult woman continues wearing ...",
        "The adult man continues wearing ..."
      ],
      "environment_anchors": [
        "The room layout remains ...",
        "Warm key light remains camera-left ..."
      ],
      "camera_axis": "Camera remains on the south side of the action axis.",
      "screen_direction": "Primary movement continues toward screen right."
    },
    "segments": [
      {
        "index": 0,
        "requested_duration_seconds": 5.0,
        "positive_prompt": "Resolved first beat prompt.",
        "negative_prompt_additions": [],
        "action_beat": "The ordered action for this interval.",
        "camera": {
          "shot_scale": "medium",
          "movement": "slow dolly right",
          "direction": "right",
          "speed": "constant",
          "axis_policy": "preserve"
        },
        "continuity_start": {
          "pose": "Initial adult pose.",
          "screen_position": "center-left",
          "gaze": "toward ...",
          "contacts": [
            "Right hand holds ..."
          ],
          "motion_vector": "toward screen right",
          "motion_state": "already moving"
        },
        "continuity_end": {
          "pose": "Expected end pose.",
          "screen_position": "center",
          "gaze": "toward ...",
          "contacts": [
            "Right hand still holds ..."
          ],
          "motion_vector": "toward screen right",
          "motion_state": "continues moving at the same pace"
        },
        "audio": {
          "ambience": "Continuous room ambience.",
          "dialogue": null,
          "boundary_must_be_silent": true,
          "one_shot_events": []
        },
        "seed_policy": "derived_v1",
        "seed_override": null,
        "variation_index": 0,
        "ltx_loras": []
      }
    ]
  }
}
```

### 8.1 Schema rules

- `segments` is optional.
- Existing `i2v.prompt` is required as a fallback until migration is complete.
- If segments are present, their requested durations must cover the scene.
- The planner converts duration to exact timeline frames and LTX generation frames.
- Segment boundaries may be adjusted by up to the generation rounding allowance.
- A segment may supply `new_transition_frames` instead of seconds for deterministic control.
- Segment transition counts must resolve to multiples of eight.
- Human-containing segments must inherit `all_characters_unambiguously_adult = true`.
- At least one explicit adult identity anchor must exist for every depicted person.
- Segment-specific LoRAs require:
  - `family`
  - `filename`
  - `hash`
  - `strength`
  - `compatibility_source`
- The resolved generation manifest stores the fully expanded prompt and LoRA list, not merely references to scene defaults.

### 8.2 Old-prompt fallback

A non-LLM splitter may safely split an existing prompt only when the prompt already has:

- Explicit timestamps
- Numbered beats
- Clearly ordered clauses
- A machine-readable shot list

It should not attempt semantic action decomposition from arbitrary prose.

Fallback for an ordinary old prompt:

- Reuse the same prompt for every chunk.
- Add a deterministic continuation instruction.
- Preserve the global anchors.
- Mark `prompt_segmentation_quality = "fallback_reused_prompt"`.

This is backward compatible but lower confidence than Grok-authored beats.

---

## 9. Scheduling and VRAM lifecycle

### 9.1 Job-level lifecycle

1. Load the selected T2I model.
2. Generate all scene-start images.
3. Persist and validate every start image.
4. Release T2I model, VAE, CLIP, and all T2I LoRAs.
5. Clear references and perform ComfyUI model cleanup.
6. Load:
   - LTX-2.3 model
   - LTX VAE
   - Gemma/text encoder as required
   - Spatial upscaler
   - Mandatory LTX LoRAs
7. Keep the LTX stack resident for all scenes.
8. Release it only after all scene revisions selected for the final job are complete.

Official LTX documentation recommends FP8 casting and CPU or disk offload for constrained consumer GPUs. citeturn206669view0turn504112view4

### 9.2 Scene ordering

Process:

`scene 1 chunk 0 → scene 1 chunk 1 → … → scene 1 assembly → scene 2`

Do not interleave dependent chunks from different scenes unless there is a demonstrated GPU utilization problem. Sequential scene completion provides:

- Better data locality
- Simpler dependency handling
- Faster visible completion
- Less state swapping
- Simpler crash recovery
- No risk of confusing latent tails

### 9.3 Chunk ordering

Within a scene:

1. Generate first-pass chunk.
2. Persist first-pass state.
3. Generate next first-pass chunk.
4. Fuse overlap.
5. Refine and commit the prior stabilized range.
6. Continue.
7. Flush final pending range.
8. Assemble the scene.

Only one diffusion sampling operation should be active on the GPU.

### 9.4 Dynamic LoRA scheduling

Mandatory LoRAs remain loaded.

When dynamic LTX LoRAs vary by scene:

- Group complete scenes by identical model-patch signature only if scene order is not semantically important.
- Never interleave chunks from separate scenes solely to save LoRA changes.
- Cache patched model variants only when this fits VRAM and ComfyUI’s model-patcher contract.
- Otherwise patch at scene boundaries.

### 9.5 Memory residency

GPU-resident:

- Active LTX model or its offloaded working subset
- Active 121-frame-or-shorter latent window
- Active sampler tensors
- Active spatial-upscale window
- Active VAE decode tiles

CPU/disk-resident:

- Accepted prior latent segments
- Previous overlap guide frames
- First-pass handoff latent tail when not actively sampled
- Manifests
- Audio
- FFprobe output
- Preview files

Do not retain:

- Previous ComfyUI execution graphs
- Complete decoded frame tensors
- Complete cumulative scene latent on the GPU
- T2I model during video generation

---

## 10. Database and state-machine changes

### 10.1 Scene states

- `PLANNED`
- `START_IMAGE_READY`
- `VIDEO_READY`
- `ASSEMBLING`
- `VALIDATING`
- `COMPLETE`
- `FAILED_RETRYABLE`
- `FAILED_TERMINAL`
- `STALE`
- `CANCELLED`

### 10.2 Chunk states

- `PLANNED`
- `BLOCKED_UPSTREAM`
- `READY`
- `GENERATING_STAGE1`
- `STAGE1_PERSISTING`
- `STAGE1_COMPLETE`
- `GENERATING_STAGE2`
- `STAGE2_PERSISTING`
- `DECODED`
- `VALIDATING`
- `COMPLETE`
- `FAILED_RETRYABLE`
- `FAILED_TERMINAL`
- `STALE_UPSTREAM`
- `INVALIDATED`
- `CANCELLED`

### 10.3 Attempt model

Do not overwrite attempts.

Each chunk attempt stores:

- `chunk_attempt_id`
- `chunk_id`
- `attempt_number`
- `variation_index`
- `state`
- `seed`
- `seed_policy_version`
- `prompt_hash`
- `negative_prompt_hash`
- `continuity_hash`
- `workflow_hash`
- `comfyui_commit`
- `custom_node_commit`
- `model_hash`
- `vae_hash`
- `upscaler_hash`
- Ordered LoRA hashes and strengths
- Input start-frame hash
- Input handoff-latent hash
- Input overlap-frame hash
- Expected and observed frame counts
- GPU peak-memory telemetry
- Start/end timestamps
- Error type and diagnostic
- Artifact manifest path
- Validation result

### 10.4 Dependencies

Chunk zero depends on:

- Selected scene-start image revision
- Scene prompt/continuity revision
- LTX configuration revision

Chunk `N` depends on:

- Selected accepted attempt of chunk `N-1`
- Exact `handoff_artifact_hash` from that attempt
- The current scene revision
- The selected beat revision

An attempt is valid only when its recorded upstream hash equals the selected upstream artifact’s current hash.

### 10.5 Valid cache determination

A cached chunk is usable only when:

1. Database state is `COMPLETE`.
2. Generation manifest exists and reports completion.
3. All required files exist.
4. File sizes are nonzero.
5. Cryptographic hashes match.
6. Model, workflow, LoRA, prompt, and seed signatures match the requested generation.
7. Upstream handoff hash matches.
8. FFprobe validation passes.
9. Expected frame count, resolution, and fps match.
10. Latent-tail shape metadata matches the current VAE contract.
11. Boundary validation is accepted or explicitly overridden.
12. No ancestor is stale or invalidated.

---

## 11. D-drive artifact layout

```text
D:\10MinVideoMaker\
  jobs\
    {job_id}\
      job-manifest.json
      database-snapshot.json
      source\
        grok-job.json
      scenes\
        scene_0001\
          scene-manifest.json
          revisions\
            0001\
              revision-manifest.json
              start\
                start_frame.png
                start_frame-manifest.json
              chunks\
                chunk_0000\
                  chunk-manifest.json
                  attempts\
                    0001\
                      attempt-manifest.json
                      resolved-input.json
                      workflow-api.json
                      comfy-prompt-response.json
                      stage1\
                        denoised_window.safetensors
                        handoff_tail.safetensors
                        committed_segment.safetensors
                        latent-metadata.json
                      stage2\
                        refined_window.safetensors
                        refined-metadata.json
                      video\
                        raw_window.mkv
                        committed_video.mkv
                        overlap_preview.mkv
                        final_frame.png
                      audio\
                        raw_audio.flac
                        committed_audio.flac
                      validation\
                        validation.json
                        ffprobe.json
                        boundary-metrics.json
                        boundary-preview.mp4
                      logs\
                        comfyui.log
                        ffmpeg.log
                      COMPLETE.json
                chunk_0001\
                  ...
              assembly\
                generation_master.mkv
                generation_master.ffprobe.json
                scene_raw.mp4
                scene_raw.ffprobe.json
                scene_audio.flac
                assembly-manifest.json
                validation.json
              previews\
                scene_preview.mp4
              delivery\
                discord_watermarked.mp4
                delivery-manifest.json
```

### 11.1 File-format recommendations

- Latents: safetensors or another explicitly versioned tensor container
- Lossless/intermediate audio: FLAC or WAV
- Window video: high-quality intra-friendly mezzanine or lossless FFV1/MKV where storage permits
- Normal scene artifact: H.264 MP4 with a fixed production profile
- Manifests: UTF-8 JSON with a schema version

### 11.2 Atomic writes

For every durable artifact:

1. Write to an attempt-local temporary name.
2. Flush application buffers.
3. Close the file.
4. Compute the hash.
5. Run format validation.
6. Rename atomically into its final attempt-local path.
7. Write the attempt manifest to a temporary file.
8. Rename the manifest.
9. Write `COMPLETE.json` last.
10. Commit the database transaction selecting the attempt.

Never use the existence of an MP4 alone as proof that generation completed.

---

## 12. Chunk-to-scene FFmpeg and audio assembly

### 12.1 Video assembly

Use explicit frame-index ranges from the chunk manifests.

For each accepted window:

- Include only its committed range.
- Exclude provisional overlap that was replaced by a later fused overlap.
- Include the final pending tail only from the final chunk.
- Normalize each input to:
  - 768×1344
  - 24/1 fps
  - Square pixels
  - Identical pixel format
  - Zero-based timestamps

Use FFmpeg’s trim and timestamp-reset filters conceptually as:

- Exact frame trim
- `setpts` reset
- Ordered concat

FFmpeg documents the concat filter as the appropriate method when re-encoding, while the concat demuxer is intended for cases that can avoid re-encoding. citeturn411493view1

### 12.2 Stream-copy concatenation

Lossless stream-copy concatenation is not the normal chunk-to-scene path because:

- Exact overlap removal is frame-level editing.
- Selected seam positions may not be packet boundaries.
- Crossfaded or retimed audio requires filtering.
- Independently encoded chunks may differ in codec parameters or GOP structure.
- A cut inside a GOP cannot be represented safely through ordinary stream copy.

Use one controlled scene-level re-encode.

Scene-to-final-project stream-copy concatenation may be possible later when every `scene_raw.mp4` uses exactly the same:

- Codec
- Profile
- Level
- Dimensions
- Pixel format
- Frame rate
- Time base
- Audio codec
- Audio sample rate
- Channel layout
- GOP policy

If any scene receives a filter, transition, rescale, or watermark, use a controlled final encode instead.

### 12.3 Pixel-space seam behavior

Default:

- Latent-overlap fusion
- Hard cut at the selected low-discontinuity point in the overlap
- No ordinary video crossfade over a face or body

A conventional dissolve can create:

- Double faces
- Double limbs
- Apparent anatomical deformation
- Ghosted clothing edges
- Transparent object contacts

A two-to-four-frame micro-dissolve may be allowed only when:

- No person occupies the blended region, or
- The validator confirms nearly identical geometry
- It materially reduces a small exposure or background mismatch

### 12.4 Seam selection

Within the overlap, score candidate seam positions using:

- Local perceptual difference
- Optical-flow continuity
- Luminance/color continuity
- Face/body landmark displacement when available
- Sharpness
- Duplicate/freeze detection

Select the lowest-cost valid seam away from:

- Eye blinks at mismatched phases
- Foot contact transitions
- Hand-object contact changes
- Fast limb motion
- Mouth closures or syllable transitions
- Motion-direction reversals

### 12.5 Scene codec and GOP

Recommended production scene artifact:

- H.264
- High profile
- YUV 4:2:0
- CFR 24 fps
- Closed GOP
- Keyframe interval: 24 frames
- Minimum keyframe interval: 24
- Scene-cut keyframe insertion disabled
- CRF approximately 14–16
- Slow or equivalent production preset
- First frame forced as a keyframe

Rationale:

- One-second GOPs support later editing and seeking.
- Closed GOPs reduce cross-file dependencies.
- Fixed GOP behavior makes future scene concatenation predictable.
- A low CRF limits additional compression damage without the storage cost of permanent lossless RGB frames.

Retain an optional lossless or near-lossless generation master for high-value jobs.

### 12.6 Audio assembly

Normalize each chunk audio stream to:

- 48 kHz
- Consistent channel layout
- Floating-point processing during filtering
- Timestamp zero before concatenation

For each boundary:

1. Trim audio to match the committed video range.
2. Remove any duplicated overlap duration.
3. Search for a nearby low-energy or zero-crossing position.
4. Apply a short gain ramp or equal-power crossfade.
5. Start with 100 ms.
6. Permit 80–150 ms for ambience.
7. Do not use a long one-second dissolve by default.
8. Do not overlap conflicting speech.

FFmpeg’s `acrossfade` supports specified durations and selectable fade curves. citeturn411493view0

### 12.7 Audio continuity policy

For non-dialogue scenes:

- Carry a continuous scene-level ambience bed under chunk audio.
- Crossfade chunk ambience around the seam.
- Preserve isolated generated events only once.

For dialogue:

- Grok must avoid placing boundaries inside spoken lines.
- Each line belongs wholly to one beat.
- A boundary should have a small silence or room-tone interval.
- If a seam intersects speech, fail validation and regenerate or move the boundary.
- Do not solve mismatched phonemes through crossfade.

### 12.8 Timestamp and FFprobe validation

Validate:

- Video width: 768
- Video height: 1344
- `r_frame_rate = 24/1`
- `avg_frame_rate = 24/1`
- Exact decoded frame count
- No unexpected VFR timestamps
- Start time at zero
- Monotonically increasing timestamps
- Expected duration within one frame
- Audio sample rate: 48000
- Audio starts at zero
- Audio/video duration difference no greater than one video frame
- No decoder errors
- Expected codec/profile/pixel format
- First packet or frame is a keyframe

The normal scene artifact must contain exactly `timeline_output_frames`.

---

## 13. Remake and downstream-invalidation rules

### 13.1 Remaking an earlier chunk

Remaking chunk 3 invalidates chunks 4 through the end of the scene.

Reason:

- Chunk 4 depends on chunk 3’s accepted latent tail.
- Any changed chunk-3 pose, motion, camera position, identity detail, or latent state changes the continuation history.
- Retaining chunks 4–6 would break the dependency chain even if their first frame looked superficially similar.

Old downstream artifacts remain as immutable historical attempts but are marked `STALE_UPSTREAM` and are not selected.

### 13.2 Remaking the final chunk

Only the final chunk and scene assembly are invalidated.

### 13.3 Video Only remake

A scene-level **Video Only** remake:

- Reuses the selected cached T2I start image.
- Creates a new scene-video revision.
- Regenerates chunk zero onward.
- Leaves the original T2I image revision unchanged.
- Invalidates scene assembly and project assembly.

### 13.4 Image + Video remake

An **Image + Video** remake:

- Creates a new T2I image revision.
- Creates a new scene-video revision linked to it.
- Invalidates every chunk.
- Regenerates the entire scene continuation chain.

### 13.5 Prompt-only change

Changing:

- Global identity anchors
- Wardrobe
- Environment
- Camera axis
- Scene-level negative prompt
- Mandatory LoRA set

invalidates all chunks.

Changing beat `N` invalidates chunk `N` and every later chunk.

### 13.6 Validation override

A user may accept a soft-warning boundary without regenerating. Record:

- User identity
- Timestamp
- Warning metrics
- Selected attempt
- Override reason

A structural failure, missing artifact, wrong frame count, corrupt latent, or adult-safety contract failure cannot be overridden as a valid cached chunk.

---

## 14. Boundary quality validation

### 14.1 Tier 1: structural checks

Always run:

- File existence and hash
- Resolution
- Pixel format
- Frame rate
- Exact frame count
- Decoder scan
- Black or zero-valued frame detection
- Timestamp monotonicity
- Audio format and duration
- Latent shape validation

These are deterministic hard failures.

### 14.2 Tier 2: inexpensive image checks

Run on downscaled frames:

- Perceptual hash
- Mean absolute pixel difference
- SSIM over boundary candidates
- Luma histogram difference
- Chroma histogram difference
- Laplacian-variance sharpness
- Highlight/shadow clipping
- Consecutive-frame duplicate detection
- Frozen-frame run detection

FFmpeg includes SSIM and related comparison filters; OpenCV or a small in-process image library can calculate the same class of metrics without loading a generative model. citeturn411493view0

### 14.3 Tier 3: optical flow

Use CPU OpenCV Farnebäck flow on approximately 192×336 grayscale frames.

Calculate:

- Median x/y flow
- Median magnitude
- Dominant direction
- Flow variance
- Acceleration over the final eight incoming and first eight outgoing frames
- Direction change at the seam
- Camera/background flow separately where practical

Soft-warn when:

- Significant motion reverses abruptly without the beat requesting it.
- Median magnitude collapses toward zero at every boundary.
- Flow jumps sharply relative to the preceding local distribution.

This directly detects the “motion restart” problem expected from single-frame continuation.

### 14.4 Lightweight landmarks

Optional advisory detectors:

- MediaPipe face landmarks
- MediaPipe pose
- A compact ONNX pose detector
- A compact face detector

Use them to measure:

- Face-box center and scale changes
- Shoulder/hip alignment
- Wrist and ankle jumps
- Number of detected people
- Sudden detector-confidence collapse

Do not hard-fail solely because a lightweight detector loses a face or body. Occlusion, unusual poses, framing, and stylization can cause false negatives.

### 14.5 Malformed handoff detection

Hard-fail or force review for:

- Almost-black or almost-white handoff frame
- Severe blur relative to neighboring frames
- One-frame temporal outlier
- Corrupt decode
- Unexpected person-count change
- Large body-layout jump with no requested cut
- Frozen run longer than the configured tolerance
- Missing expected overlap frames

### 14.6 Safer handoff-frame selection

Under the recommended latent-overlap path, do not replace the temporal endpoint with an unrelated “cleaner” frame. Use the complete recent overlap and select only the final assembly seam inside it.

Under the single-frame fallback:

- Search only the last eight frames.
- If an earlier handoff frame is selected, trim the source scene to end at that frame.
- Record the changed timeline.
- Never use an earlier image as the next start while retaining later source frames before the seam.

### 14.7 Initial threshold policy

Hard failures should be deterministic.

Soft metric thresholds should begin conservative and be calibrated from accepted internal examples. Do not invent universal landmark or optical-flow thresholds before measuring the pipeline’s normal distribution.

Store raw metrics so thresholds can be changed without regenerating video.

---

## 15. Failure recovery behavior

### 15.1 Restart reconciliation

On application startup:

1. Read the database.
2. Scan incomplete job revisions.
3. Reconcile each selected attempt against its manifest.
4. Treat temporary files as incomplete.
5. Validate files marked complete.
6. Downgrade invalid `COMPLETE` records to `FAILED_RETRYABLE` or `STALE`.
7. Identify the earliest incomplete or stale chunk.
8. Resume from that dependency point.

### 15.2 Stage-specific resume

If stage 1 completed and its full required latent is valid:

- Resume stage 2 without rerunning stage 1.

If stage 2 completed but decode failed:

- Resume decode.

If decode completed but validation or assembly failed:

- Resume validation or assembly.

If only a preview is missing:

- Regenerate the preview without touching diffusion output.

### 15.3 OOM behavior

On CUDA OOM:

1. Record peak memory and the failing phase.
2. Release graph outputs and invoke cleanup.
3. Retry the same attempt inputs once using:
   - Tiled VAE decode
   - More aggressive CPU offload
   - Serialized overlap guide loading
4. Do not silently change:
   - Resolution
   - Frame count
   - Prompt
   - Seed
   - LoRA strengths
   - Sigma sequence
5. If the retry fails, mark the configuration unsupported on the current worker.

Reducing overlap from 24 to 16 is a new configuration and a new attempt, not a transparent retry.

### 15.4 Boundary-validation failure

Allow one bounded automatic retry:

- Preserve all settings.
- Increment `variation_index`.
- Derive a new seed.
- Regenerate the failed chunk and all downstream chunks.

Do not launch repeated unbounded generations.

After the bounded retry:

- Select the better passing attempt, or
- Mark the scene as requiring review.

### 15.5 Corrupt handoff latent

If a handoff latent is corrupt or incompatible:

- Attempt to recover it from the persisted full stage-one window.
- If recovery is impossible, regenerate that chunk.
- Invalidate all descendants.

---

## 16. Minimal UI changes

Initial UI should remain scene-oriented.

Add:

- “Chunked continuation” badge
- Calculated generation-window count
- Scene progress such as `Chunk 3 of 8`
- Current phase:
  - First pass
  - Upscale/refinement
  - Validation
  - Assembly
- Seam status:
  - Pass
  - Warning
  - Failed
  - User accepted
- Resume indicator
- Selected scene revision
- “Video Only” remake
- “Image + Video” remake

Advanced expandable panel:

- Beat list
- Resolved frame counts
- Overlap size
- Seeds
- LoRAs
- Artifact paths
- Boundary previews
- Validation metrics
- Exact model/workflow hashes

Do not expose arbitrary “remake only this chunk” initially.

A later UI may expose:

- “Remake from chunk N”
- “Use alternate attempt from chunk N”
- “Open seam preview”
- “Accept seam warning”

The backend should support per-chunk attempts from the beginning even though the initial GUI remains scene-level.

---

## 17. Migration plan for existing jobs

### Phase 1: schema and persistence

- Add nullable continuation fields.
- Add scene-video revisions.
- Add chunk and chunk-attempt tables.
- Add artifact manifests.
- Existing jobs continue using the old single-generation path.

### Phase 2: feature flag

Add:

`ltx_chunked_continuation_v1`

Enable only for new test jobs.

### Phase 3: old prompt fallback

For migrated jobs without segments:

- Reuse the existing prompt.
- Generate deterministic chunk prompts with a fixed continuation wrapper.
- Mark segmentation quality as fallback.
- Do not rewrite the original Grok input.

### Phase 4: new Grok contract

Require explicit beats for newly generated long scenes.

### Phase 5: default activation

Enable chunked continuation by default for scenes whose generation master exceeds 121 frames.

Allow explicit opt-out for:

- Intentionally static shots
- Planned first/last-frame interpolation
- Existing validated legacy workflows
- Diagnostic comparisons

### Phase 6: legacy completed jobs

Do not invalidate completed legacy jobs automatically.

A legacy scene is converted only when the user remakes it or explicitly migrates it.

---

## 18. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| `LTXVExtendSampler` plus LCM/STG guider behaves differently from current sampler graph | Quality regression | Four-generation validation gate; pin node commits |
| Local custom-node version differs from researched `master` | Contract mismatch | Inspect installed source and workflow API JSON before coding |
| 24-frame overlap exceeds 16 GB budget | OOM | Rolling tail, no concurrent chunks, tiled decode; tested 16-frame fallback |
| Latent overlap changes a previously encoded tail | Seam mismatch | One-overlap delayed commit |
| Second pass changes anatomy inside overlap | Visible seam | 25-frame final-res guide; seam selection inside overlap |
| Pixel crossfade creates ghost anatomy | Severe visual artifact | Hard cuts by default |
| Single-frame fallback restarts motion | Repeated pauses | Use only as emergency fallback |
| Identity drifts over many chunks | Character inconsistency | Stable anchors, short beats, original-reference option in phase two |
| Lighting/color accumulates | Visible drift | Histogram validation; optional normalizing/AdaIN latent after separate check |
| Independent chunk audio resets | Clicks or ambience discontinuity | Exact trims, short equal-power crossfades, continuous ambience |
| Dialogue crosses a seam | Unusable speech | Grok boundary rules; scene-level dialogue track or regeneration |
| JoyAI/DMD adapters overlap semantically | Oversharpening, conditioning loss, quality instability | Identify exact artifacts and target modules; controlled compatibility test |
| JoyAI-derived artifact licensing is unsuitable | Distribution/commercial risk | Record source and license; do not infer rights from filename |
| Full latent persistence is large | Storage growth | Save bounded tails and committed segments; retention policy |
| Codec re-encode reduces quality | Generational loss | One scene-level high-quality encode; retain mezzanine master |
| Automated body detectors false-fail | Unnecessary retries | Advisory only except structural catastrophic cases |
| User edits an earlier beat | Downstream work wasted | Clear invalidation preview before remake |

---

## 19. Concrete acceptance criteria

### Functional

- A 30-second requested scene produces exactly 720 frames in `scene_raw.mp4`.
- The generation master contains sufficient LTX-compatible frames, normally 721 for that duration.
- Output is CFR 24 fps at 768×1344.
- The exact cached T2I image remains the opening composition.
- Chunk overlap is not duplicated.
- Scene output is accepted by the existing scene-to-project pipeline without special handling.
- Watermarking occurs only in the delivery branch.
- Raw scene and chunk artifacts remain unwatermarked.

### Continuity

- No recurring pause or restart is perceptible at every seam.
- Camera-flow direction remains continuous unless the beat requests a change.
- Identity, wardrobe, environment, and lighting remain stable.
- No seam creates double anatomy through pixel blending.
- Boundary-validator results are stored for every seam.
- Every depicted person remains unambiguously adult in prompts and continuity metadata.

### Reproducibility

- A chunk regenerated with identical inputs and `variation_index` reproduces its seed and workflow signature.
- Every artifact records exact model, LoRA, node, workflow, and prompt hashes.
- Changing a beat invalidates that chunk and every descendant.
- Crash recovery resumes from the latest valid persisted phase.

### Residency and resources

- T2I is not loaded during any continuation chunk.
- LTX remains loaded through the complete video phase.
- Only one chunk samples on the GPU at a time.
- The complete prior computation graph is not retained.
- No accepted test generation OOMs under the production memory profile.
- Peak VRAM is recorded.

### Audio

- No duplicated overlap audio.
- No audible click at accepted seams.
- Dialogue does not cross independently generated chunk boundaries.
- Audio/video duration differs by no more than one video frame.

### Compatibility

- Image-model LoRAs cannot enter the LTX model path.
- Dynamic LoRAs require declared LTX 2.x compatibility.
- The current first- and second-pass sigmas are manifest-pinned.
- The exact local JoyAI and DMD artifacts are identified before production activation.

---

## 20. Minimal bounded validation plan

Use one deliberately difficult but non-cutting adult-character shot:

- One clearly adult character
- Continuous lateral movement
- Visible hands
- A torso turn
- Persistent hand-object contact
- Slow lateral camera tracking
- Stable directional lighting
- No dialogue crossing the test seam

Use the same first 121-frame base generation for all continuation comparisons.

### Generation 1: common base

Generate the first 121-frame I2V chunk with the production model, LoRAs, LCM sampler, sigmas, and exact cached image.

Save:

- First-pass latent
- Final output
- Last 25 frames
- Metrics

### Generation 2: single-frame baseline

Continue from only the exact final decoded frame.

Purpose:

- Establish the expected motion-restart and VAE-round-trip baseline.

### Generation 3: decoded 25-frame guide

Continue with the final 25 decoded frames through `LTXVAddGuide`.

Purpose:

- Test the best simple AV-compatible temporal guide.
- Determine how much motion information survives decode and re-encode.

### Generation 4: latent-overlap continuation

Continue through `LTXVExtendSampler` with:

- 24-frame overlap
- Existing LCM sampler
- Existing first-pass sigmas
- Pass-through distilled guider configuration
- Same beat prompt and deterministic seed policy

Purpose:

- Verify local node compatibility, VRAM behavior, and continuity advantage.

### Measurements

For the three continuation outputs, compare:

- Peak VRAM
- Runtime
- Face-box displacement
- Optional pose-landmark displacement
- Optical-flow direction and magnitude discontinuity
- Boundary luma/chroma change
- Perceptual frame difference
- Hand-object contact continuity
- Visible motion restart
- Anatomy
- Camera velocity
- Identity and wardrobe
- Seam preference in a blinded side-by-side review

### Decision rule

Adopt the latent-overlap architecture when:

1. It completes without OOM.
2. LCM and the required guider produce the intended distilled result.
3. It has lower optical-flow discontinuity than the single-frame method.
4. It is at least as stable anatomically as the 25-frame decoded guide.
5. It does not create a more visible second-pass seam.
6. Runtime remains operationally acceptable.

The decoded 25-frame guide remains the emergency fallback, not a coequal production strategy.

---

# Final recommended architecture

Use **rolling, first-pass latent-overlap continuation with `LTXVExtendSampler`, 121-frame nominal model windows, a 24-transition/25-sample overlap, deterministic beat-specific prompts and seeds, bounded persisted latent tails, one-overlap delayed commits, overlap-aware second-pass refinement, hard seam selection, and exact CFR scene trimming**.

Keep the original T2I image as the first-frame anchor, but never reload the T2I model during video generation.

Do not use routine T2I boundary repairs.

Do not use one-frame continuation except as a diagnostic or emergency fallback.

Do not place the whole job inside one monolithic LoopingSampler execution.

---

# Ordered implementation checklist for Codex

1. Pin the installed ComfyUI commit.
2. Pin the installed Lightricks `ComfyUI-LTXVideo` commit.
3. Export the current working two-stage ComfyUI workflow in API format.
4. Hash the LTX checkpoint, VAE, upscaler, mandatory LoRAs, and text encoder.
5. Identify the exact JoyAI-derived artifact and its model type.
6. Confirm whether the DMD and JoyAI-derived artifacts modify overlapping modules.
7. Add job, scene-video-revision, chunk, chunk-attempt, and dependency tables.
8. Add schema-versioned artifact manifests.
9. Add separate timeline-frame and generation-frame fields.
10. Implement the frame-planning algorithm.
11. Add backward-compatible segment parsing.
12. Add resolved prompt and continuity-anchor generation.
13. Add deterministic derived-seed policy `derived_v1`.
14. Add an LTX LoRA compatibility allowlist.
15. Add a hard guard preventing image-model LoRAs from entering the LTX graph.
16. Create the first-chunk API workflow based on the current validated I2V graph.
17. Create a continuation-worker workflow around `LTXVExtendSampler`.
18. Configure the worker to accept only a bounded rolling video latent.
19. Add handoff-tail serialization and restoration.
20. Add one-overlap delayed-commit bookkeeping.
21. Add stage-one persisted-resume behavior.
22. Add second-pass overlap refinement.
23. Add 25-frame final-resolution `LTXVAddGuide` conditioning for later windows.
24. Add stage-two persisted-resume behavior.
25. Add tiled VAE decoding where required.
26. Add immutable per-attempt artifact directories.
27. Add atomic temporary-write and rename behavior.
28. Add file hashing and manifest finalization.
29. Add structural FFprobe validation.
30. Add perceptual, duplicate, luma, and sharpness metrics.
31. Add downscaled CPU optical-flow metrics.
32. Add optional advisory landmark metrics.
33. Add seam-candidate scoring.
34. Add committed-range assembly manifests.
35. Add chunk-to-scene controlled re-encode.
36. Add exact timeline-frame trimming.
37. Add 48-kHz audio normalization and short equal-power seam crossfades.
38. Add dialogue-boundary rejection.
39. Add scene-to-project compatibility validation.
40. Add downstream invalidation from the remade chunk onward.
41. Add Video Only and Image + Video scene-revision behavior.
42. Add restart reconciliation.
43. Add one bounded automatic retry.
44. Add model-residency assertions and telemetry.
45. Add the four-generation validation fixture.
46. Block feature activation until its acceptance criteria pass.
47. Roll out behind `ltx_chunked_continuation_v1`.
48. Migrate Grok to explicit beat generation.
49. Enable the feature by default for new scenes longer than 121 generation frames.
50. Preserve the legacy path for completed jobs and explicit overrides.

---

# Exact local ComfyUI contracts Codex must inspect before coding

Codex must inspect the **installed**, not merely current-online, versions of:

1. `LTXVExtendSampler`
   - Input and return types
   - `num_new_frames` semantics
   - Pixel-frame versus transition-frame accounting
   - Temporal compression factor
   - Exact overlap output length
   - Whether input mutation occurs
   - Device placement
   - Optional image/keyframe behavior

2. `LinearOverlapLatentTransition`
   - Whether it changes the old overlap
   - Inclusive/exclusive indices
   - Output length
   - Blend weights
   - Tensor-device requirements

3. `STGGuiderAdvanced`
   - Required object class
   - Raw-conditioning storage
   - CFG 1.0/STG 0.0 pass-through behavior
   - Compatibility with the current DMD/LCM graph
   - Whether negative conditioning remains required

4. Current LCM sampler node
   - Exact sampler identifier
   - Expected model-sampling patch
   - Sigma consumption
   - Whether it returns output or denoised output as the persisted latent

5. `LTXVAddGuide`
   - IMAGE batch order
   - 25-frame guide encoding
   - Frame-index rounding
   - Strength behavior
   - Guide-token cleanup through `LTXVCropGuides`
   - Behavior before and after AV concatenation

6. `LTXVImgToVideoConditionOnly`
   - Whether multi-frame IMAGE batches replace the intended prefix
   - Noise-mask semantics
   - Input latent mutation
   - Exact first-frame preservation

7. `LTXVLatentUpsampler`
   - Input/output temporal shape
   - Whether temporal samples change
   - Device and dtype behavior
   - VAE requirement
   - Compatibility with serialized segment latents

8. `EmptyLTXVLatentVideo`
   - Frame-to-latent-length formula
   - Behavior for shorter final windows
   - Minimum accepted length

9. `LTXVConcatAVLatent` and `LTXVSplitAVLatent`
   - Nested-tensor representation
   - Audio/video temporal alignment
   - Whether second-pass audio is sampled, frozen, or merely carried
   - Serialization compatibility

10. `SamplerCustomAdvanced`
    - Difference between `output` and `denoised_output`
    - Which result the official two-stage workflow feeds to the upscaler
    - Noise-mask behavior

11. `ManualSigmas`
    - Exact list parsing
    - Whether terminal zero is retained
    - Dtype/device

12. VAE decode nodes
    - Temporal output frame count
    - First-latent asymmetry
    - Tiled temporal-overlap behavior
    - Last-frame fix behavior

13. `CreateVideo` and `SaveVideo`
    - FPS metadata
    - Audio attachment
    - Codec defaults
    - Whether output is CFR
    - Frame-count preservation

14. Model and LoRA loaders
    - Patch order
    - Whether repeated LoRA application mutates the resident model
    - Ability to clone or cache patched models
    - Unload behavior
    - Q8/FP8 loader compatibility

15. Current API workflow
    - Exact node IDs and class types
    - Model residency behavior between queued prompts
    - Whether application-level partial execution can reuse loaded models
    - Output artifact identifiers returned through ComfyUI history

---

# Changes to the Grok skill

1. Require `schema_version`.
2. Require every depicted person to be explicitly and unambiguously adult.
3. Generate ordered approximately-five-second beats for scenes longer than one window.
4. Keep a stable scene continuity prefix.
5. Include start and end state for each beat.
6. Include camera axis, direction, movement, and speed.
7. Include screen direction and subject screen position.
8. Track which hand holds or touches each relevant object.
9. Track wardrobe and accessories explicitly.
10. Track environment and lighting explicitly.
11. Keep one coherent action progression per beat.
12. Avoid restarting language at each beat.
13. Avoid dialogue across segment boundaries.
14. Place segment boundaries in silence or steady ambience.
15. Supply one-shot audio events only in the beat where they occur.
16. Mark whether a camera or motion discontinuity is intentional.
17. Supply segment-specific LoRAs only with explicit LTX 2.x compatibility metadata.
18. Retain `scene.i2v.prompt` as the complete fallback prompt.
19. Never attempt to repair an old arbitrary prompt with a naïve text splitter.
20. Return requested seconds while allowing the application to resolve exact frame counts.

---

# Assumptions still requiring confirmation

1. Exact ComfyUI version and commit.
2. Exact `ComfyUI-LTXVideo` version and commit.
3. Exact LTX-2.3 checkpoint variant and quantization.
4. Exact local LCM sampler node and settings.
5. Exact implementation of `STGGuiderAdvanced`.
6. Whether the current two-pass graph samples audio in both passes or carries first-pass audio through the second.
7. Whether native LTX-generated synchronized audio is mandatory for every scene.
8. Exact filename, hash, source, type, and license of the local “JoyAI Echo” adapter.
9. Whether that adapter already includes DMD-derived changes.
10. Whether the mandatory LoRA combination has been tested with `LTXVExtendSampler`.
11. Whether 24-frame overlap fits the current 16 GB low-VRAM workflow.
12. Whether `LTXVAddGuide` at strength 1.0 is the best second-pass overlap anchor locally.
13. Whether the current second-pass start sigma of 0.909375 is intentional relative to the official current 0.85 example.
14. Whether scene `duration_seconds` should mean exact encoded duration or first-to-last-frame span. This architecture uses exact encoded duration for `scene_raw.mp4`.
15. Whether lossless generation masters should be retained indefinitely or governed by a storage-retention policy.
16. Whether the final project encoder already enforces a common codec, GOP, time base, and audio profile.
17. Whether optional landmark validation is acceptable for the types of framing and occlusion used by the production system.
