# Continuation Findings and Seam Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the unexpected style-conversion experiments and add an exact assembled-seam review video while leaving production continuation and automatic rollout unchanged.

**Architecture:** Keep immutable experiment evidence in a versioned `experiments/` folder and large media on the existing D-drive acceptance run. Extend `AcceptanceReviewService` with one storage-bounded FFmpeg concat proxy and expose it through the existing authenticated review API. Add one player below the current comparison UI; existing views and controls remain intact.

**Tech Stack:** Python 3.12, FastAPI, FFmpeg, vanilla JavaScript/CSS, `unittest`.

## Global Constraints

- Read and write only the authorized repository plus project-owned `D:\LTX_Supervisor_Storage`.
- Never access protected VRGDGirl/Violets projects.
- Do not queue ComfyUI, restart ComfyUI, modify shared resources, or interrupt active renders.
- Keep `continuation-validation-v1.json` absent/unchanged and automatic continuation locked.
- Keep raw and review media unwatermarked.
- Keep all large video artifacts on D:.
- Preserve current side-by-side and still-frame review surfaces.
- Use `apply_patch` for source edits and test-driven development for behavior changes.

---

### Task 1: Freeze experiment evidence

**Files:**
- Create: `experiments/ltx23-style-conversion/README.md`
- Create: `experiments/ltx23-style-conversion/manifest.json`
- Create: `experiments/ltx23-style-conversion/frozen-workflows/decoded_17_frame_stage1.api.json`
- Create: `experiments/ltx23-style-conversion/frozen-workflows/decoded_17_frame_stage2.api.json`
- Create: `experiments/ltx23-style-conversion/frozen-workflows/latent_overlap_stage1.api.json`
- Create: `experiments/ltx23-style-conversion/frozen-workflows/latent_overlap_stage2.api.json`
- Create: `tests/test_experiment_preservation.py`

**Interfaces:**
- Consumes: exact workflow snapshots and SHA-256 values from acceptance run `continuation-acceptance-20260731-065935`.
- Produces: immutable, auditable experiment evidence with no production routing effect.

- [ ] **Step 1: Write the failing preservation test**

```python
class StyleConversionPreservationTests(unittest.TestCase):
    def test_manifest_binds_all_frozen_workflows_to_exact_sha256(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for entry in manifest["workflows"]:
            path = EXPERIMENT_ROOT / entry["relative_path"]
            self.assertTrue(path.is_file())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])

    def test_frozen_workflows_contain_no_secret_markers(self):
        text = "\n".join(path.read_text(encoding="utf-8") for path in WORKFLOWS)
        for marker in (
            "discord.com/api/webhooks",
            "api_key",
            "password",
            "client_secret",
            "refresh_token",
            "github_token",
        ):
            self.assertNotIn(marker, text.casefold())
```

- [ ] **Step 2: Run the focused test and verify it fails because evidence files do not exist**

Run:

```powershell
python -m unittest discover -s tests -p "test_experiment_preservation.py" -v
```

Expected: FAIL on missing `manifest.json` or frozen workflow snapshots.

- [ ] **Step 3: Add the exact workflow snapshots and manifest**

The manifest must identify:

```json
{
  "schema_version": 1,
  "source_run_id": "continuation-acceptance-20260731-065935",
  "source_job_id": "20260730-0217",
  "source_scene_id": 1,
  "production_status": "rejected_for_style_continuation",
  "methods": {
    "decoded_17_frame": "preserved_live_action_conversion_lead",
    "latent_overlap": "preserved_semirealistic_3d_conversion_lead"
  },
  "workflows": [],
  "raw_media": []
}
```

Each workflow entry contains `relative_path` and exact SHA-256. Each raw-media entry contains case name, D-drive audit path, and SHA-256 from `run.json`. The README states the precise graph behavior, model/LoRA/sigma settings, observed transformations, and that one test case is a reproducible lead rather than a general quality claim.

- [ ] **Step 4: Run the preservation test**

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add experiments/ltx23-style-conversion tests/test_experiment_preservation.py
git commit -m "docs: preserve LTX style conversion discoveries"
```

---

### Task 2: Add exact assembled-seam proxy service

**Files:**
- Modify: `tenminvideomaker/acceptance_review.py`
- Modify: `tests/test_acceptance_review.py`

**Interfaces:**
- Consumes: `AcceptanceReviewService._load_run`, `_raw_video_path`, `_safe_path`, `_proxy_lock`.
- Produces: `AcceptanceReviewService.assembled_proxy_path(run_id: str, case_name: str) -> Path`.
- Produces per-case review fields: `assembled_video_url` and `assembly`.

- [ ] **Step 1: Write failing boundary and FFmpeg tests**

```python
def test_review_document_describes_production_faithful_assembly(self):
    document = self.service.review_document(RUN_ID)
    self.assertEqual(
        document["cases"]["single_frame"]["assembly"],
        {
            "base_end_frame": 120,
            "continuation_start_frame": 1,
            "dropped_continuation_frames": [0, 0],
            "summary": "Base 0–120, then continuation 1 onward.",
        },
    )
    self.assertEqual(document["cases"]["decoded_17_frame"]["assembly"]["base_end_frame"], 112)
    self.assertEqual(document["cases"]["decoded_17_frame"]["assembly"]["continuation_start_frame"], 17)
    self.assertEqual(document["cases"]["latent_overlap"]["assembly"]["continuation_start_frame"], 25)

def test_assembled_proxy_trims_overlap_and_concats_atomically(self):
    with patch("tenminvideomaker.acceptance_review.subprocess.run", side_effect=fake_run) as run:
        path = self.service.assembled_proxy_path(RUN_ID, "single_frame")
    command = run.call_args.args[0]
    filter_graph = command[command.index("-filter_complex") + 1]
    self.assertIn("trim=start_frame=0:end_frame=121", filter_graph)
    self.assertIn("trim=start_frame=1", filter_graph)
    self.assertIn("concat=n=2:v=1:a=0", filter_graph)
    self.assertEqual(path.name, "assembled-single_frame.mp4")
```

Add rejection tests for unknown cases and failed FFmpeg output.

- [ ] **Step 2: Run focused service tests and verify expected failures**

Run:

```powershell
python -m unittest discover -s tests -p "test_acceptance_review.py" -v
```

Expected: FAIL because assembly metadata and `assembled_proxy_path` do not exist.

- [ ] **Step 3: Implement assembly metadata and proxy**

Add immutable case values:

```python
"single_frame": {
    "assembly": {"base_end_frame": 120, "continuation_start_frame": 1}
}
"decoded_17_frame": {
    "assembly": {"base_end_frame": 112, "continuation_start_frame": 17}
}
"latent_overlap": {
    "assembly": {"base_end_frame": 120, "continuation_start_frame": 25}
}
```

Build this FFmpeg filter:

```text
[0:v]trim=start_frame=0:end_frame=<base_end+1>,setpts=PTS-STARTPTS[base];
[1:v]trim=start_frame=<continuation_start>,setpts=PTS-STARTPTS[continuation];
[base][continuation]concat=n=2:v=1:a=0[outv]
```

Map `[outv]`, encode `libx264`, `crf=18`, `yuv420p`, `+faststart`, and write atomically to `review/assembled-<case>.mp4`. Reuse a non-empty cached file.

- [ ] **Step 4: Run focused service tests**

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add tenminvideomaker/acceptance_review.py tests/test_acceptance_review.py
git commit -m "feat: build exact continuation seam previews"
```

---

### Task 3: Expose assembled preview through GUI

**Files:**
- Modify: `tenminvideomaker/gui_app.py`
- Modify: `web/acceptance-review.html`
- Modify: `web/acceptance-review.js`
- Modify: `web/acceptance-review.css`
- Modify: `tests/test_gui_app.py`

**Interfaces:**
- Consumes: `AcceptanceReviewService.assembled_proxy_path`.
- Produces: `GET /api/acceptance-runs/{run_id}/assembled/{case_name}`.
- Produces DOM video `#assembled-video`.

- [ ] **Step 1: Write failing route and static-UI tests**

```python
self.assertEqual(
    client.get(f"/api/acceptance-runs/{run_id}/assembled/single_frame")
    .headers["content-type"].split(";", 1)[0],
    "video/mp4",
)
```

Static assertions:

```python
self.assertRegex(
    markup,
    r'<video[^>]*id="assembled-video"[^>]*controls[^>]*playsinline[^>]*webkit-playsinline',
)
self.assertIn("assembled_video_url", script)
self.assertIn("assembly.summary", script)
self.assertIn(".assembled-video-card", styles)
```

- [ ] **Step 2: Run GUI tests and verify expected failures**

Run the embedded ComfyUI Python `test_gui_app.py` suite.

Expected: FAIL on missing route/player.

- [ ] **Step 3: Add the authenticated route**

Register before the static mount:

```python
@app.get("/api/acceptance-runs/{run_id}/assembled/{case_name}")
def acceptance_assembled(run_id: str, case_name: str) -> FileResponse:
    try:
        path = acceptance_review.assembled_proxy_path(run_id, case_name)
    except AcceptanceReviewProxyError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except AcceptanceReviewError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return FileResponse(path, media_type="video/mp4", filename=path.name)
```

- [ ] **Step 4: Add assembled player without removing current views**

Place one full-width video card after `.comparison-grid`. On case selection,
load `caseDocument.assembled_video_url` and render
`caseDocument.assembly.summary`. Keep `controls`, `playsinline`,
`webkit-playsinline`, intrinsic sizing, and mobile width `100%`.

- [ ] **Step 5: Run GUI tests**

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add tenminvideomaker/gui_app.py web/acceptance-review.* tests/test_gui_app.py
git commit -m "feat: show assembled continuation seams"
```

---

### Task 4: Persist the human decision record

**Files:**
- Create: `docs/research/continuation_acceptance_20260731_human_review.md`
- Modify: `docs/user-guide.md`
- Modify: `AI_DEVELOPMENT_RULES.md`
- Create outside Git: `D:\LTX_Supervisor_Storage\acceptance\continuation-acceptance-20260731-065935\human-review.json`
- Copy outside Git: three censored screenshots into the run's `human-review\` directory.

**Interfaces:**
- Consumes: owner's five-point findings and censored screenshots.
- Produces: durable human evidence without changing `run.json` or rollout state.

- [ ] **Step 1: Add the structured D-drive decision**

The record contains:

```json
{
  "schema_version": 1,
  "run_id": "continuation-acceptance-20260731-065935",
  "reviewer": "project owner",
  "production_decision": "no_method_approved",
  "next_baseline": "single_frame",
  "methods": {
    "single_frame": {"production": "conditional_baseline", "style_conversion": null},
    "decoded_17_frame": {"production": "rejected", "style_conversion": "live_action_lead"},
    "latent_overlap": {"production": "rejected", "style_conversion": "semirealistic_3d_lead"}
  }
}
```

Include full prose observations, mechanical metrics, screenshot filenames, and
the explicit statement that this record does not approve automatic rollout.

- [ ] **Step 2: Preserve the censored screenshots**

Copy only:

- `codex-clipboard-d0038428-47e7-41ae-b35d-faded91f46f8.png` as `single_frame.png`;
- `codex-clipboard-eb35ddf0-0102-488d-a577-5139fb877b9f.png` as `decoded_17_frame.png`;
- `codex-clipboard-fe1c843c-fa13-4586-8a18-0d4d2e19789e.png` as `latent_overlap.png`.

Do not copy uncensored source media into Git.

- [ ] **Step 3: Document interpretation and UI usage**

Explain that the assembled player removes conditioning/overlap frames and that
the current production decision is no approval. Document both experimental
leads as isolated future work.

- [ ] **Step 4: Commit**

```powershell
git add docs/research/continuation_acceptance_20260731_human_review.md docs/user-guide.md AI_DEVELOPMENT_RULES.md
git commit -m "docs: record continuation human review"
```

---

### Task 5: Full verification and publication

**Files:**
- Verify all changed files.

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: tested, committed, pushed main branch.

- [ ] **Step 1: Run focused tests**

```powershell
python -m unittest discover -s tests -p "test_experiment_preservation.py" -v
python -m unittest discover -s tests -p "test_acceptance_review.py" -v
```

Run the embedded ComfyUI Python `test_gui_app.py` suite.

- [ ] **Step 2: Run full tests and syntax checks**

```powershell
python -m unittest discover -s tests -v
python -m compileall -q tenminvideomaker scripts tests
git diff --check
```

- [ ] **Step 3: Perform a read-only runtime smoke test**

Start only the review-only server on a temporary unused port, request the
assembled endpoint, verify HTTP 200 and H.264 output, then stop only that
temporary process. Do not start the supervisor or submit a ComfyUI prompt.

- [ ] **Step 4: Verify repository scope and push**

```powershell
git status --short --branch
git log --oneline -8
git push origin main
```

Confirm no production state, approval manifest, active queue, or shared ComfyUI
resource was changed.
