# Continuation Acceptance Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add secure, mobile-friendly GUI showing exact LTX continuation boundaries without File Explorer.

**Architecture:** Add focused acceptance-review service. It reads completed acceptance `run.json`, validates every raw path remains under project D-drive storage root, then creates reusable unwatermarked H.264 review proxies. Extend existing FastAPI app with read-only review APIs and standalone static review page. Normal supervisor state, raw FFV1/FLAC files, and production rollout stay unchanged.

**Tech Stack:** Python 3.12, FastAPI, stdlib `json`/`subprocess`/`pathlib`, FFmpeg, existing vanilla HTML/CSS/JavaScript GUI, `unittest`.

## Global Constraints

- Persist review assets beneath `D:\LTX_Supervisor_Storage\acceptance\<run_id>\review\`.
- Accept only completed `awaiting_human_review` acceptance runs and storage-root-contained artifacts.
- Never queue ComfyUI work, alter supervisor state, approve rollout, watermark, overwrite, or delete raw media.
- Use browser-safe H.264/AAC MP4 proxies only; retain raw FFV1/FLAC windows unchanged.
- Preserve LAN Basic-auth middleware coverage for every new API and static review page.
- Desktop and mobile need native HTML5 video controls, `playsinline`, and fullscreen.

---

### Task 1: Acceptance review domain service

**Files:**
- Create: `tenminvideomaker/acceptance_review.py`
- Create: `tests/test_acceptance_review.py`

**Interfaces:**
- Consumes: `StorageLayout.root`, `StorageLayout.job_root(run_id)`, completed acceptance `run.json`, and FFmpeg binary string.
- Produces: `AcceptanceReviewService(storage, ffmpeg_command="ffmpeg")`; `list_runs() -> list[dict[str, object]]`; `review_document(run_id: str) -> dict[str, object]`; `review_proxy_path(run_id: str, role: str) -> Path`; `still_path(run_id: str, case_name: str, asset_name: str) -> Path`.

- [ ] **Step 1: Write failing domain tests**

```python
def test_review_document_labels_single_frame_boundary_and_safe_urls(self):
    service = AcceptanceReviewService(self.storage)
    document = service.review_document("continuation-acceptance-20260731-065935")
    case = document["cases"]["single_frame"]
    self.assertEqual(case["boundary"], {"left": [119, 120], "right": [0, 1]})
    self.assertIn("/api/acceptance-runs/", case["video_url"])

def test_proxy_transcode_is_unwatermarked_atomic_and_reused(self):
    with patch("subprocess.run") as run:
        proxy = service.review_proxy_path(RUN_ID, "base")
    self.assertIn("-c:v", run.call_args.args[0])
    self.assertIn("libx264", run.call_args.args[0])
    self.assertTrue(proxy.name.endswith(".mp4"))
```

Include rejection tests: traversal run ID, raw path outside `storage.root`, missing source video, invalid/missing acceptance document, unknown role/case/still, and FFmpeg exit without temporary output.

- [ ] **Step 2: Run focused tests; verify failure**

Run: `python -m unittest tests.test_acceptance_review -v`

Expected: import failure because `tenminvideomaker.acceptance_review` does not exist.

- [ ] **Step 3: Implement `AcceptanceReviewService`**

```python
class AcceptanceReviewService:
    def review_document(self, run_id: str) -> dict[str, object]: ...
    def review_proxy_path(self, run_id: str, role: str) -> Path: ...
    def still_path(self, run_id: str, case_name: str, asset_name: str) -> Path: ...
```

Validate `run_id` against `continuation-acceptance-YYYYMMDD-HHMMSS`; resolve every candidate and require `candidate.relative_to(self.storage.root.resolve())`. Require `run.json["state"] == "awaiting_human_review"`, known cases, `common_base` for base role, and each source `raw_video_path` from `run.json`.

Build semantic documents only: case label, left/right frame boundary, explanatory text, video URL, and still URL. Do not expose filesystem paths or raw JSON.

Create proxy atomically at `acceptance/<run_id>/review/<role>.mp4`: write same-directory temporary MP4, then replace destination only after FFmpeg succeeds and output is non-empty. Use `-map 0:v:0 -map 0:a? -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p -c:a aac -b:a 160k -movflags +faststart`. Reuse existing non-empty proxy. Never touch source.

- [ ] **Step 4: Run focused tests; verify pass**

Run: `python -m unittest tests.test_acceptance_review -v`

Expected: all acceptance-review service tests pass.

- [ ] **Step 5: Commit domain service**

```bash
git add tenminvideomaker/acceptance_review.py tests/test_acceptance_review.py
git commit -m "feat: add acceptance review service"
```

### Task 2: Secure FastAPI review APIs

**Files:**
- Modify: `tenminvideomaker/gui_app.py: imports and create_gui_app`
- Modify: `tests/test_gui_app.py: FastAPI route tests`

**Interfaces:**
- Consumes: `AcceptanceReviewService`, existing `create_gui_app` LAN middleware, FastAPI `FileResponse`.
- Produces: `GET /api/acceptance-runs`; `GET /api/acceptance-runs/{run_id}`; `GET /api/acceptance-runs/{run_id}/media/{role}`; `GET /api/acceptance-runs/{run_id}/stills/{case_name}/{asset_name}`.

- [ ] **Step 1: Write failing route tests**

```python
response = client.get(f"/api/acceptance-runs/{RUN_ID}")
self.assertEqual(response.status_code, 200)
self.assertEqual(response.json()["cases"]["latent_overlap"]["boundary"]["right"], [24, 25])

blocked = client.get("/api/acceptance-runs/../../jobs/other")
self.assertEqual(blocked.status_code, 404)
```

Mock only service proxy creation for media tests. Assert valid response uses `video/mp4`, still response uses `image/png`, missing service data maps to 404, proxy/FFmpeg error maps to 503. Add LAN-auth coverage for one new API.

- [ ] **Step 2: Run focused route tests; verify failure**

Run: `python -m unittest tests.test_gui_app.GuiAppTests -v`

Expected: 404 because acceptance routes do not exist.

- [ ] **Step 3: Add routes before static-file mount**

```python
@app.get("/api/acceptance-runs/{run_id}/media/{role}")
def acceptance_media(run_id: str, role: str) -> FileResponse:
    return FileResponse(service.review_proxy_path(run_id, role), media_type="video/mp4")
```

Instantiate one `AcceptanceReviewService(storage)` inside `create_gui_app`. Translate validation errors to 404 and proxy/FFmpeg failures to 503. Declare routes before `app.mount("/", StaticFiles(...))`; reuse global LAN middleware without bypass.

- [ ] **Step 4: Run focused route tests; verify pass**

Run: `python -m unittest tests.test_gui_app.GuiAppTests -v`

Expected: all GUI route tests pass.

- [ ] **Step 5: Commit FastAPI routes**

```bash
git add tenminvideomaker/gui_app.py tests/test_gui_app.py
git commit -m "feat: expose acceptance review media"
```

### Task 3: Responsive review page

**Files:**
- Create: `web/acceptance-review.html`
- Create: `web/acceptance-review.js`
- Create: `web/acceptance-review.css`
- Modify: `web/index.html: add continuation review link`
- Modify: `web/app.js: route link to latest available acceptance run`
- Modify: `tests/test_gui_app.py: static-review markup assertions`

**Interfaces:**
- Consumes: `GET /api/acceptance-runs`, semantic review document URLs, boundary labels.
- Produces: review page at `/acceptance-review.html`, defaulting to newest eligible run or explicit `?run=<run_id>`.

- [ ] **Step 1: Write failing static-page assertions**

```python
markup = (ROOT / "web" / "acceptance-review.html").read_text(encoding="utf-8")
self.assertRegex(markup, r'<video[^>]*controls[^>]*playsinline[^>]*webkit-playsinline')
self.assertIn('id="show-seam"', markup)
self.assertIn('id="visual-checklist"', markup)
```

Also assert CSS uses two-video grid desktop, one column below `760px`, and `video { width: 100%; }` with no control-blocking overlay.

- [ ] **Step 2: Run static-page tests; verify failure**

Run: `python -m unittest tests.test_gui_app.GuiStaticTests -v`

Expected: file-not-found for `web/acceptance-review.html`.

- [ ] **Step 3: Build page and interactions**

Create accessible markup: run selector, case selector, base/continuation video cards, frame labels, seam still cards, checklist, status/error text, back link.

JavaScript loads newest eligible run when query absent, then selected case. Assign only returned server URLs to video/image sources. `Show seam` pauses both videos and seeks left/right to boundary frame divided by 24. Do not autoplay or synchronize playback. Put exact visual instructions beside each video: identity, anatomy, hand/object contact, camera direction/velocity, freeze/jump, visible seam.

CSS follows current dark GUI. Desktop uses equal cards. Mobile stacks controls, labels, cards, stills. Video controls stay tappable; native fullscreen remains available.

Add normal-GUI header link opening review page on same origin. It must not affect job/remake controls.

- [ ] **Step 4: Run static-page tests; verify pass**

Run: `python -m unittest tests.test_gui_app.GuiStaticTests -v`

Expected: all static-review assertions pass.

- [ ] **Step 5: Commit review page**

```bash
git add web/acceptance-review.html web/acceptance-review.js web/acceptance-review.css web/index.html web/app.js tests/test_gui_app.py
git commit -m "feat: add continuation acceptance review UI"
```

### Task 4: Documentation and full verification

**Files:**
- Modify: `docs/user-guide.md: continuation acceptance review instructions`
- Modify: `AI_DEVELOPMENT_RULES.md: implementation decision and test evidence`

**Interfaces:**
- Consumes: completed acceptance run and launched supervisor GUI.
- Produces: owner instructions for reviewing without File Explorer.

- [ ] **Step 1: Document owner workflow**

State owner opens GUI review link, selects case, uses `Show seam`, watches base then case, chooses method only after visual review. State proxies are unwatermarked H.264 MP4 under acceptance run; raw FFV1/FLAC clips remain unchanged; page does not approve or alter rollout.

- [ ] **Step 2: Run full verification**

```bash
python -m compileall -q tenminvideomaker scripts tests
python -m unittest discover -s tests -v
python scripts/validate_continuation_workflows.py
git diff --check
```

Expected: exit 0; all tests pass; live node-contract validation queues no prompt; no whitespace errors.

- [ ] **Step 3: Browser-safe smoke test, no ComfyUI render**

Launch test FastAPI app against temporary storage with fixture raw MP4 and mocked FFmpeg. Request review document, media, still endpoints. Confirm video `Content-Type: video/mp4`, image `image/png`, traversal returns 404. Do not start supervisor or queue ComfyUI prompt.

- [ ] **Step 4: Commit documentation**

```bash
git add docs/user-guide.md AI_DEVELOPMENT_RULES.md
git commit -m "docs: explain continuation review UI"
```

- [ ] **Step 5: Push intended branch only**

```bash
git push origin main
git status --short --branch
```

Expected: `main...origin/main` with no working-tree changes.
