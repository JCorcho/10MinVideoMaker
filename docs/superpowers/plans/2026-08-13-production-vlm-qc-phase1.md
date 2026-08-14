# Production VLM QC Phase 1 Implementation Plan

> Implement only on `hotfix/production-vlm-qc-phase1`. Use test-first changes, `apply_patch`, focused commits, and the existing production services. Do not modify the protected v3 worktree or standalone lab.

**Goal:** Add an opt-in, crash-safe, serialized production QC lane that judges existing post-two-pass scene videos, permits one deterministic A1 seed retry and one constrained B1 I2V-prompt repair, and requires human approval for PASS by default.

**Architecture:** Extend normal scene revisions and SQLite state with immutable candidate/evaluation/repair/human-decision evidence. A deterministic controller alternates between existing ComfyUI generation batches and an owned local llama.cpp Qwen epoch. Vision judging and text-only repair planning use separate fresh requests and prompts behind a backend interface. Existing delivery and final assembly remain blocked until every scene has an accepted candidate.

**Tech stack:** Python 3.12, unittest, SQLite, FastAPI, vanilla JavaScript, stdlib HTTP/subprocess, FFmpeg/FFprobe, existing ComfyUI API, llama.cpp OpenAI-compatible loopback API, Qwen3.6 27B IQ3_M plus FP16 projector.

**Same-day scope budget:** Tasks 1-7 are the production implementation path and are intended as one focused same-day change, approximately 8-10 implementation/test hours with agent assistance. Task 8 documentation and one-scene readiness preflight follow in the same branch. If GPU identity, llama request freshness, or baseline-equivalent finalization cannot be proven within that window, stop before a real canary; do not expand into NVFP4 staging, concurrency, or another backend.

## Fixed Phase-1 policy

- Defaults: `quality_control_enabled=false`, `auto_advance_pass=false`.
- Candidate video: existing revision-facing 768x1344 `video.mp4` after both LTX passes.
- Judge: 2 FPS, four chronological images, `--image-min-tokens 1024`.
- Strong evidence: severity >= 3, error confidence >= 0.85, non-empty concrete evidence.
- Automatic early FAIL: two distinct strong windows.
- Lone strong window after full coverage: one shifted overlapping fresh confirmation; unconfirmed result is UNCERTAIN.
- ORIGINAL FAIL -> one A1 new-seed retry with no prompt change.
- A1 FAIL -> one B1 text-only I2V-prompt repair plus new deterministic seed.
- UNCERTAIN at any tier -> HOLD_FOR_REVIEW.
- B1 not accepted -> HOLD_FOR_REVIEW.
- PASS -> human approval queue unless `auto_advance_pass=true`.
- Human REJECT/HOLD creates no additional automatic retry.
- No QC-enabled final may omit an unaccepted scene.

## Global constraints

- Do not install or update dependencies.
- Do not start ComfyUI, the supervisor, or a model during unit/integration implementation tests.
- No unit/integration test may write to `D:\LTX_Supervisor_Storage`.
- Do not copy/import code from `C:\AI\LTX23-VLM-Video-QC-Lab`; port the reviewed behavior into production-owned modules.
- Do not modify or run tests in `C:\AI\GitWorktrees\10MinVideoMaker-recovery-ltx23-latent-overlap-v3`.
- Keep model outputs as data. No model-originated code, SQL, shell, path, node type, or arbitrary configuration may execute.
- Every schema migration must be additive and idempotent against a copied temporary SQLite fixture before production use.
- Use the existing `scene_revisions`, `validate_scene_edit()`, `PipelineSupervisor.render_i2v_scene()`, `StorageLayout`, and controller ownership instead of creating a parallel generation system.

---

### Task 1: Add strict QC configuration, enums, schemas, and deterministic seeds

**Files:**

- Create: `tenminvideomaker/qc_config.py`
- Create: `tenminvideomaker/qc_contracts.py`
- Create: `tests/test_qc_config.py`
- Create: `tests/test_qc_contracts.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class QualityControlSettings:
    quality_control_enabled: bool = False
    auto_advance_pass: bool = False

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> "QualityControlSettings": ...

class QcDecision(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"

class QcTier(StrEnum):
    ORIGINAL = "ORIGINAL"
    A1 = "A1"
    B1 = "B1"

class QcCandidateState(StrEnum):
    PENDING_GENERATION = "PENDING_GENERATION"
    GENERATING = "GENERATING"
    PENDING_QC = "PENDING_QC"
    QC_RUNNING = "QC_RUNNING"
    PASS_PENDING_HUMAN = "PASS_PENDING_HUMAN"
    ACCEPTED = "ACCEPTED"
    HOLD_FOR_REVIEW = "HOLD_FOR_REVIEW"
    SUPERSEDED = "SUPERSEDED"

class QcHumanDecision(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    HOLD = "HOLD"

@dataclass(frozen=True)
class QcEvidencePolicy: ...

@dataclass(frozen=True)
class QcError: ...

@dataclass(frozen=True)
class JudgeResponse: ...

@dataclass(frozen=True)
class JudgeWindowResult: ...

@dataclass(frozen=True)
class NormalizedEvaluation: ...

def parse_judge_response(raw_text: str) -> JudgeResponse: ...
def is_strong_evidence(error: QcError, policy: QcEvidencePolicy) -> bool: ...
def normalize_window_results(
    windows: Sequence[JudgeWindowResult],
    confirmation: JudgeWindowResult | None,
    policy: QcEvidencePolicy,
) -> NormalizedEvaluation: ...
def derive_retry_seed(
    *, job_id: str, scene_id: int, source_revision: int,
    original_seed: int, tier: QcTier
) -> int: ...
```

- [ ] Write failing config tests for default-off QC, default human canary, strict boolean parsing, and `auto_advance_pass=true` without a schema change.
- [ ] Write failing parser tests for valid PASS/FAIL/UNCERTAIN, malformed JSON, refusal text, invalid enum, missing fields, out-of-range confidence/severity, invalid timestamps, PASS with errors, and FAIL without evidence.
- [ ] Write failing normalization tests proving two-window early FAIL, a bare FAIL/refusal is insufficient, one strong window becomes UNCERTAIN unless a shifted confirmation independently qualifies, and UNCERTAIN never normalizes to PASS.
- [ ] Write failing seed tests proving A1 is deterministic, unsigned-64-bit, unique from original, distinct from B1, and stable across restart.
- [ ] Implement the smallest strict dataclasses/enums/parsers and canonical JSON/hash helpers needed by those tests.
- [ ] Run: `python -m unittest tests.test_qc_config tests.test_qc_contracts -v`.

---

### Task 2: Extract the headless judge backend and owned llama.cpp lifecycle

**Files:**

- Create: `tenminvideomaker/qc_backend.py`
- Create: `tenminvideomaker/qc_llama.py`
- Create: `tenminvideomaker/qc_video.py`
- Create: `prompts/production_ltx_video_qc_v1.txt`
- Create: `prompts/production_i2v_repair_v1.txt`
- Create: `tests/test_qc_backend.py`
- Create: `tests/test_qc_llama.py`
- Create: `tests/test_qc_video.py`

**Interfaces:**

```python
class QcBackend(Protocol):
    def start(self) -> BackendIdentity: ...
    def evaluate(self, request: VisionJudgeRequest) -> VisionJudgeEvaluation: ...
    def plan_repair(self, request: RepairPlannerRequest) -> RepairPlannerResponse: ...
    def close(self) -> None: ...

@dataclass(frozen=True)
class BackendIdentity: ...

@dataclass(frozen=True)
class VisionJudgeRequest: ...

@dataclass(frozen=True)
class VisionJudgeEvaluation: ...

@dataclass(frozen=True)
class RepairPlannerRequest: ...

@dataclass(frozen=True)
class RepairPlannerResponse: ...

class LlamaCppProcess:
    def start(self) -> BackendIdentity: ...
    def close(self) -> None: ...

def sample_video_frames(
    video_path: Path,
    *, target_fps: float,
    ffprobe_command: str,
    ffmpeg_command: str,
    temporary_root: Path,
) -> SampledVideo: ...

def chronological_windows(
    sampled: SampledVideo, *, frame_count: int = 4
) -> tuple[QcWindow, ...]: ...
```

- [ ] Write failing pure tests for deterministic 2-FPS indices/timestamps, chronological four-frame windows, short final windows, shifted confirmation bounds, and truthful unique-versus-confirmation frame accounting.
- [ ] Use mocked FFprobe/FFmpeg runners and temporary roots; add no PyAV/OpenCV production dependency.
- [ ] Write failing request-shape tests proving judge messages contain only the QC rubric, current images, timestamps, and window number; assert absence of `ORIGINAL`, `A1`, `B1`, prior decisions, repair summaries, human expectations, and all `tools`/`tool_choice` fields.
- [ ] Write failing planner request tests proving it is text-only, receives normalized QC JSON and explicit mutable/locked fields, uses a separate system prompt, and has no tools.
- [ ] Write failing lifecycle tests for fixed list arguments, loopback-only bind, model/projector paths, context, one parallel slot, image-min-tokens 1024, no session/prompt-cache persistence, hidden Windows process, readiness timeout, owned-PID-only shutdown, graceful terminate then bounded kill, port-close verification, and log paths below `StorageLayout.logs_root`.
- [ ] Require configured GPU UUID/name; mock `nvidia-smi` and fail closed when the physical 4080 identity or llama startup telemetry does not match. Do not select by ordinal alone.
- [ ] Inspect the installed llama.cpp 2.28.2 help/behavior and use its supported cache/session controls to ensure each completed request's KV state is cleared while weights remain loaded. Add a sequential-request smoke assertion; if fresh KV cannot be proven, fail the canary preflight instead of weakening the boundary.
- [ ] Implement stdlib OpenAI-compatible HTTP requests with temperature 0, no conversation/session persistence, one request context per window/planner call, and no user-provided endpoint.
- [ ] Persist backend executable/version/hash, model/projector hashes, effective args, GPU identity, and prompt hashes in `BackendIdentity`.
- [ ] Run: `python -m unittest tests.test_qc_video tests.test_qc_backend tests.test_qc_llama -v`.

---

### Task 3: Add revision-local evidence paths and additive SQLite persistence

**Files:**

- Modify: `tenminvideomaker/storage.py`
- Modify: `tenminvideomaker/state_store.py`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_state_store.py`

**New records/tables:**

- `QcCandidateRecord` / `qc_candidates`
- `QcEvaluationRecord` / `qc_evaluations`
- `QcRepairRecord` / `qc_repairs`
- `QcHumanDecisionRecord` / `qc_human_decisions`

**Storage methods:**

```python
def qc_evaluation_root(
    self, job_id: str, scene_id: int, revision: int, evaluation_id: str
) -> Path: ...
def qc_evaluation_manifest_path(...) -> Path: ...
def qc_repair_manifest_path(...) -> Path: ...
```

**Store methods:**

```python
def ensure_qc_candidate(...) -> QcCandidateRecord: ...
def begin_qc_evaluation(...) -> QcEvaluationRecord: ...
def complete_qc_evaluation(...) -> QcEvaluationRecord: ...
def record_qc_repair(...) -> QcRepairRecord: ...
def record_qc_human_decision(...) -> QcHumanDecisionRecord: ...
def qc_candidates(self, job_id: str, scene_id: int | None = None) -> tuple[QcCandidateRecord, ...]: ...
def promote_accepted_qc_candidate(self, candidate_id: str) -> None: ...
```

- [ ] Write failing tests for additive/idempotent initialization against a temporary pre-QC database.
- [ ] Write failing tests for candidate uniqueness `(job, scene, tier)`, original revision binding, A1/B1 revision binding, unsigned seeds as text, parent identity, and immutable source video hashes.
- [ ] Persist and test a bounded infrastructure-failure count so one worker crash retries the same evaluation and a second crash holds without creating a new candidate.
- [ ] Write failing tests for evaluator/model/projector/config/prompt versions, sampling/window config, raw/normalized results, suspect windows, next action, and timestamps being persisted.
- [ ] Write failing tests for canonical evaluation idempotency after restart and rejection of the same candidate ID with changed video/config hashes.
- [ ] Write failing tests for append-only repair and human decisions, terminal human-decision idempotency, and promotion only after ACCEPTED.
- [ ] Add atomic immutable evidence writers: an existing identical document is reusable; different bytes at the same identity are an error. Store manifest SHA-256 in SQLite.
- [ ] Verify every path is below the candidate revision and rejects traversal.
- [ ] Run: `python -m unittest tests.test_storage tests.test_state_store -v`.

---

### Task 4: Implement A1 and constrained B1 document construction

**Files:**

- Create: `tenminvideomaker/qc_repair.py`
- Create: `tests/test_qc_repair.py`
- Reuse: `tenminvideomaker/review.py`

**Interfaces:**

```python
def build_a1_document(
    source_document: Mapping[str, Any], *, seed: int
) -> Mapping[str, Any]: ...

def parse_and_validate_b1_patch(
    raw_text: str,
    *, source_document: Mapping[str, Any],
    required_seed: int,
    repair_input_hash: str,
    current_candidate_hash: str,
) -> ValidatedRepairPatch: ...

def apply_b1_patch(
    original_job: JobPayload,
    scene_id: int,
    source_document: Mapping[str, Any],
    patch: ValidatedRepairPatch,
) -> ValidatedSceneEdit: ...
```

- [ ] Write failing A1 tests proving exactly one retry, unique deterministic seed, byte-equivalent I2V prompt, and no other semantic/document mutation.
- [ ] Write failing B1 tests proving only `i2v.prompt` and `i2v.seed` can change; negative/safety prompt, LoRAs, T2I, character/job facts, duration, continuation, segments, pass settings, spatial upscaler, and production profile remain locked.
- [ ] Write failing tests for stale input hash, stale candidate hash, duplicate/current prompt, prior repair-summary duplicate, duplicate seed, wrong required seed, unknown fields, refusal/malformed output, and empty prompt.
- [ ] Build the full candidate document by deep copy, alter only the two allowed leaves, run `validate_scene_edit()`, then deep-diff the validated document against the source allowlist.
- [ ] Persist accepted and rejected proposals; rejected B1 planning goes directly to HOLD and never invokes the planner again.
- [ ] Run: `python -m unittest tests.test_qc_repair -v`.

---

### Task 5: Add deterministic QC controller and serialized epochs

**Files:**

- Create: `tenminvideomaker/qc_controller.py`
- Modify: `tenminvideomaker/state_store.py`
- Modify: `tenminvideomaker/supervisor.py`
- Modify: `tenminvideomaker/gui_service.py`
- Modify: `scripts/run_supervisor.py`
- Create: `tests/test_qc_controller.py`
- Modify: `tests/test_supervisor.py`
- Modify: `tests/test_gui_service.py`

**Interfaces:**

```python
class ProductionQcController:
    def advance_job(self, job: JobPayload) -> QcAdvanceResult: ...
    def record_human_decision(
        self, *, candidate_id: str, decision: QcHumanDecision,
        note: str | None = None
    ) -> QcCandidateRecord: ...

@dataclass(frozen=True)
class QcAdvanceResult: ...

def _finalize_accepted_job(
    self, job: JobPayload, scene_by_id: Mapping[int, SceneSpec]
) -> None: ...
```

- [ ] Add `PipelineState.RUNNING_QC` and `PipelineState.AWAITING_QC_REVIEW`; make restart/recovery and GUI status recognize them without treating them as ComfyUI prompt ownership.
- [ ] First write failing controller transition tests for every row in the design transition table.
- [ ] Write failing tests proving PASS waits for human approval by default, `auto_advance_pass=true` accepts, and UNCERTAIN never accepts.
- [ ] Write failing budget tests proving A1 max one, B1 max one, B1 failure/uncertainty holds, human rejection holds, and no state can create C/D or loop back.
- [ ] Write failing blind-judge tests that inspect the actual backend request after ORIGINAL/A1/B1 creation and prove repair tier/history is absent.
- [ ] Write failing restart/idempotency tests for crashes before evaluation insert, after immutable evidence write, after evaluation completion, after candidate generation, and after human approval but before finalization.
- [ ] Write failing worker-crash tests: first owned llama crash requeues the same candidate once; the second holds with error evidence; no candidate/revision/video is dropped.
- [ ] Write failing resource-order tests: Comfy queue empty -> `/free` -> llama start -> QC/planning -> llama stop/port close -> candidate generation -> `/free`. Assert no simultaneous generation/QC path exists.
- [ ] Refactor current post-I2V delivery/stitch code into `_finalize_accepted_job()` without changing behavior. Reuse it from the baseline path and after QC acceptance.
- [ ] After the original I2V batch and existing release, register revision-1 candidates and enter `RUNNING_QC` only when enabled. Let `advance_job()` create VIDEO_ONLY A1/B1 revisions, validate their documents, resolve existing scene assets, call `render_i2v_scene()`, and batch all pending candidate generations before one release.
- [ ] During an A1-failure QC epoch, call the text-only planner before unloading Qwen when practical. Never judge B1 until its later generation completes and a fresh judge request is built.
- [ ] Block delivery/finalization until every scene has one accepted valid candidate. When accepted, atomically promote that revision's video into the current scene pointer while preserving all revisions/evidence.
- [ ] If QC is enabled and any scene has no valid candidate, hold the job; do not use current partial assembly to omit it.
- [ ] Implement kill-switch recovery for an already-held QC job: preserve evidence, select revision 1, and execute the pre-feature finalization path.
- [ ] Run: `python -m unittest tests.test_qc_controller tests.test_supervisor tests.test_gui_service -v`.

---

### Task 6: Add minimal human canary routes and scene UI

**Files:**

- Modify: `tenminvideomaker/gui_app.py`
- Modify: `tenminvideomaker/gui_service.py`
- Modify: `web/index.html`
- Modify: `web/app.js`
- Modify: `web/styles.css`
- Modify: `tests/test_gui_app.py`
- Modify: `tests/test_gui_service.py`

**Routes:**

```text
GET  /api/jobs/{job_id}/scenes/{scene_id}/qc
POST /api/jobs/{job_id}/scenes/{scene_id}/qc/{candidate_id}/decision
```

Decision request body:

```json
{"decision":"APPROVE|REJECT|HOLD","note":"optional local review note"}
```

- [ ] Write failing API tests for job/scene/candidate identity, current-state conflicts, invalid decisions, duplicate terminal decisions, and path-safe evidence serialization.
- [ ] Write failing persistence/service tests proving the human decision and evidence hash are stored before promotion/wake.
- [ ] Extend scene detail with candidate history, current decision, suspect windows, evaluator/config/prompt versions, and existing revision media URL; never return raw filesystem paths.
- [ ] Add the QC panel beneath the existing generated-video view with Approve/Reject/Hold. Show pending/accepted/held state and disable buttons after a terminal decision.
- [ ] Keep the existing desktop/mobile layout and HTML5 video. Do not add a new frontend dependency or a second review application.
- [ ] Assert PASS canary approval wakes the controller, while Reject/Hold cannot queue another retry.
- [ ] Run: `python -m unittest tests.test_gui_app tests.test_gui_service -v`.

---

### Task 7: Prove baseline equivalence, dry-run integration, and lifecycle recovery

**Files:**

- Create: `tests/test_qc_integration.py`
- Modify: `tests/test_configuration.py`
- Modify as needed: focused QC test modules only

- [ ] Build a temporary-root, fake-Comfy, fake-backend end-to-end fixture with two scenes and actual temporary SQLite state.
- [ ] Test kill switch exact baseline behavior: same supervisor method order, same original revision paths, same delivery/assembly decision, no QC/backend calls, and no required QC rows.
- [ ] Test canary path: original PASS does not deliver or stitch until human approval; after approval it finalizes exactly once.
- [ ] Test automatic PASS path with `auto_advance_pass=true`.
- [ ] Test original FAIL -> A1 PASS -> human approval.
- [ ] Test original FAIL -> A1 FAIL -> valid B1 PASS -> human approval.
- [ ] Test UNCERTAIN at ORIGINAL/A1/B1, duplicate B1 rejection, B1 FAIL, human rejection, and missing scene candidate all block final processing.
- [ ] Test evaluator/model/projector/config/prompt versions, raw response, normalized decision, windows, timestamps, repair input/patch status, human decision, and next action are reconstructable after a new store/controller instance.
- [ ] Test launch/termination releases both owned llama and ComfyUI generation resources, including exception paths.
- [ ] Test generation prompt recovery and llama worker-crash recovery without silent candidate loss.
- [ ] Patch `StorageLayout.configured()`/process/HTTP/subprocess in every integration test; assert the real production root never appears in write calls.
- [ ] Run focused QC suite: `python -m unittest discover -s tests -p 'test_qc*.py' -v`.
- [ ] Run full CPU baseline with bytecode disabled: `$env:PYTHONDONTWRITEBYTECODE='1'; python -m unittest discover -s tests -v`.
- [ ] Run: `git diff --check`.

---

### Task 8: Documentation and bounded canary readiness

**Files:**

- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/user-guide.md`
- Modify: `AI_DEVELOPMENT_RULES.md`
- Create: `scripts/run_qc_dry_run.py`
- Create: `tests/test_run_qc_dry_run.py`

- [ ] Document the candidate definition (post-two-pass scene video), config defaults, canary behavior, A1/B1 budgets, evidence locations, state recovery, process ownership, and kill switch.
- [ ] Add a no-GPU dry-run command that uses an explicit temporary storage root, fake backend responses, and fake candidate files; it must never call ComfyUI or the real llama binary.
- [ ] Add a bounded readiness command/checklist that performs read-only preflight until a human explicitly authorizes the one-scene canary.
- [ ] Preflight must prove: configured model/projector/executable exist; hashes recorded; exact RTX 4080 UUID/name resolved; llama args include image-min-tokens 1024/four-window policy; port free; Comfy queue empty; QC defaults canary; auto-advance false; source scene selected; disk headroom; rollback switch documented.
- [ ] The authorized real canary must use one safe selected production-like scene, persist the original candidate before QC, terminate Qwen before any repair generation, require human approval even on PASS, and stop before whole-job autonomous rollout.
- [ ] After the canary, verify candidate/evidence reconstruction, model/process exit, closed QC port, Comfy `/free`, unchanged protected v3/lab, and no missing scene artifact.
- [ ] Expand the labeled evaluation set before considering `auto_advance_pass=true`; the existing 3-bad/1-good smoke set is insufficient for autonomous production.

## Same-day canary risk ranking

1. **GPU identity mismatch:** the lab's ordinal config conflicts with its hardware report. Resolve and verify the physical 4080 UUID in llama startup telemetry before any real run.
2. **Supervisor resume/finalization split:** human canary holds cross process restarts; extracting finalization without changing baseline behavior is the most delicate production-code change.
3. **Small validation set:** 3/3 bad and 1/1 good supports canary use only. False positives and false negatives remain the largest product risk.
4. **llama.cpp version/flags:** validate the installed executable, mmproj compatibility, request freshness, image-token flag, process shutdown, and loopback port behavior in a bounded smoke test.
5. **Continuation revision generation:** A1/B1 must reuse existing frame/assets and preserve continuation recovery/lineage while creating new normal revisions.
6. **SQLite migration and promotion atomicity:** additive tables are straightforward, but a crash between human approval, candidate promotion, and finalization must remain idempotent.
7. **Latency:** the validated four-clip 4080 run took about 215 seconds; good videos receive full coverage and can extend job completion materially.
8. **Planner quality:** Qwen repair planning is not independently benchmarked. The allowlist, deterministic seed, one-attempt budget, blind re-judging, and human canary contain this risk.

## Completion gate

Phase 1 is ready for a bounded canary only when all focused/full CPU tests pass, no test writes production storage, the kill switch proves baseline equivalence, the physical 4080 is verified, model/projector/backend/prompt identities are persisted, worker-crash recovery loses no candidate, every non-accepted scene blocks finalization, and a human can reconstruct and decide the selected candidate from the existing GUI.

Do not enable `auto_advance_pass` during the initial production canary. Do not implement Phase 1.5 concurrency or any Phase-2 item in this plan.
