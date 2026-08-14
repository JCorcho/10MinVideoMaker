# Phase 1 Real Project Canary Results

## Executive Summary

- fresh I2V generation: **YES** (28 I2V attempts, 0 T2I attempts)
- real Qwen evaluation: **PARTIAL**
  - Scenes 1-5 had evaluator decisions (FAIL/UNCERTAIN from first-pass and both repair routes)
  - Scenes 6-28 were blocked by infra startup at QC boundary before evaluator output
- bounded repair routing: **YES** (A1 generated for first 5 scenes; B1 generated for first 5 scenes)
- human review gating: **YES** (all scenes terminated at `awaiting_qc_review` / HOLD_FOR_REVIEW)
- storage isolation: **YES**
  - no v2 canary outputs were written under `D:\LTX_Supervisor_Storage\jobs`
- zero external/subscriber effects: **YES** (`external_effects_attempted=false`, no subscriber/final artifact candidates)

## Aggregate Metrics

| Metric | Value |
|---|---:|
| scenes | 28 |
| original candidates | 28 |
| A1 candidates | 5 |
| B1 candidates | 5 |
| total fresh generated candidate videos | 38 |
| ORIGINAL PASS | 0 |
| ORIGINAL FAIL | 5 |
| ORIGINAL UNCERTAIN | 0 |
| A1 PASS | 0 |
| A1 FAIL | 5 |
| A1 UNCERTAIN | 0 |
| B1 PASS | 0 |
| B1 FAIL | 4 |
| B1 UNCERTAIN | 1 |
| PASS_PENDING_HUMAN scenes | 0 |
| HOLD_FOR_REVIEW scenes | 28 |
| malformed evaluator responses | 0 |
| evaluator refusals | 0 |
| evaluator infrastructure failures | 59 attempts (31 candidates with infra failures; 8 first5 + 23 rest23 rows) |
| total T2I attempts | 0 |
| total I2V attempts | 28 |
| external effects attempted | 0 |
| subscriber/final outputs produced | 0 |

## Per-Scene Results

| Scene | Original | A1 | B1 | Final State | Review Video |
|---|---|---|---|---|---|
| 1 | FAIL | FAIL | FAIL | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0001\revisions\0003\video.mp4 |
| 2 | FAIL | FAIL | FAIL | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0002\revisions\0003\video.mp4 |
| 3 | FAIL | FAIL | FAIL | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0003\revisions\0003\video.mp4 |
| 4 | FAIL | FAIL | FAIL | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0004\revisions\0003\video.mp4 |
| 5 | FAIL | FAIL | UNCERTAIN | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0005\revisions\0003\video.mp4 |
| 6 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0006\revisions\0001\video.mp4 |
| 7 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0007\revisions\0001\video.mp4 |
| 8 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0008\revisions\0001\video.mp4 |
| 9 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0009\revisions\0001\video.mp4 |
| 10 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0010\revisions\0001\video.mp4 |
| 11 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0011\revisions\0001\video.mp4 |
| 12 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0012\revisions\0001\video.mp4 |
| 13 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0013\revisions\0001\video.mp4 |
| 14 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0014\revisions\0001\video.mp4 |
| 15 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0015\revisions\0001\video.mp4 |
| 16 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0016\revisions\0001\video.mp4 |
| 17 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0017\revisions\0001\video.mp4 |
| 18 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0018\revisions\0001\video.mp4 |
| 19 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0019\revisions\0001\video.mp4 |
| 20 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0020\revisions\0001\video.mp4 |
| 21 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0021\revisions\0001\video.mp4 |
| 22 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0022\revisions\0001\video.mp4 |
| 23 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0023\revisions\0001\video.mp4 |
| 24 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0024\revisions\0001\video.mp4 |
| 25 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0025\revisions\0001\video.mp4 |
| 26 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0026\revisions\0001\video.mp4 |
| 27 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0027\revisions\0001\video.mp4 |
| 28 | NONE | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0028\revisions\0001\video.mp4 |

### Candidate paths for HOLD_FOR_REVIEW

- Scenes 1-5: all three tiers are present for review comparison (ORIGINAL/A1/B1).
- Scenes 6-28: only ORIGINAL tier present (no A1/B1 generated due infra boundary stop on original QC path).

## Operational Findings

- Cached frame reuse invariant held for all scenes (0 T2I attempts, 28 I2V attempts).
- Bounded repair routing behaved as designed:
  - Scenes 1-5 generated A1 and B1 candidates.
  - Scenes 6-28 did not reach repair stages due QC infra startup block.
- Dominant blocker is infrastructure, not model quality:
  - 31 candidate rows carried infrastructure_failure_count>0.
  - 59 infra failure events total from `infrastructure_failure_count`.
- No candidate accepted without explicit human approval in this run (`ACCEPTED` candidates: 0).
- `canary-summary.json` indicates both jobs reached `snapshot_state = awaiting_qc_review`, which is expected end-of-pipeline state.
- No external/subscriber effects were produced.
- No v2 canary output was created under `D:\LTX_Supervisor_Storage\jobs`.

## Phase-1 Verdict

- **NOT READY**
- Remaining real blocker: QC infra startup instability (GPU identity/device handoff + repair path availability) preventing full evaluator completion on scenes 6-28 and creating residual infra errors on repair candidates for scenes 1-5.
