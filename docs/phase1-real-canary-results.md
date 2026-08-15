# Phase 1 Real Project Canary Results

## Executive Summary

- fresh I2V generation: **YES**
- real Qwen evaluation: **YES** (all ORIGINALs have normalized decisions)
- bounded repair routing: **YES** (A1/B1 only for failed scenes)
- human review gating: **YES** (pipeline state is `awaiting_qc_review`)
- storage isolation: **YES** (all generated candidates are in the two v2 canary roots)
- zero external/subscriber effects: **YES**

## Aggregate Metrics

- scenes: 28
- ORIGINAL candidates: 28
- A1 candidates: 18
- B1 candidates: 5
- total fresh generated candidate videos: 51
- ORIGINAL PASS: 1
- ORIGINAL FAIL: 23
- ORIGINAL UNCERTAIN: 4
- A1 PASS: 0
- A1 FAIL: 5
- A1 UNCERTAIN: 0
- B1 PASS: 0
- B1 FAIL: 4
- B1 UNCERTAIN: 1
- PASS_PENDING_HUMAN scenes: 1
- HOLD_FOR_REVIEW scenes: 17
- malformed evaluator responses: 0
- evaluator refusals: 0
- evaluator infrastructure failures: 59
- total T2I attempts: 0
- total I2V attempts: 28
- external effects attempted: 0
- subscriber/final outputs produced: 0

### Infrastructure failure groups
- 46x ORIGINAL::The configured physical QC GPU UUID/name pair was not found; refusing to substitute a CUDA ordinal or another device.
- 8x A1::The dedicated QC loopback port is already in use; refusing to own an unknown process.
- 2x B1::ComfyUI is unavailable at the QC repair generation boundary.
- 2x B1::Could not prove fresh llama.cpp request context; slot erase failed.
- 1x B1::QC repair render failed: ComfyUI GET /history/a844b595-97f0-4edf-8e0b-a6ad16728fa1 failed: timed out

## Per-Scene Results
| Scene | Original | A1 | B1 | Final State | Review Video |
| --- | --- | --- | --- | --- | --- |
| 1 | FAIL (rev 1) | FAIL (rev 2) | FAIL (rev 3) | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0001\revisions\0003\video.mp4 |
| 2 | FAIL (rev 1) | FAIL (rev 2) | FAIL (rev 3) | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0002\revisions\0003\video.mp4 |
| 3 | FAIL (rev 1) | FAIL (rev 2) | FAIL (rev 3) | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0003\revisions\0003\video.mp4 |
| 4 | FAIL (rev 1) | FAIL (rev 2) | FAIL (rev 3) | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0004\revisions\0003\video.mp4 |
| 5 | FAIL (rev 1) | FAIL (rev 2) | UNCERTAIN (rev 3) | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0005\revisions\0003\video.mp4 |
| 6 | FAIL (rev 1) | PENDING_GENERATION (rev 2) | — | SUPERSEDED | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0006\revisions\0001\video.mp4 |
| 7 | FAIL (rev 1) | PENDING_GENERATION (rev 2) | — | SUPERSEDED | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0007\revisions\0001\video.mp4 |
| 8 | FAIL (rev 1) | PENDING_GENERATION (rev 2) | — | SUPERSEDED | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0008\revisions\0001\video.mp4 |
| 9 | UNCERTAIN (rev 1) | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0009\revisions\0001\video.mp4 |
| 10 | FAIL (rev 1) | PENDING_GENERATION (rev 2) | — | SUPERSEDED | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0010\revisions\0001\video.mp4 |
| 11 | FAIL (rev 1) | PENDING_GENERATION (rev 2) | — | SUPERSEDED | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0011\revisions\0001\video.mp4 |
| 12 | FAIL (rev 1) | PENDING_GENERATION (rev 2) | — | SUPERSEDED | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0012\revisions\0001\video.mp4 |
| 13 | FAIL (rev 1) | PENDING_GENERATION (rev 2) | — | SUPERSEDED | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0013\revisions\0001\video.mp4 |
| 14 | FAIL (rev 1) | PENDING_GENERATION (rev 2) | — | SUPERSEDED | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0014\revisions\0001\video.mp4 |
| 15 | UNCERTAIN (rev 1) | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0015\revisions\0001\video.mp4 |
| 16 | FAIL (rev 1) | PENDING_GENERATION (rev 2) | — | SUPERSEDED | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0016\revisions\0001\video.mp4 |
| 17 | FAIL (rev 1) | PENDING_GENERATION (rev 2) | — | SUPERSEDED | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0017\revisions\0001\video.mp4 |
| 18 | FAIL (rev 1) | PENDING_GENERATION (rev 2) | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0018\revisions\0001\video.mp4 |
| 19 | FAIL (rev 1) | PENDING_GENERATION (rev 2) | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0019\revisions\0001\video.mp4 |
| 20 | FAIL (rev 1) | PENDING_GENERATION (rev 2) | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0020\revisions\0001\video.mp4 |
| 21 | FAIL (rev 1) | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0021\revisions\0001\video.mp4 |
| 22 | PASS (rev 1) | — | — | PASS_PENDING_HUMAN | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0022\revisions\0001\video.mp4 |
| 23 | FAIL (rev 1) | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0023\revisions\0001\video.mp4 |
| 24 | FAIL (rev 1) | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0024\revisions\0001\video.mp4 |
| 25 | FAIL (rev 1) | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0025\revisions\0001\video.mp4 |
| 26 | FAIL (rev 1) | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0026\revisions\0001\video.mp4 |
| 27 | UNCERTAIN (rev 1) | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0027\revisions\0001\video.mp4 |
| 28 | UNCERTAIN (rev 1) | — | — | HOLD_FOR_REVIEW | D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0028\revisions\0001\video.mp4 |

## HOLD_FOR_REVIEW comparison

### Scene 1
- ORIGINAL: D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0001\revisions\0001\video.mp4
- A1: D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0001\revisions\0002\video.mp4
- B1: D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0001\revisions\0003\video.mp4

### Scene 2
- ORIGINAL: D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0002\revisions\0001\video.mp4
- A1: D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0002\revisions\0002\video.mp4
- B1: D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0002\revisions\0003\video.mp4

### Scene 3
- ORIGINAL: D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0003\revisions\0001\video.mp4
- A1: D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0003\revisions\0002\video.mp4
- B1: D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0003\revisions\0003\video.mp4

### Scene 4
- ORIGINAL: D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0004\revisions\0001\video.mp4
- A1: D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0004\revisions\0002\video.mp4
- B1: D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0004\revisions\0003\video.mp4

### Scene 5
- ORIGINAL: D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0005\revisions\0001\video.mp4
- A1: D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0005\revisions\0002\video.mp4
- B1: D:\LTX_Phase1_Canary_Storage\20260814-1844-first5-v2\jobs\canary-20260814-1844-s01-s05-v2\scenes\scene_0005\revisions\0003\video.mp4

### Scene 9
- ORIGINAL: D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0009\revisions\0001\video.mp4

### Scene 15
- ORIGINAL: D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0015\revisions\0001\video.mp4

### Scene 18
- ORIGINAL: D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0018\revisions\0001\video.mp4
- A1: D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0018\revisions\0002\video.mp4

### Scene 19
- ORIGINAL: D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0019\revisions\0001\video.mp4
- A1: D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0019\revisions\0002\video.mp4

### Scene 20
- ORIGINAL: D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0020\revisions\0001\video.mp4
- A1: D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0020\revisions\0002\video.mp4

### Scene 21
- ORIGINAL: D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0021\revisions\0001\video.mp4

### Scene 23
- ORIGINAL: D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0023\revisions\0001\video.mp4

### Scene 24
- ORIGINAL: D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0024\revisions\0001\video.mp4

### Scene 25
- ORIGINAL: D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0025\revisions\0001\video.mp4

### Scene 26
- ORIGINAL: D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0026\revisions\0001\video.mp4

### Scene 27
- ORIGINAL: D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0027\revisions\0001\video.mp4

### Scene 28
- ORIGINAL: D:\LTX_Phase1_Canary_Storage\20260814-1844-rest23-v2\jobs\canary-20260814-1844-s06-s28-v2\scenes\scene_0028\revisions\0001\video.mp4

## Operational Findings
- no new candidate generation occurred during recovery; existing MP4s were re-evaluated.
- dominant infra errors were startup GPU identity matching and first5 loopback/rework transport retries.
- no ACCEPTED candidates were introduced by recovery routing.

## Phase-1 Verdict

**NOT READY**

Remaining blockers:
- continuing `LlamaCppLifecycleError` on rest23 due physical GPU UUID/name lookup mismatch remains the blocker for evaluator bootstrap determinism.
