from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.qc_contracts import (
    QcCandidateState,
    QcDecision,
    QcHumanDecision,
    QcTier,
    evaluation_idempotency_key,
)
from tenminvideomaker.state_store import (
    ChunkState,
    JobState,
    ManualFinalState,
    PipelineState,
    PipelineStateStore,
    RemakeBatchState,
    RemakeMode,
    SceneState,
    StateTransitionError,
)

from test_contracts import payload


class StateStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = PipelineStateStore(Path(self.temporary_directory.name) / "pipeline.sqlite3")
        self.job = parse_job_payload(payload())

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _plan_chunks(self, count: int = 3) -> None:
        plan = {
            "schema_version": 1,
            "strategy": "ltx23_latent_overlap_v1",
            "chunks": count,
        }
        self.store.ensure_continuation_plan(
            self.job.job_id,
            1,
            1,
            "plan-sha256",
            plan,
        )
        self.store.plan_chunks(
            self.job.job_id,
            1,
            1,
            "plan-sha256",
            [
                {
                    "index": index,
                    "seed": index + 100,
                    "model_window_frames": 121,
                }
                for index in range(count)
            ],
        )

    def _complete_chunk(self, chunk_index: int, artifact_hash: str) -> None:
        attempt = self.store.begin_chunk_attempt(
            self.job.job_id,
            1,
            1,
            chunk_index,
            seed=(1 << 63) + chunk_index,
            parameters={"prompt_hash": f"prompt-{chunk_index}"},
        )
        self.store.update_chunk_attempt(
            self.job.job_id,
            1,
            1,
            chunk_index,
            attempt.attempt_number,
            ChunkState.COMPLETE,
            artifact_manifest_path=f"chunk-{chunk_index}.json",
            artifact_hash=artifact_hash,
            video_path=f"chunk-{chunk_index}.mp4",
            result={"frames": 121},
        )
        self.store.select_chunk_attempt(
            self.job.job_id,
            1,
            1,
            chunk_index,
            attempt.attempt_number,
        )

    def _create_qc_candidate(self, *, tier: QcTier = QcTier.ORIGINAL):
        self.store.claim_job(self.job)
        frame = Path(self.temporary_directory.name) / "frame.png"
        video = Path(self.temporary_directory.name) / "video.mp4"
        frame.write_bytes(b"frame")
        video.write_bytes(b"video")
        self.store.set_scene_state(
            self.job.job_id,
            1,
            SceneState.SUCCEEDED,
            frame_path=str(frame),
            video_path=str(video),
        )
        self.store.ensure_original_scene_revision(
            self.job.job_id,
            1,
            parameters={"job_id": self.job.job_id, "scene_id": 1},
            frame_path=str(frame),
            video_path=str(video),
        )
        return self.store.ensure_qc_candidate(
            candidate_id="candidate-original",
            job_id=self.job.job_id,
            scene_id=1,
            revision=1,
            tier=tier,
            parent_candidate_id=None,
            source_video_path=str(video),
            source_video_sha256=hashlib.sha256(b"video").hexdigest(),
            original_prompt="exact original prompt",
            current_prompt="exact original prompt",
            original_seed=2**64 - 1,
            current_seed=2**64 - 1,
            negative_prompt="locked negative",
            negative_prompt_sha256="b" * 64,
            state=QcCandidateState.PENDING_QC,
            next_action="evaluate",
        )

    def test_claiming_a_job_creates_pending_scene_and_asset_state(self) -> None:
        self.store.claim_job(self.job)
        snapshot = self.store.snapshot()
        self.assertEqual(snapshot.state, PipelineState.DOWNLOADING_ASSETS)
        self.assertEqual(snapshot.job_id, self.job.job_id)
        self.assertEqual(self.store.scene_states(self.job.job_id), {1: SceneState.PENDING})

    def test_human_review_claim_waits_until_explicit_approval(self) -> None:
        self.store.claim_job(self.job, review_required=True)
        self.assertEqual(self.store.snapshot().state, PipelineState.AWAITING_REVIEW)
        self.assertEqual(self.store.list_jobs()[0].status, JobState.AWAITING_REVIEW)
        self.store.approve_job(self.job.job_id)
        self.assertEqual(self.store.snapshot().state, PipelineState.DOWNLOADING_ASSETS)
        self.assertEqual(self.store.list_jobs()[0].status, JobState.QUEUED)

    def test_cross_job_remake_batch_versions_scenes_and_requires_cached_video_frame(self) -> None:
        self.store.claim_job(self.job)
        parameters = {"job_id": self.job.job_id, "scene_id": 1}
        with self.assertRaisesRegex(StateTransitionError, "cached frame"):
            self.store.create_remake_batch(
                [(self.job.job_id, 1, RemakeMode.VIDEO_ONLY, parameters)]
            )
        frame = Path(self.temporary_directory.name) / "frame.png"
        frame.write_bytes(b"frame")
        self.store.set_scene_state(
            self.job.job_id,
            1,
            SceneState.SUCCEEDED,
            frame_path=str(frame),
        )
        batch_id, revisions = self.store.create_remake_batch(
            [
                (self.job.job_id, 1, RemakeMode.VIDEO_ONLY, parameters),
                (self.job.job_id, 1, RemakeMode.IMAGE_AND_VIDEO, parameters),
            ]
        )
        self.assertEqual([item[2] for item in revisions], [1, 2])
        self.store.queue_remake_batch(batch_id, "after_current")
        batch = self.store.list_remake_batches()[0]
        self.assertEqual(batch.state, RemakeBatchState.QUEUED)
        self.assertEqual(batch.item_count, 2)

    def test_interrupted_remake_requeues_without_losing_prompt_or_chunk_checkpoint(
        self,
    ) -> None:
        self.store.claim_job(self.job)
        self.store.ensure_original_scene_revision(
            self.job.job_id,
            1,
            parameters={"job_id": self.job.job_id, "scene_id": 1},
        )
        batch_id, created = self.store.create_remake_batch(
            [
                (
                    self.job.job_id,
                    1,
                    RemakeMode.IMAGE_AND_VIDEO,
                    {"job_id": self.job.job_id, "scene_id": 1},
                )
            ]
        )
        revision = created[0][2]
        self.store.queue_remake_batch(batch_id, "after_current")
        self.store.set_remake_batch_state(batch_id, RemakeBatchState.RUNNING)
        self.store.set_remake_item_state(batch_id, 1, SceneState.RUNNING)
        self.store.update_scene_revision(
            self.job.job_id,
            1,
            revision,
            state=SceneState.RUNNING,
        )
        self.store.set_remake_item_prompt_id(
            batch_id,
            1,
            "persisted-remake-prompt",
            stage="i2v_legacy",
        )
        self.store.ensure_continuation_plan(
            self.job.job_id,
            1,
            revision,
            "remake-plan",
            {"strategy": "ltx23_latent_overlap_v1"},
        )
        self.store.plan_chunks(
            self.job.job_id,
            1,
            revision,
            "remake-plan",
            [{"index": 0, "model_window_frames": 121}],
        )
        attempt = self.store.begin_chunk_attempt(
            self.job.job_id,
            1,
            revision,
            0,
            seed=123,
            parameters={"immutable": "inputs"},
        )
        self.store.update_chunk_attempt(
            self.job.job_id,
            1,
            revision,
            0,
            attempt.attempt_number,
            ChunkState.STAGE1_COMPLETE,
            result={"stage1_prompt_id": "persisted-remake-prompt"},
        )

        self.assertEqual(
            self.store.recover_interrupted_remake_batches(),
            (batch_id,),
        )

        batch = self.store.next_queued_remake_batch()
        self.assertIsNotNone(batch)
        self.assertEqual(batch.batch_id, batch_id)
        item = self.store.remake_items(batch_id)[0]
        self.assertEqual(item.state, SceneState.PENDING)
        self.assertEqual(item.prompt_id, "persisted-remake-prompt")
        self.assertEqual(item.prompt_stage, "i2v_legacy")
        recovered_revision = next(
            item
            for item in self.store.scene_revisions(self.job.job_id, 1)
            if item.revision == revision
        )
        self.assertEqual(recovered_revision.state, SceneState.PENDING)
        self.assertEqual(
            self.store.chunk_records(self.job.job_id, 1, revision)[0].state,
            ChunkState.STAGE1_COMPLETE,
        )
        recovered_attempt = self.store.chunk_attempts(
            self.job.job_id,
            1,
            revision,
            0,
        )[0]
        self.assertEqual(recovered_attempt.state, ChunkState.STAGE1_COMPLETE)
        self.assertEqual(
            recovered_attempt.result["stage1_prompt_id"],
            "persisted-remake-prompt",
        )
        self.assertEqual(self.store.recover_interrupted_remake_batches(), ())

    def test_manual_final_snapshots_latest_included_successful_revision(self) -> None:
        self.store.claim_job(self.job)
        original = Path(self.temporary_directory.name) / "original.mp4"
        remake = Path(self.temporary_directory.name) / "remake.mp4"
        original.write_bytes(b"original")
        remake.write_bytes(b"remake")
        self.store.set_scene_state(
            self.job.job_id,
            1,
            SceneState.SUCCEEDED,
            video_path=str(original),
        )
        self.store.create_scene_revision(
            self.job.job_id,
            1,
            remake_mode=RemakeMode.IMAGE_AND_VIDEO,
            parameters={"version": 1},
            state=SceneState.SUCCEEDED,
            video_path=str(original),
        )
        self.store.create_scene_revision(
            self.job.job_id,
            1,
            remake_mode=RemakeMode.VIDEO_ONLY,
            parameters={"version": 2},
            state=SceneState.SUCCEEDED,
            video_path=str(remake),
        )

        request = self.store.queue_manual_final(self.job.job_id)

        self.assertEqual(request.state, ManualFinalState.QUEUED)
        self.assertEqual(
            self.store.manual_final_selection(request.request_id)[0].revision,
            2,
        )
        self.store.set_scene_manual_final_inclusion(
            self.job.job_id,
            1,
            included=False,
        )
        self.assertFalse(self.store.scene_records(self.job.job_id)[0].include_in_manual_final)
        self.store.set_manual_final_state(
            request.request_id,
            ManualFinalState.SUCCEEDED,
            output_path=str(Path(self.temporary_directory.name) / "final.mp4"),
        )
        with self.assertRaisesRegex(StateTransitionError, "At least one scene"):
            self.store.queue_manual_final(self.job.job_id)

    def test_job_cannot_be_claimed_twice(self) -> None:
        self.store.claim_job(self.job)
        with self.assertRaises(StateTransitionError):
            self.store.claim_job(self.job)

    def test_message_claim_is_idempotent(self) -> None:
        self.assertTrue(self.store.claim_message("imap:42"))
        self.assertFalse(self.store.claim_message("imap:42"))

    def test_inbound_job_claim_is_atomic_and_message_deduplicated(self) -> None:
        self.assertTrue(self.store.claim_inbound_job("imap:42", self.job).accepted)
        self.assertFalse(self.store.claim_inbound_job("imap:42", self.job).accepted)
        self.assertEqual(self.store.snapshot().state, PipelineState.DOWNLOADING_ASSETS)

    def test_previously_rejected_message_can_be_upgraded_to_a_job(self) -> None:
        self.assertTrue(self.store.claim_message("imap:42"))

        self.assertTrue(self.store.claim_inbound_job("imap:42", self.job).accepted)

        snapshot = self.store.snapshot()
        self.assertEqual(snapshot.state, PipelineState.DOWNLOADING_ASSETS)
        self.assertEqual(snapshot.job_id, self.job.job_id)

    def test_duplicate_grok_job_id_with_same_content_is_skipped(self) -> None:
        self.store.claim_job(self.job)
        self.store.transition(PipelineState.WAITING_FOR_GROK, job_id=self.job.job_id)
        resent = payload()
        resent["created_at"] = "2026-07-29T23:59:59Z"

        claim = self.store.claim_inbound_job("imap:duplicate", parse_job_payload(resent))

        self.assertFalse(claim.accepted)
        self.assertTrue(claim.duplicate_content)
        self.assertEqual(claim.source_job_id, self.job.job_id)

    def test_duplicate_grok_job_id_with_new_content_receives_local_suffix(self) -> None:
        self.store.claim_job(self.job)
        self.store.transition(PipelineState.WAITING_FOR_GROK, job_id=self.job.job_id)
        second = payload()
        second["scenes"][0]["i2v"]["prompt"] = "a genuinely new shot"

        claim = self.store.claim_inbound_job("imap:collision", parse_job_payload(second))

        self.assertTrue(claim.accepted)
        self.assertEqual(claim.payload.job_id, "20260724-1610-local-2")
        self.assertEqual(claim.payload.raw["source_job_id"], "20260724-1610")

    def test_local_collision_suffix_respects_job_id_limit(self) -> None:
        data = payload()
        data["job_id"] = "a" * 128
        original = parse_job_payload(data)
        self.store.claim_job(original)
        self.store.transition(PipelineState.WAITING_FOR_GROK, job_id=original.job_id)
        second = payload()
        second["job_id"] = original.job_id
        second["scenes"][0]["t2i"]["prompt"] = "new production content"

        claim = self.store.claim_inbound_job("imap:long-collision", parse_job_payload(second))

        self.assertTrue(claim.accepted)
        self.assertLessEqual(len(claim.payload.job_id), 128)
        self.assertTrue(claim.payload.job_id.endswith("-local-2"))

    def test_requeue_keeps_successful_scene_intact(self) -> None:
        self.store.claim_job(self.job)
        self.store.set_scene_state(self.job.job_id, 1, SceneState.SUCCEEDED, frame_path="frame.png", video_path="scene.mp4")
        self.assertEqual(self.store.requeue_unfinished_scenes(self.job.job_id), [])
        self.assertEqual(self.store.scene_states(self.job.job_id), {1: SceneState.SUCCEEDED})

    def test_new_job_is_rejected_while_pipeline_is_busy(self) -> None:
        self.store.claim_job(self.job)
        second = payload()
        second["job_id"] = "20260724-1611"
        with self.assertRaises(StateTransitionError):
            self.store.claim_job(parse_job_payload(second))

    def test_job_and_scene_records_restore_payload_and_attempts(self) -> None:
        self.store.claim_job(self.job)
        self.assertEqual(self.store.load_job(self.job.job_id).job_id, self.job.job_id)
        attempt = self.store.begin_scene_stage(
            self.job.job_id,
            1,
            PipelineState.RUNNING_T2I,
        )
        self.store.set_scene_prompt_id(
            self.job.job_id,
            1,
            "prompt-123",
            stage="t2i",
        )
        record = self.store.scene_records(self.job.job_id)[0]
        self.assertEqual(attempt, 1)
        self.assertEqual(record.t2i_attempts, 1)
        self.assertEqual(record.i2v_attempts, 0)
        self.assertEqual(record.prompt_id, "prompt-123")
        self.assertEqual(record.prompt_stage, "t2i")
        self.assertEqual(self.store.snapshot().active_scene_id, 1)

    def test_retry_job_preserves_attempts_and_prompt_ownership(self) -> None:
        self.store.claim_job(self.job)
        self.store.begin_scene_stage(
            self.job.job_id,
            1,
            PipelineState.RUNNING_T2I,
        )
        self.store.set_scene_prompt_id(
            self.job.job_id,
            1,
            "prompt-123",
            stage="t2i",
        )

        self.assertEqual(self.store.retry_job(self.job.job_id), [1])

        snapshot = self.store.snapshot()
        record = self.store.scene_records(self.job.job_id)[0]
        self.assertEqual(snapshot.state, PipelineState.DOWNLOADING_ASSETS)
        self.assertEqual(snapshot.job_id, self.job.job_id)
        self.assertEqual(record.state, SceneState.PENDING)
        self.assertIsNone(record.error)
        self.assertEqual(record.prompt_id, "prompt-123")
        self.assertEqual(record.prompt_stage, "t2i")
        self.assertEqual(record.t2i_attempts, 1)

    def test_retry_job_clears_nonactive_running_prompt_ownership(self) -> None:
        raw = payload()
        second = json.loads(json.dumps(raw["scenes"][0]))
        second["id"] = 2
        second["title"] = "Second scene"
        raw["scenes"].append(second)
        job = parse_job_payload(raw)
        self.store.claim_job(job)
        self.store.begin_scene_stage(
            job.job_id,
            1,
            PipelineState.RUNNING_T2I,
        )
        self.store.set_scene_prompt_id(
            job.job_id,
            1,
            "active-prompt",
            stage="t2i",
        )
        with self.store._connection() as connection:
            connection.execute(
                """
                UPDATE scenes
                SET state = ?, t2i_attempts = 2,
                    prompt_id = 'stale-prompt', prompt_stage = 't2i'
                WHERE job_id = ? AND scene_id = 2
                """,
                (SceneState.RUNNING, job.job_id),
            )

        self.store.retry_job(job.job_id)

        active, stale = self.store.scene_records(job.job_id)
        self.assertEqual(active.prompt_id, "active-prompt")
        self.assertEqual(active.t2i_attempts, 1)
        self.assertIsNone(stale.prompt_id)
        self.assertEqual(stale.prompt_stage, "t2i")
        self.assertEqual(stale.t2i_attempts, 0)

    def test_explicit_retry_refreshes_exhausted_t2i_budget(self) -> None:
        self.store.claim_job(self.job)
        self.store.begin_scene_stage(
            self.job.job_id,
            1,
            PipelineState.RUNNING_T2I,
        )
        self.store.begin_scene_stage(
            self.job.job_id,
            1,
            PipelineState.RUNNING_T2I,
        )
        self.store.set_scene_state(
            self.job.job_id,
            1,
            SceneState.FAILED,
            error="T2I retry budget exhausted",
        )
        self.store.transition(
            PipelineState.ERROR,
            job_id=self.job.job_id,
            error="T2I failed",
        )

        self.store.retry_job(self.job.job_id)

        record = self.store.scene_records(self.job.job_id)[0]
        self.assertEqual(record.t2i_attempts, 0)
        self.assertIsNone(record.prompt_id)
        self.assertEqual(record.prompt_stage, "t2i")

    def test_explicit_retry_refreshes_exhausted_legacy_i2v_budget(self) -> None:
        self.store.claim_job(self.job)
        frame = Path(self.temporary_directory.name) / "frame.png"
        frame.write_bytes(b"frame")
        self.store.set_scene_state(
            self.job.job_id,
            1,
            SceneState.PENDING,
            frame_path=str(frame),
        )
        self.store.begin_scene_stage(
            self.job.job_id,
            1,
            PipelineState.RUNNING_I2V,
            prompt_stage="i2v_legacy",
        )
        self.store.begin_scene_stage(
            self.job.job_id,
            1,
            PipelineState.RUNNING_I2V,
            prompt_stage="i2v_legacy",
        )
        self.store.set_scene_state(
            self.job.job_id,
            1,
            SceneState.FAILED,
            error="legacy I2V retry budget exhausted",
        )
        self.store.transition(
            PipelineState.ERROR,
            job_id=self.job.job_id,
            error="I2V failed",
        )

        self.store.retry_job(self.job.job_id)

        record = self.store.scene_records(self.job.job_id)[0]
        self.assertEqual(record.i2v_attempts, 0)
        self.assertIsNone(record.prompt_id)
        self.assertEqual(record.prompt_stage, "i2v_legacy")

    def test_explicit_retry_resets_both_budgets_when_saved_frame_is_missing(
        self,
    ) -> None:
        self.store.claim_job(self.job)
        missing_frame = Path(self.temporary_directory.name) / "missing.png"
        self.store.set_scene_state(
            self.job.job_id,
            1,
            SceneState.PENDING,
            frame_path=str(missing_frame),
        )
        for _ in range(2):
            self.store.begin_scene_stage(
                self.job.job_id,
                1,
                PipelineState.RUNNING_T2I,
            )
        for _ in range(2):
            self.store.begin_scene_stage(
                self.job.job_id,
                1,
                PipelineState.RUNNING_I2V,
                prompt_stage="i2v_legacy",
            )
        self.store.set_scene_state(
            self.job.job_id,
            1,
            SceneState.FAILED,
            error="frame disappeared after both budgets were consumed",
        )
        self.store.transition(
            PipelineState.ERROR,
            job_id=self.job.job_id,
            error="retry required",
        )

        self.store.retry_job(self.job.job_id)

        record = self.store.scene_records(self.job.job_id)[0]
        self.assertEqual(record.t2i_attempts, 0)
        self.assertEqual(record.i2v_attempts, 0)
        self.assertIsNone(record.prompt_id)
        self.assertEqual(record.prompt_stage, "t2i")

    def test_i2v_requeue_preserves_frame_and_clears_contaminated_clip(self) -> None:
        self.store.claim_job(self.job)
        self.store.begin_scene_stage(
            self.job.job_id,
            1,
            PipelineState.RUNNING_I2V,
        )
        self.store.set_scene_state(
            self.job.job_id,
            1,
            SceneState.SUCCEEDED,
            frame_path="frame.png",
            video_path="contaminated.mp4",
        )

        self.assertEqual(self.store.requeue_i2v_for_job(self.job.job_id), [1])

        snapshot = self.store.snapshot()
        record = self.store.scene_records(self.job.job_id)[0]
        self.assertEqual(snapshot.state, PipelineState.DOWNLOADING_ASSETS)
        self.assertEqual(record.state, SceneState.PENDING)
        self.assertEqual(record.frame_path, "frame.png")
        self.assertIsNone(record.video_path)
        self.assertEqual(record.i2v_attempts, 0)
        self.assertIsNone(record.prompt_id)

    def test_single_scene_i2v_requeue_resets_only_interrupted_scene(self) -> None:
        data = payload()
        second = payload()["scenes"][0]
        second["id"] = 2
        data["scenes"].append(second)
        job = parse_job_payload(data)
        self.store.claim_job(job)
        for scene_id in (1, 2):
            self.store.begin_scene_stage(
                job.job_id,
                scene_id,
                PipelineState.RUNNING_I2V,
            )
        self.store.set_scene_state(
            job.job_id,
            1,
            SceneState.SUCCEEDED,
            frame_path="frame-1.png",
            video_path="scene-1.mp4",
        )
        self.store.set_scene_state(
            job.job_id,
            2,
            SceneState.RUNNING,
            frame_path="frame-2.png",
        )

        self.store.requeue_scene_i2v(job.job_id, 2)

        first, second_record = self.store.scene_records(job.job_id)
        self.assertEqual(first.state, SceneState.SUCCEEDED)
        self.assertEqual(first.video_path, "scene-1.mp4")
        self.assertEqual(first.i2v_attempts, 1)
        self.assertEqual(second_record.state, SceneState.PENDING)
        self.assertEqual(second_record.frame_path, "frame-2.png")
        self.assertIsNone(second_record.video_path)
        self.assertEqual(second_record.i2v_attempts, 0)
        self.assertEqual(
            self.store.snapshot().state,
            PipelineState.DOWNLOADING_ASSETS,
        )

    def test_abandon_job_cancels_unfinished_scene_and_releases_pipeline(self) -> None:
        self.store.claim_job(self.job)
        self.store.begin_scene_stage(
            self.job.job_id,
            1,
            PipelineState.RUNNING_T2I,
        )
        self.store.set_scene_prompt_id(
            self.job.job_id,
            1,
            "prompt-123",
            stage="t2i",
        )
        self.store.set_scene_state(
            self.job.job_id,
            1,
            SceneState.FAILED,
            error="unsafe or malformed payload",
        )
        self.store.transition(
            PipelineState.ERROR,
            job_id=self.job.job_id,
            error="asset preparation failed",
        )

        self.assertEqual(self.store.abandon_job(self.job.job_id), [1])

        snapshot = self.store.snapshot()
        record = self.store.scene_records(self.job.job_id)[0]
        self.assertEqual(snapshot.state, PipelineState.IDLE)
        self.assertIsNone(snapshot.job_id)
        self.assertIsNone(snapshot.error)
        self.assertEqual(record.state, SceneState.CANCELLED)
        self.assertEqual(
            record.error,
            "unsafe or malformed payload | "
            "Abandoned by the user from the one-click launcher.",
        )
        self.assertIsNone(record.prompt_id)
        self.assertEqual(record.t2i_attempts, 1)
        self.assertEqual(self.store.load_job(self.job.job_id).job_id, self.job.job_id)

        replacement_payload = payload()
        replacement_payload["job_id"] = "20260724-1611"
        replacement = parse_job_payload(replacement_payload)
        self.store.claim_job(replacement)
        self.assertEqual(self.store.snapshot().job_id, replacement.job_id)
        self.assertEqual(
            self.store.snapshot().state,
            PipelineState.DOWNLOADING_ASSETS,
        )

    def test_initialize_migrates_legacy_database_without_changing_saved_job(self) -> None:
        database_path = Path(self.temporary_directory.name) / "legacy.sqlite3"
        raw = payload()
        created_at = "2026-07-24T16:10:45Z"
        connection = sqlite3.connect(database_path)
        try:
            connection.executescript(
                """
                CREATE TABLE jobs (
                    job_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE scenes (
                    job_id TEXT NOT NULL,
                    scene_id INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    frame_path TEXT,
                    video_path TEXT,
                    error TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, scene_id)
                );
                """
            )
            connection.execute(
                "INSERT INTO jobs (job_id, payload_json, created_at) VALUES (?, ?, ?)",
                (raw["job_id"], json.dumps(raw), created_at),
            )
            connection.execute(
                """
                INSERT INTO scenes (
                    job_id, scene_id, state, frame_path, video_path, error, updated_at
                ) VALUES (?, 1, ?, NULL, NULL, NULL, ?)
                """,
                (raw["job_id"], SceneState.PENDING, created_at),
            )
            connection.commit()
        finally:
            connection.close()

        migrated = PipelineStateStore(database_path)
        migrated.initialize()
        migrated.initialize()

        self.assertEqual(migrated.load_job(raw["job_id"]).job_id, raw["job_id"])
        connection = sqlite3.connect(database_path)
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        finally:
            connection.close()
        self.assertTrue(
            {"continuation_plans", "scene_chunks", "chunk_attempts"} <= tables
        )

    def test_initialize_classifies_only_provable_intermediate_prompt_owner(self) -> None:
        database_path = Path(self.temporary_directory.name) / "intermediate.sqlite3"
        raw = payload()
        created_at = "2026-07-24T16:10:45Z"
        connection = sqlite3.connect(database_path)
        try:
            connection.executescript(
                """
                CREATE TABLE pipeline_state (
                    singleton INTEGER PRIMARY KEY,
                    state TEXT NOT NULL,
                    job_id TEXT,
                    active_scene_id INTEGER,
                    error TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE jobs (
                    job_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE scenes (
                    job_id TEXT NOT NULL,
                    scene_id INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    frame_path TEXT,
                    video_path TEXT,
                    error TEXT,
                    t2i_attempts INTEGER NOT NULL DEFAULT 0,
                    i2v_attempts INTEGER NOT NULL DEFAULT 0,
                    prompt_id TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, scene_id)
                );
                """
            )
            connection.execute(
                """
                INSERT INTO pipeline_state (
                    singleton, state, job_id, active_scene_id, error, updated_at
                ) VALUES (1, ?, ?, 1, NULL, ?)
                """,
                (PipelineState.RUNNING_I2V, raw["job_id"], created_at),
            )
            connection.execute(
                "INSERT INTO jobs (job_id, payload_json, created_at) VALUES (?, ?, ?)",
                (raw["job_id"], json.dumps(raw), created_at),
            )
            connection.executemany(
                """
                INSERT INTO scenes (
                    job_id, scene_id, state, frame_path, video_path, error,
                    t2i_attempts, i2v_attempts, prompt_id, updated_at
                ) VALUES (?, ?, ?, ?, NULL, NULL, 1, 1, ?, ?)
                """,
                (
                    (
                        raw["job_id"],
                        1,
                        SceneState.RUNNING,
                        "frame.png",
                        "active-prompt",
                        created_at,
                    ),
                    (
                        raw["job_id"],
                        2,
                        SceneState.PENDING,
                        "frame-2.png",
                        "stale-prompt",
                        created_at,
                    ),
                ),
            )
            connection.commit()
        finally:
            connection.close()

        migrated = PipelineStateStore(database_path)
        migrated.initialize()

        active, stale = migrated.scene_records(raw["job_id"])
        self.assertEqual(active.prompt_id, "active-prompt")
        self.assertEqual(active.prompt_stage, "i2v_legacy")
        self.assertIsNone(stale.prompt_id)
        self.assertIsNone(stale.prompt_stage)

    def test_initialize_normalizes_existing_generic_i2v_by_exact_prompt_owner(
        self,
    ) -> None:
        self.store.claim_job(self.job)
        self._plan_chunks(count=1)
        attempt = self.store.begin_chunk_attempt(
            self.job.job_id,
            1,
            1,
            0,
            seed=123,
            parameters={"immutable": "inputs"},
        )
        self.store.update_chunk_attempt(
            self.job.job_id,
            1,
            1,
            0,
            attempt.attempt_number,
            ChunkState.GENERATING_STAGE1,
            result={"stage1_prompt_id": "continuation-prompt"},
        )
        self.store.begin_scene_stage(
            self.job.job_id,
            1,
            PipelineState.RUNNING_I2V,
            prompt_stage="i2v_continuation",
        )
        self.store.set_scene_prompt_id(
            self.job.job_id,
            1,
            "continuation-prompt",
            stage="i2v_continuation",
        )
        with self.store._connection() as connection:
            connection.execute(
                "UPDATE scenes SET prompt_stage = 'i2v' WHERE job_id = ?",
                (self.job.job_id,),
            )

        self.store.initialize()
        self.store.initialize()
        record = self.store.scene_records(self.job.job_id)[0]
        self.assertEqual(record.prompt_stage, "i2v_continuation")
        self.assertEqual(record.prompt_id, "continuation-prompt")

        self.store.set_scene_prompt_id(
            self.job.job_id,
            1,
            "new-legacy-prompt",
            stage="i2v_legacy",
        )
        with self.store._connection() as connection:
            connection.execute(
                "UPDATE scenes SET prompt_stage = 'i2v' WHERE job_id = ?",
                (self.job.job_id,),
            )

        self.store.initialize()
        self.store.initialize()
        record = self.store.scene_records(self.job.job_id)[0]
        self.assertEqual(record.prompt_stage, "i2v_legacy")
        self.assertEqual(record.prompt_id, "new-legacy-prompt")

    def test_initialize_normalizes_generic_remake_i2v_by_exact_prompt_owner(
        self,
    ) -> None:
        self.store.claim_job(self.job)
        batch_id, created = self.store.create_remake_batch(
            [
                (
                    self.job.job_id,
                    1,
                    RemakeMode.IMAGE_AND_VIDEO,
                    {"job_id": self.job.job_id, "scene_id": 1},
                )
            ]
        )
        revision = created[0][2]
        self.store.ensure_continuation_plan(
            self.job.job_id,
            1,
            revision,
            "remake-plan",
            {"strategy": "ltx23_latent_overlap_v1"},
        )
        self.store.plan_chunks(
            self.job.job_id,
            1,
            revision,
            "remake-plan",
            [{"index": 0, "model_window_frames": 121}],
        )
        attempt = self.store.begin_chunk_attempt(
            self.job.job_id,
            1,
            revision,
            0,
            seed=123,
            parameters={"immutable": "inputs"},
        )
        self.store.update_chunk_attempt(
            self.job.job_id,
            1,
            revision,
            0,
            attempt.attempt_number,
            ChunkState.GENERATING_STAGE1,
            result={"stage1_prompt_id": "remake-continuation-prompt"},
        )
        self.store.set_remake_item_prompt_id(
            batch_id,
            1,
            "remake-continuation-prompt",
            stage="i2v_continuation",
        )
        with self.store._connection() as connection:
            connection.execute(
                """
                UPDATE remake_items SET prompt_stage = 'i2v'
                WHERE batch_id = ? AND position = 1
                """,
                (batch_id,),
            )

        self.store.initialize()
        item = self.store.remake_items(batch_id)[0]
        self.assertEqual(item.prompt_stage, "i2v_continuation")

        self.store.set_remake_item_prompt_id(
            batch_id,
            1,
            "remake-legacy-prompt",
            stage="i2v_legacy",
        )
        with self.store._connection() as connection:
            connection.execute(
                """
                UPDATE remake_items SET prompt_stage = 'i2v'
                WHERE batch_id = ? AND position = 1
                """,
                (batch_id,),
            )
        self.store.initialize()
        self.store.initialize()
        item = self.store.remake_items(batch_id)[0]
        self.assertEqual(item.prompt_stage, "i2v_legacy")
        self.assertEqual(item.prompt_id, "remake-legacy-prompt")

    def test_full_i2v_requeue_invalidates_accepted_continuation_chain(self) -> None:
        self.store.claim_job(self.job)
        self.store.ensure_original_scene_revision(
            self.job.job_id,
            1,
            parameters={"job_id": self.job.job_id, "scene_id": 1},
        )
        self._plan_chunks(count=2)
        self._complete_chunk(0, "accepted-zero")
        self._complete_chunk(1, "accepted-one")
        self.store.set_scene_state(
            self.job.job_id,
            1,
            SceneState.SUCCEEDED,
            frame_path="frame.png",
            video_path="assembled.mp4",
        )
        self.store.update_scene_revision(
            self.job.job_id,
            1,
            1,
            state=SceneState.SUCCEEDED,
            frame_path="frame.png",
            video_path="assembled.mp4",
        )

        self.store.requeue_i2v_for_job(self.job.job_id)

        chunks = self.store.chunk_records(self.job.job_id, 1, 1)
        self.assertEqual(
            [chunk.state for chunk in chunks],
            [ChunkState.READY, ChunkState.STALE_UPSTREAM],
        )
        self.assertTrue(
            all(chunk.accepted_attempt_number is None for chunk in chunks)
        )
        self.assertEqual(
            self.store.chunk_attempts(self.job.job_id, 1, 1, 0)[0].state,
            ChunkState.INVALIDATED,
        )
        self.assertEqual(
            self.store.chunk_attempts(self.job.job_id, 1, 1, 1)[0].state,
            ChunkState.STALE_UPSTREAM,
        )
        revision = self.store.scene_revisions(self.job.job_id, 1)[0]
        self.assertEqual(revision.state, SceneState.PENDING)
        self.assertIsNone(revision.video_path)

    def test_continuation_plan_and_chunks_are_idempotent_and_immutable(self) -> None:
        self.store.claim_job(self.job)
        plan = {"strategy": "latent-overlap", "generation_master_frames": 241}
        first = self.store.ensure_continuation_plan(
            self.job.job_id, 1, 1, "plan-v1", plan
        )
        second = self.store.ensure_continuation_plan(
            self.job.job_id, 1, 1, "plan-v1", dict(reversed(list(plan.items())))
        )
        self.assertEqual(first, second)
        with self.assertRaisesRegex(StateTransitionError, "immutable"):
            self.store.ensure_continuation_plan(
                self.job.job_id,
                1,
                1,
                "plan-v2",
                {"strategy": "different"},
            )

        chunks = [{"index": 0, "frames": 121}, {"index": 1, "frames": 121}]
        first_chunks = self.store.plan_chunks(
            self.job.job_id, 1, 1, "plan-v1", chunks
        )
        second_chunks = self.store.plan_chunks(
            self.job.job_id, 1, 1, "plan-v1", chunks
        )
        self.assertEqual(first_chunks, second_chunks)
        self.assertEqual(
            [chunk.state for chunk in first_chunks],
            [ChunkState.READY, ChunkState.BLOCKED_UPSTREAM],
        )
        with self.assertRaisesRegex(StateTransitionError, "immutable"):
            self.store.plan_chunks(
                self.job.job_id,
                1,
                1,
                "plan-v1",
                [{"index": 0, "frames": 81}],
            )

    def test_explicit_legacy_strategy_upgrade_preserves_snapshot_and_resets_scene(self) -> None:
        self.store.claim_job(self.job)
        self.store.ensure_original_scene_revision(
            self.job.job_id,
            1,
            parameters={"job_id": self.job.job_id, "scene_id": 1},
        )
        self._plan_chunks(2)
        self._complete_chunk(0, "legacy-artifact")

        snapshot = self.store.continuation_revision_snapshot(
            self.job.job_id,
            1,
            1,
        )
        self.assertEqual(snapshot["plan"]["document"]["strategy"], "ltx23_latent_overlap_v1")
        self.assertEqual(len(snapshot["chunks"]), 2)
        self.assertEqual(len(snapshot["attempts"]), 1)

        self.store.reset_legacy_original_continuation(
            self.job.job_id,
            1,
            expected_plan_hash="plan-sha256",
            expected_strategy="ltx23_latent_overlap_v1",
        )

        self.assertIsNone(self.store.continuation_plan(self.job.job_id, 1, 1))
        self.assertEqual(self.store.chunk_records(self.job.job_id, 1, 1), ())
        scene = self.store.scene_records(self.job.job_id)[0]
        self.assertEqual(scene.state, SceneState.PENDING)
        self.assertEqual(scene.i2v_attempts, 0)
        self.assertIsNone(scene.prompt_id)
        self.assertEqual(self.store.snapshot().state, PipelineState.DOWNLOADING_ASSETS)

    def test_strategy_upgrade_rejects_unexpected_plan_identity(self) -> None:
        self.store.claim_job(self.job)
        self._plan_chunks(1)

        with self.assertRaisesRegex(StateTransitionError, "changed before upgrade"):
            self.store.reset_legacy_original_continuation(
                self.job.job_id,
                1,
                expected_plan_hash="wrong",
                expected_strategy="ltx23_latent_overlap_v1",
            )

    def test_chunk_attempt_resume_preserves_full_uint64_seed_as_text(self) -> None:
        self.store.claim_job(self.job)
        self._plan_chunks()
        maximum_seed = (1 << 64) - 1
        first = self.store.begin_chunk_attempt(
            self.job.job_id,
            1,
            1,
            0,
            seed=maximum_seed,
            parameters={"prompt_hash": "abc"},
            attempt_number=1,
        )
        reopened = PipelineStateStore(self.store.database_path)
        resumed = reopened.begin_chunk_attempt(
            self.job.job_id,
            1,
            1,
            0,
            seed=maximum_seed,
            parameters={"prompt_hash": "abc"},
            attempt_number=1,
        )

        self.assertEqual(first, resumed)
        self.assertEqual(resumed.seed, maximum_seed)
        connection = sqlite3.connect(self.store.database_path)
        try:
            stored_seed, storage_type = connection.execute(
                "SELECT seed, typeof(seed) FROM chunk_attempts"
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual(stored_seed, str(maximum_seed))
        self.assertEqual(storage_type, "text")
        with self.assertRaisesRegex(StateTransitionError, "different immutable inputs"):
            reopened.begin_chunk_attempt(
                self.job.job_id,
                1,
                1,
                0,
                seed=maximum_seed - 1,
                parameters={"prompt_hash": "abc"},
                attempt_number=1,
            )
        with self.assertRaisesRegex(StateTransitionError, "active attempt"):
            reopened.begin_chunk_attempt(
                self.job.job_id,
                1,
                1,
                0,
                seed=5,
                attempt_number=2,
            )
        with self.assertRaisesRegex(StateTransitionError, "unsigned 64-bit"):
            reopened.begin_chunk_attempt(
                self.job.job_id,
                1,
                1,
                0,
                seed=1 << 64,
            )

    def test_selecting_attempt_unlocks_next_chunk_and_reports_progress(self) -> None:
        self.store.claim_job(self.job)
        self._plan_chunks()
        attempt = self.store.begin_chunk_attempt(
            self.job.job_id,
            1,
            1,
            0,
            seed=99,
            parameters={"workflow_hash": "workflow"},
        )
        self.store.update_chunk_attempt(
            self.job.job_id,
            1,
            1,
            0,
            attempt.attempt_number,
            ChunkState.STAGE1_COMPLETE,
            artifact_manifest_path="stage1.json",
        )
        with self.assertRaisesRegex(StateTransitionError, "requires an artifact hash"):
            self.store.update_chunk_attempt(
                self.job.job_id,
                1,
                1,
                0,
                attempt.attempt_number,
                ChunkState.COMPLETE,
            )
        completed = self.store.update_chunk_attempt(
            self.job.job_id,
            1,
            1,
            0,
            attempt.attempt_number,
            ChunkState.COMPLETE,
            artifact_hash="handoff-0",
            video_path="chunk-0.mp4",
            result={"observed_frames": 121},
        )
        with self.assertRaisesRegex(StateTransitionError, "immutable outputs"):
            self.store.update_chunk_attempt(
                self.job.job_id,
                1,
                1,
                0,
                attempt.attempt_number,
                ChunkState.COMPLETE,
                artifact_hash="different-handoff",
            )
        selected = self.store.select_chunk_attempt(
            self.job.job_id,
            1,
            1,
            0,
            completed.attempt_number,
        )

        self.assertEqual(selected.accepted_artifact_hash, "handoff-0")
        self.assertEqual(
            [chunk.state for chunk in self.store.chunk_records(self.job.job_id, 1, 1)],
            [
                ChunkState.COMPLETE,
                ChunkState.READY,
                ChunkState.BLOCKED_UPSTREAM,
            ],
        )
        progress = self.store.chunk_progress(self.job.job_id, 1, 1)
        self.assertEqual(progress.total_count, 3)
        self.assertEqual(progress.complete_count, 1)
        self.assertEqual(progress.next_chunk_index, 1)
        with self.assertRaisesRegex(StateTransitionError, "transactionally"):
            self.store.update_chunk_attempt(
                self.job.job_id,
                1,
                1,
                0,
                completed.attempt_number,
                ChunkState.INVALIDATED,
            )

    def test_invalidation_cascades_and_preserves_historical_attempts(self) -> None:
        self.store.claim_job(self.job)
        self._plan_chunks()
        self._complete_chunk(0, "handoff-0")
        self._complete_chunk(1, "handoff-1")
        self._complete_chunk(2, "handoff-2")

        self.assertEqual(
            self.store.invalidate_chunks_from(
                self.job.job_id,
                1,
                1,
                1,
                reason="prompt changed",
            ),
            [1, 2],
        )

        chunks = self.store.chunk_records(self.job.job_id, 1, 1)
        self.assertEqual(chunks[0].state, ChunkState.COMPLETE)
        self.assertEqual(chunks[1].state, ChunkState.READY)
        self.assertEqual(chunks[2].state, ChunkState.STALE_UPSTREAM)
        self.assertIsNone(chunks[1].accepted_attempt_number)
        self.assertIsNone(chunks[2].accepted_attempt_number)
        self.assertEqual(
            self.store.chunk_attempts(self.job.job_id, 1, 1, 1)[0].state,
            ChunkState.INVALIDATED,
        )
        self.assertEqual(
            self.store.chunk_attempts(self.job.job_id, 1, 1, 2)[0].state,
            ChunkState.STALE_UPSTREAM,
        )
        self.store.invalidate_chunks_from(self.job.job_id, 1, 1, 2)
        self.assertEqual(
            self.store.chunk_records(self.job.job_id, 1, 1)[2].state,
            ChunkState.STALE_UPSTREAM,
        )

    def test_selecting_alternate_attempt_invalidates_only_descendants(self) -> None:
        self.store.claim_job(self.job)
        self._plan_chunks()
        self._complete_chunk(0, "handoff-0")
        self._complete_chunk(1, "handoff-1")
        self._complete_chunk(2, "handoff-2")
        alternate = self.store.begin_chunk_attempt(
            self.job.job_id,
            1,
            1,
            1,
            seed=987,
            variation_index=1,
            parameters={"prompt_hash": "prompt-1-alt"},
        )
        self.assertEqual(alternate.attempt_number, 2)
        self.assertEqual(
            self.store.chunk_records(self.job.job_id, 1, 1)[1].state,
            ChunkState.COMPLETE,
        )
        self.store.update_chunk_attempt(
            self.job.job_id,
            1,
            1,
            1,
            alternate.attempt_number,
            ChunkState.COMPLETE,
            artifact_hash="handoff-1-alt",
            video_path="chunk-1-alt.mp4",
        )

        selected = self.store.select_chunk_attempt(
            self.job.job_id,
            1,
            1,
            1,
            alternate.attempt_number,
        )

        self.assertEqual(selected.accepted_attempt_number, 2)
        chunks = self.store.chunk_records(self.job.job_id, 1, 1)
        self.assertEqual(chunks[0].accepted_artifact_hash, "handoff-0")
        self.assertEqual(chunks[1].accepted_artifact_hash, "handoff-1-alt")
        self.assertEqual(chunks[2].state, ChunkState.READY)
        self.assertEqual(
            self.store.chunk_attempts(self.job.job_id, 1, 1, 2)[0].state,
            ChunkState.INVALIDATED,
        )

    def test_abandon_job_cancels_active_chunk_attempts(self) -> None:
        self.store.claim_job(self.job)
        self._plan_chunks()
        attempt = self.store.begin_chunk_attempt(
            self.job.job_id,
            1,
            1,
            0,
            seed=123,
        )
        self.store.update_chunk_attempt(
            self.job.job_id,
            1,
            1,
            0,
            attempt.attempt_number,
            ChunkState.GENERATING_STAGE2,
        )

        self.store.abandon_job(self.job.job_id, reason="cancelled for test")

        self.assertEqual(
            self.store.chunk_attempts(self.job.job_id, 1, 1, 0)[0].state,
            ChunkState.CANCELLED,
        )
        self.assertEqual(
            [chunk.state for chunk in self.store.chunk_records(self.job.job_id, 1, 1)],
            [ChunkState.CANCELLED, ChunkState.CANCELLED, ChunkState.CANCELLED],
        )

    def test_detached_job_cannot_create_continuation_plan(self) -> None:
        self.store.claim_job(self.job)
        self.store.abandon_job(self.job.job_id, reason="cancelled before planning")

        with self.assertRaisesRegex(StateTransitionError, "detached scene"):
            self.store.ensure_continuation_plan(
                self.job.job_id,
                1,
                1,
                "late-plan",
                {"strategy": "ltx23_latent_overlap_v1"},
            )

    def test_detached_job_cannot_materialize_continuation_chunks(self) -> None:
        self.store.claim_job(self.job)
        self.store.ensure_continuation_plan(
            self.job.job_id,
            1,
            1,
            "planned-before-cancel",
            {"strategy": "ltx23_latent_overlap_v1"},
        )
        self.store.abandon_job(self.job.job_id, reason="cancelled before chunk insert")

        with self.assertRaisesRegex(StateTransitionError, "detached scene"):
            self.store.plan_chunks(
                self.job.job_id,
                1,
                1,
                "planned-before-cancel",
                [{"index": 0, "model_window_frames": 121}],
            )

    def test_continuation_scene_stage_resume_does_not_consume_an_attempt(self) -> None:
        self.store.claim_job(self.job)
        first = self.store.begin_scene_stage(
            self.job.job_id,
            1,
            PipelineState.RUNNING_I2V,
        )
        resumed = self.store.begin_scene_stage(
            self.job.job_id,
            1,
            PipelineState.RUNNING_I2V,
            resume=True,
        )

        self.assertEqual((first, resumed), (1, 1))
        self.assertEqual(
            self.store.scene_records(self.job.job_id)[0].i2v_attempts,
            1,
        )

    def test_retry_abandoned_job_revives_first_incomplete_continuation_chunk(self) -> None:
        self.store.claim_job(self.job)
        self._plan_chunks()
        self._complete_chunk(0, "accepted-zero")
        attempt = self.store.begin_chunk_attempt(
            self.job.job_id,
            1,
            1,
            1,
            seed=456,
            upstream_artifact_hash="accepted-zero",
        )
        self.store.update_chunk_attempt(
            self.job.job_id,
            1,
            1,
            1,
            attempt.attempt_number,
            ChunkState.GENERATING_STAGE2,
        )
        self.store.abandon_job(self.job.job_id)

        unfinished = self.store.retry_job(self.job.job_id)

        self.assertEqual(unfinished, [1])
        self.assertEqual(self.store.list_jobs()[0].status, JobState.QUEUED)
        self.assertEqual(
            [chunk.state for chunk in self.store.chunk_records(self.job.job_id, 1, 1)],
            [ChunkState.COMPLETE, ChunkState.READY, ChunkState.BLOCKED_UPSTREAM],
        )
        self.assertEqual(
            self.store.chunk_attempts(self.job.job_id, 1, 1, 1)[0].state,
            ChunkState.CANCELLED,
        )
        replacement = self.store.begin_chunk_attempt(
            self.job.job_id,
            1,
            1,
            1,
            seed=456,
            upstream_artifact_hash="accepted-zero",
        )
        self.assertEqual(replacement.attempt_number, 2)

    def test_late_automatic_mutations_cannot_reclaim_abandoned_job(self) -> None:
        self.store.claim_job(self.job)
        self.store.ensure_original_scene_revision(
            self.job.job_id,
            1,
            parameters={"job_id": self.job.job_id, "scene_id": 1},
        )
        self.store.abandon_job(self.job.job_id)

        for operation in (
            lambda: self.store.transition(
                PipelineState.RUNNING_I2V,
                job_id=self.job.job_id,
                active_scene_id=1,
            ),
            lambda: self.store.set_job_status(self.job.job_id, JobState.RUNNING),
            lambda: self.store.begin_scene_stage(
                self.job.job_id,
                1,
                PipelineState.RUNNING_I2V,
            ),
            lambda: self.store.set_scene_prompt_id(
                self.job.job_id,
                1,
                "late-prompt",
                stage="t2i",
            ),
            lambda: self.store.update_scene_revision(
                self.job.job_id,
                1,
                1,
                state=SceneState.SUCCEEDED,
            ),
        ):
            with self.subTest(operation=operation):
                with self.assertRaises(StateTransitionError):
                    operation()
        self.assertEqual(self.store.snapshot().state, PipelineState.IDLE)
        self.assertIsNone(self.store.snapshot().job_id)
        self.assertEqual(self.store.list_jobs()[0].status, JobState.CANCELLED)

    def test_abandon_cancels_unselected_completion_and_prevents_resurrection(self) -> None:
        self.store.claim_job(self.job)
        self._plan_chunks(count=1)
        attempt = self.store.begin_chunk_attempt(
            self.job.job_id,
            1,
            1,
            0,
            seed=456,
        )
        self.store.update_chunk_attempt(
            self.job.job_id,
            1,
            1,
            0,
            attempt.attempt_number,
            ChunkState.COMPLETE,
            artifact_hash="unselected-handoff",
        )

        self.store.abandon_job(self.job.job_id)

        self.assertEqual(
            self.store.chunk_records(self.job.job_id, 1, 1)[0].state,
            ChunkState.CANCELLED,
        )
        with self.assertRaisesRegex(StateTransitionError, "cancelled job"):
            self.store.select_chunk_attempt(
                self.job.job_id,
                1,
                1,
                0,
                attempt.attempt_number,
            )
        with self.assertRaisesRegex(StateTransitionError, "cancelled job"):
            self.store.invalidate_chunks_from(self.job.job_id, 1, 1, 0)
        with self.assertRaisesRegex(StateTransitionError, "cancelled job"):
            self.store.begin_chunk_attempt(
                self.job.job_id,
                1,
                1,
                0,
                seed=789,
            )
        with self.assertRaisesRegex(StateTransitionError, "cannot be resurrected"):
            self.store.set_scene_state(
                self.job.job_id,
                1,
                SceneState.SUCCEEDED,
            )

    def test_cancelled_job_still_allows_an_active_historical_remake_revision(self) -> None:
        self.store.claim_job(self.job)
        self.store.ensure_original_scene_revision(
            self.job.job_id,
            1,
            parameters={"job_id": self.job.job_id, "scene_id": 1},
        )
        self.store.abandon_job(self.job.job_id)
        _batch_id, created = self.store.create_remake_batch(
            [
                (
                    self.job.job_id,
                    1,
                    RemakeMode.IMAGE_AND_VIDEO,
                    {"job_id": self.job.job_id, "scene_id": 1},
                )
            ]
        )
        revision = created[0][2]
        self.assertEqual(revision, 2)
        self.store.update_scene_revision(
            self.job.job_id,
            1,
            revision,
            state=SceneState.RUNNING,
        )
        self.store.ensure_continuation_plan(
            self.job.job_id,
            1,
            revision,
            "remake-plan",
            {"strategy": "ltx23_latent_overlap_v1"},
        )
        self.store.plan_chunks(
            self.job.job_id,
            1,
            revision,
            "remake-plan",
            [{"index": 0, "model_window_frames": 121}],
        )

        self.assertTrue(
            self.store.continuation_work_is_active(
                self.job.job_id,
                1,
                revision,
            )
        )
        attempt = self.store.begin_chunk_attempt(
            self.job.job_id,
            1,
            revision,
            0,
            seed=123,
        )
        completed = self.store.update_chunk_attempt(
            self.job.job_id,
            1,
            revision,
            0,
            attempt.attempt_number,
            ChunkState.COMPLETE,
            artifact_hash="remake-handoff",
        )
        selected = self.store.select_chunk_attempt(
            self.job.job_id,
            1,
            revision,
            0,
            completed.attempt_number,
        )
        self.assertEqual(selected.accepted_artifact_hash, "remake-handoff")

    def test_qc_schema_upgrades_pre_qc_and_empty_databases_idempotently(self) -> None:
        legacy_path = Path(self.temporary_directory.name) / "legacy.sqlite3"
        connection = sqlite3.connect(legacy_path)
        connection.executescript(
            """
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL, status TEXT, final_path TEXT, updated_at TEXT
            );
            CREATE TABLE scenes (
                job_id TEXT NOT NULL, scene_id INTEGER NOT NULL, state TEXT NOT NULL,
                frame_path TEXT, video_path TEXT, error TEXT, t2i_attempts INTEGER DEFAULT 0,
                i2v_attempts INTEGER DEFAULT 0, prompt_id TEXT, prompt_stage TEXT,
                include_in_manual_final INTEGER DEFAULT 1, updated_at TEXT NOT NULL,
                PRIMARY KEY (job_id, scene_id)
            );
            CREATE TABLE scene_revisions (
                job_id TEXT NOT NULL, scene_id INTEGER NOT NULL, revision INTEGER NOT NULL,
                remake_mode TEXT NOT NULL, parameters_json TEXT NOT NULL, state TEXT NOT NULL,
                frame_path TEXT, video_path TEXT, error TEXT, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, PRIMARY KEY (job_id, scene_id, revision)
            );
            """
        )
        connection.commit()
        connection.close()

        legacy = PipelineStateStore(legacy_path)
        legacy.initialize()
        legacy.initialize()
        upgraded = sqlite3.connect(legacy_path)
        try:
            tables = {
                row[0]
                for row in upgraded.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        finally:
            upgraded.close()
        self.assertTrue(
            {"qc_candidates", "qc_evaluations", "qc_repairs", "qc_human_decisions"}
            <= tables
        )

        empty_path = Path(self.temporary_directory.name) / "empty.sqlite3"
        PipelineStateStore(empty_path).initialize()
        empty = sqlite3.connect(empty_path)
        try:
            self.assertIsNotNone(
                empty.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='qc_evaluations'"
                ).fetchone()
            )
        finally:
            empty.close()

    def test_qc_candidate_creation_is_unique_immutable_and_uses_text_seeds(self) -> None:
        candidate = self._create_qc_candidate()
        duplicate = self.store.ensure_qc_candidate(
            candidate_id=candidate.candidate_id,
            job_id=candidate.job_id,
            scene_id=candidate.scene_id,
            revision=candidate.revision,
            tier=candidate.tier,
            parent_candidate_id=candidate.parent_candidate_id,
            source_video_path=candidate.source_video_path,
            source_video_sha256=candidate.source_video_sha256,
            original_prompt=candidate.original_prompt,
            current_prompt=candidate.current_prompt,
            original_seed=candidate.original_seed,
            current_seed=candidate.current_seed,
            negative_prompt=candidate.negative_prompt,
            negative_prompt_sha256=candidate.negative_prompt_sha256,
            state=candidate.state,
            next_action=candidate.next_action,
        )

        self.assertEqual(duplicate, candidate)
        self.assertEqual(self.store.qc_candidates(self.job.job_id), (candidate,))
        connection = sqlite3.connect(self.store.database_path)
        try:
            stored = connection.execute(
                "SELECT original_seed, current_seed FROM qc_candidates"
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual(stored, (str(2**64 - 1), str(2**64 - 1)))

        with self.assertRaises(StateTransitionError):
            self.store.ensure_qc_candidate(
                candidate_id="different-id",
                job_id=candidate.job_id,
                scene_id=candidate.scene_id,
                revision=candidate.revision,
                tier=candidate.tier,
                parent_candidate_id=None,
                source_video_path=candidate.source_video_path,
                source_video_sha256="c" * 64,
                original_prompt=candidate.original_prompt,
                current_prompt=candidate.current_prompt,
                original_seed=candidate.original_seed,
                current_seed=candidate.current_seed,
                negative_prompt=candidate.negative_prompt,
                negative_prompt_sha256=candidate.negative_prompt_sha256,
                state=candidate.state,
                next_action=candidate.next_action,
            )

        with self.assertRaises(StateTransitionError):
            self.store.ensure_qc_candidate(
                candidate_id=candidate.candidate_id,
                job_id=candidate.job_id,
                scene_id=candidate.scene_id,
                revision=candidate.revision,
                tier=candidate.tier,
                parent_candidate_id=None,
                source_video_path=str(Path(self.temporary_directory.name) / "unbound.mp4"),
                source_video_sha256=candidate.source_video_sha256,
                original_prompt=candidate.original_prompt,
                current_prompt=candidate.current_prompt,
                original_seed=candidate.original_seed,
                current_seed=candidate.current_seed,
                negative_prompt=candidate.negative_prompt,
                negative_prompt_sha256=candidate.negative_prompt_sha256,
                state=candidate.state,
                next_action=candidate.next_action,
            )

    def test_qc_evaluation_persists_versions_raw_normalized_and_is_idempotent(self) -> None:
        candidate = self._create_qc_candidate()
        identity = {
            "evaluator_id": "production-qc",
            "evaluator_version": "phase1-v1",
            "backend_family": "llama.cpp",
            "backend_version": "2.28.2",
            "executable_path": "C:/llama-server.exe",
            "executable_sha256": "1" * 64,
            "model_id": "Qwen3.6 27B",
            "model_path": "C:/model.gguf",
            "model_sha256": "2" * 64,
            "quantization": "IQ3_M GGUF",
            "projector_path": "C:/mmproj.gguf",
            "projector_sha256": "3" * 64,
            "projector_precision": "FP16",
            "gpu_uuid": "GPU-stable",
            "gpu_name": "NVIDIA GeForce RTX 4080 SUPER",
        }
        idempotency_key = evaluation_idempotency_key(
            source_video_sha256=candidate.source_video_sha256,
            evaluator_id=identity["evaluator_id"],
            evaluator_version=identity["evaluator_version"],
            backend_version=identity["backend_version"],
            executable_sha256=identity["executable_sha256"],
            model_sha256=identity["model_sha256"],
            projector_sha256=identity["projector_sha256"],
            effective_config_sha256="4" * 64,
            prompt_sha256="5" * 64,
        )
        begun = self.store.begin_qc_evaluation(
            evaluation_id="evaluation-1",
            idempotency_key=idempotency_key,
            candidate_id=candidate.candidate_id,
            source_video_path=candidate.source_video_path,
            source_video_sha256=candidate.source_video_sha256,
            evaluator_identity=identity,
            effective_config={"sampling_fps": 2.0, "frames": 4},
            effective_config_sha256="4" * 64,
            prompt_version="production_ltx_video_qc_v1",
            prompt_sha256="5" * 64,
            sampling_config={"fps": 2.0},
            window_config={"frames": 4, "minimum_strong_windows": 2},
        )
        self.assertEqual(begun.state, "RUNNING")

        manifest_path = str(
            Path(candidate.source_video_path).parent
            / "qc" / "evaluations" / "evaluation-1" / "result.json"
        )
        completed = self.store.complete_qc_evaluation(
            "evaluation-1",
            raw_result="raw model response including malformed evidence",
            normalized_decision=QcDecision.UNCERTAIN,
            suspect_windows=[{"window": 2, "times": [2.0, 3.5]}],
            strong_window_count=1,
            frame_accounting={"unique_selected_frames_inspected": 12},
            evidence_manifest_path=manifest_path,
            evidence_manifest_sha256="6" * 64,
            next_action="hold_for_review",
        )
        repeated = self.store.complete_qc_evaluation(
            "evaluation-1",
            raw_result="raw model response including malformed evidence",
            normalized_decision=QcDecision.UNCERTAIN,
            suspect_windows=[{"window": 2, "times": [2.0, 3.5]}],
            strong_window_count=1,
            frame_accounting={"unique_selected_frames_inspected": 12},
            evidence_manifest_path=manifest_path,
            evidence_manifest_sha256="6" * 64,
            next_action="hold_for_review",
        )

        self.assertEqual(repeated, completed)
        restored = self.store.qc_evaluations(candidate.candidate_id)[0]
        self.assertEqual(restored.raw_result, "raw model response including malformed evidence")
        self.assertEqual(restored.normalized_decision, QcDecision.UNCERTAIN)
        self.assertEqual(restored.evaluator_identity["model_sha256"], "2" * 64)
        self.assertEqual(restored.prompt_version, "production_ltx_video_qc_v1")
        self.assertEqual(restored.sampling_config["fps"], 2.0)
        self.assertEqual(restored.frame_accounting["unique_selected_frames_inspected"], 12)

        with self.assertRaises(StateTransitionError):
            self.store.complete_qc_evaluation(
                "evaluation-1",
                raw_result="raw model response including malformed evidence",
                normalized_decision=QcDecision.UNCERTAIN,
                suspect_windows=[{"window": 2, "times": [2.0, 3.5]}],
                strong_window_count=1,
                frame_accounting={"unique_selected_frames_inspected": 12},
                evidence_manifest_path=str(
                    Path(candidate.source_video_path).parent / ".." / "escaped.json"
                ),
                evidence_manifest_sha256="6" * 64,
                next_action="hold_for_review",
            )

        restarted = PipelineStateStore(self.store.database_path)
        same = restarted.begin_qc_evaluation(
            evaluation_id="evaluation-1",
            idempotency_key=idempotency_key,
            candidate_id=candidate.candidate_id,
            source_video_path=candidate.source_video_path,
            source_video_sha256=candidate.source_video_sha256,
            evaluator_identity=identity,
            effective_config={"sampling_fps": 2.0, "frames": 4},
            effective_config_sha256="4" * 64,
            prompt_version="production_ltx_video_qc_v1",
            prompt_sha256="5" * 64,
            sampling_config={"fps": 2.0},
            window_config={"frames": 4, "minimum_strong_windows": 2},
        )
        self.assertEqual(same.evaluation_id, completed.evaluation_id)

        with self.assertRaises(StateTransitionError):
            restarted.begin_qc_evaluation(
                evaluation_id="evaluation-changed",
                idempotency_key=idempotency_key,
                candidate_id=candidate.candidate_id,
                source_video_path=candidate.source_video_path,
                source_video_sha256="f" * 64,
                evaluator_identity=identity,
                effective_config={},
                effective_config_sha256="4" * 64,
                prompt_version="production_ltx_video_qc_v1",
                prompt_sha256="5" * 64,
                sampling_config={"fps": 2.0},
                window_config={"frames": 4},
            )

    def test_qc_infrastructure_failure_is_bounded_without_new_candidate(self) -> None:
        candidate = self._create_qc_candidate()

        first = self.store.record_qc_infrastructure_failure(
            candidate.candidate_id, {"kind": "worker_exit", "code": 1}
        )
        second = self.store.record_qc_infrastructure_failure(
            candidate.candidate_id, {"kind": "worker_exit", "code": 2}
        )
        third = self.store.record_qc_infrastructure_failure(
            candidate.candidate_id, {"kind": "worker_exit", "code": 3}
        )

        self.assertEqual(first.infrastructure_failure_count, 1)
        self.assertEqual(first.state, QcCandidateState.PENDING_QC)
        self.assertEqual(second.infrastructure_failure_count, 2)
        self.assertEqual(second.state, QcCandidateState.HOLD_FOR_REVIEW)
        self.assertEqual(third, second)
        self.assertEqual(len(self.store.qc_candidates(self.job.job_id)), 1)

    def test_repair_and_human_evidence_are_append_only_and_promotion_requires_acceptance(self) -> None:
        candidate = self._create_qc_candidate()
        repair_manifest = str(
            Path(candidate.source_video_path).parent
            / "qc" / "repairs" / "repair-1" / "result.json"
        )
        repair = self.store.record_qc_repair(
            repair_id="repair-1",
            candidate_id=candidate.candidate_id,
            evaluation_id=None,
            planner_identity={"version": "future"},
            repair_input_sha256="7" * 64,
            raw_output="not run in this tranche",
            proposed_patch={},
            status="REJECTED",
            reason="B1 disabled",
            prior_repair_summaries=[],
            evidence_manifest_path=repair_manifest,
            evidence_manifest_sha256="8" * 64,
        )
        self.assertEqual(repair.status, "REJECTED")
        with self.assertRaises(StateTransitionError):
            self.store.record_qc_repair(
                repair_id="repair-1",
                candidate_id=candidate.candidate_id,
                evaluation_id=None,
                planner_identity={"version": "changed"},
                repair_input_sha256="7" * 64,
                raw_output="changed",
                proposed_patch={},
                status="REJECTED",
                reason="changed",
                prior_repair_summaries=[],
                evidence_manifest_path=repair_manifest,
                evidence_manifest_sha256="8" * 64,
            )

        decision = self.store.record_qc_human_decision(
            decision_id="decision-1",
            candidate_id=candidate.candidate_id,
            decision=QcHumanDecision.HOLD,
            note="canary hold",
            actor="local-owner",
            result_sha256="9" * 64,
            evidence_sha256="a" * 64,
        )
        self.assertEqual(decision.decision, QcHumanDecision.HOLD)
        with self.assertRaises(StateTransitionError):
            self.store.record_qc_human_decision(
                decision_id="decision-2",
                candidate_id=candidate.candidate_id,
                decision=QcHumanDecision.APPROVE,
                note=None,
                actor="local-owner",
                result_sha256="9" * 64,
                evidence_sha256="a" * 64,
            )
        with self.assertRaises(StateTransitionError):
            self.store.promote_accepted_qc_candidate(candidate.candidate_id)

        accepted = self.store.set_qc_candidate_state(
            candidate.candidate_id,
            QcCandidateState.ACCEPTED,
            next_action="promote",
        )
        self.store.promote_accepted_qc_candidate(accepted.candidate_id)
        self.assertEqual(
            self.store.scene_records(self.job.job_id)[0].video_path,
            candidate.source_video_path,
        )

    def _complete_pass_for_candidate(self, candidate_id: str) -> None:
        candidate = self.store.qc_candidate(candidate_id)
        identity = {
            "evaluator_id": "production-qc",
            "evaluator_version": "phase1-v1",
            "backend_family": "llama.cpp",
            "backend_version": "2.28.2",
            "executable_path": "C:/llama-server.exe",
            "executable_sha256": "1" * 64,
            "model_id": "Qwen3.6 27B",
            "model_path": "C:/model.gguf",
            "model_sha256": "2" * 64,
            "quantization": "IQ3_M GGUF",
            "projector_path": "C:/mmproj.gguf",
            "projector_sha256": "3" * 64,
            "projector_precision": "FP16",
            "gpu_uuid": "GPU-stable",
            "gpu_name": "NVIDIA GeForce RTX 4080 SUPER",
        }
        key = evaluation_idempotency_key(
            source_video_sha256=candidate.source_video_sha256,
            evaluator_id=identity["evaluator_id"],
            evaluator_version=identity["evaluator_version"],
            backend_version=identity["backend_version"],
            executable_sha256=identity["executable_sha256"],
            model_sha256=identity["model_sha256"],
            projector_sha256=identity["projector_sha256"],
            effective_config_sha256="4" * 64,
            prompt_sha256="5" * 64,
        )
        evaluation = self.store.begin_qc_evaluation(
            evaluation_id="evaluation-pass-" + candidate_id,
            idempotency_key=key,
            candidate_id=candidate_id,
            source_video_path=candidate.source_video_path,
            source_video_sha256=candidate.source_video_sha256,
            evaluator_identity=identity,
            effective_config={},
            effective_config_sha256="4" * 64,
            prompt_version="production_ltx_video_qc_v1",
            prompt_sha256="5" * 64,
            sampling_config={"fps": 2.0},
            window_config={"frames": 4},
        )
        self.store.complete_qc_evaluation(
            evaluation.evaluation_id,
            raw_result='{"decision":"PASS"}',
            normalized_decision=QcDecision.PASS,
            suspect_windows=[],
            strong_window_count=0,
            frame_accounting={"unique_selected_frames_inspected": 4},
            evidence_manifest_path=str(
                Path(candidate.source_video_path).parent
                / "qc" / "evaluations" / evaluation.evaluation_id / "result.json"
            ),
            evidence_manifest_sha256="6" * 64,
            next_action="pass_pending_human",
        )
        self.store.set_qc_candidate_state(
            candidate_id,
            QcCandidateState.PASS_PENDING_HUMAN,
            next_action="await_human",
        )

    def test_human_qc_decision_is_atomic_idempotent_and_rejects_conflicts(self) -> None:
        candidate = self._create_qc_candidate()
        self._complete_pass_for_candidate(candidate.candidate_id)

        approved = self.store.decide_qc_candidate(
            job_id=candidate.job_id,
            scene_id=candidate.scene_id,
            candidate_id=candidate.candidate_id,
            decision=QcHumanDecision.APPROVE,
            note="checked",
        )
        replay = self.store.decide_qc_candidate(
            job_id=candidate.job_id,
            scene_id=candidate.scene_id,
            candidate_id=candidate.candidate_id,
            decision=QcHumanDecision.APPROVE,
            note="checked",
        )

        self.assertEqual(approved.candidate.state, QcCandidateState.ACCEPTED)
        self.assertFalse(approved.replayed)
        self.assertTrue(replay.replayed)
        with self.assertRaisesRegex(StateTransitionError, "cannot be overwritten"):
            self.store.decide_qc_candidate(
                job_id=candidate.job_id,
                scene_id=candidate.scene_id,
                candidate_id=candidate.candidate_id,
                decision=QcHumanDecision.REJECT,
                note="changed",
            )
        with self.assertRaisesRegex(StateTransitionError, "route identity"):
            self.store.decide_qc_candidate(
                job_id=candidate.job_id,
                scene_id=99,
                candidate_id=candidate.candidate_id,
                decision=QcHumanDecision.APPROVE,
                note="checked",
            )

    def test_qc_final_selection_requires_durable_acceptance_and_kill_switch_uses_original(self) -> None:
        candidate = self._create_qc_candidate()
        self._complete_pass_for_candidate(candidate.candidate_id)
        with self.assertRaisesRegex(StateTransitionError, "ACCEPTED"):
            self.store.qc_final_selection(candidate.job_id, [candidate.scene_id])

        accepted = self.store.decide_qc_candidate(
            job_id=candidate.job_id,
            scene_id=candidate.scene_id,
            candidate_id=candidate.candidate_id,
            decision=QcHumanDecision.APPROVE,
            note=None,
        )
        selected = self.store.qc_final_selection(candidate.job_id, [candidate.scene_id])
        original = self.store.original_final_selection(candidate.job_id, [candidate.scene_id])

        self.assertEqual(selected[0].revision, accepted.candidate.revision)
        self.assertEqual(original[0].revision, 1)
        self.store.set_qc_candidate_state(
            candidate.candidate_id,
            QcCandidateState.HOLD_FOR_REVIEW,
            next_action="hold_for_review",
        )
        with self.assertRaisesRegex(StateTransitionError, "ACCEPTED"):
            self.store.qc_final_selection(candidate.job_id, [candidate.scene_id])
        self.assertEqual(
            self.store.original_final_selection(candidate.job_id, [candidate.scene_id]),
            original,
        )

    def test_qc_final_selection_rejects_accepted_path_with_changed_bytes(self) -> None:
        candidate = self._create_qc_candidate()
        self._complete_pass_for_candidate(candidate.candidate_id)
        self.store.decide_qc_candidate(
            job_id=candidate.job_id,
            scene_id=candidate.scene_id,
            candidate_id=candidate.candidate_id,
            decision=QcHumanDecision.APPROVE,
            note=None,
        )
        Path(candidate.source_video_path).write_bytes(b"overwritten-after-approval")

        with self.assertRaisesRegex(StateTransitionError, "hash"):
            self.store.qc_final_selection(candidate.job_id, [candidate.scene_id])

    def test_manual_final_cannot_select_latest_unapproved_qc_revision(self) -> None:
        candidate = self._create_qc_candidate()
        latest_video = Path(self.temporary_directory.name) / "latest-unapproved.mp4"
        latest_video.write_bytes(b"retry")
        revision = self.store.create_scene_revision(
            candidate.job_id,
            candidate.scene_id,
            remake_mode=RemakeMode.VIDEO_ONLY,
            parameters={"job_id": candidate.job_id, "scene_id": candidate.scene_id},
            state=SceneState.SUCCEEDED,
            frame_path=str(Path(self.temporary_directory.name) / "frame.png"),
            video_path=str(latest_video),
        )
        self.assertEqual(revision, 2)

        with self.assertRaisesRegex(StateTransitionError, "no successful video revision"):
            self.store.queue_manual_final(
                candidate.job_id, quality_control_enabled=True
            )
        kill_switch = self.store.queue_manual_final(
            candidate.job_id, quality_control_enabled=False
        )
        selection = self.store.manual_final_selection(kill_switch.request_id)
        self.assertEqual(selection[0].revision, 1)
        self.assertEqual(selection[0].video_path, candidate.source_video_path)

    def test_qc_tests_use_only_the_temporary_database(self) -> None:
        self._create_qc_candidate()
        self.assertNotIn(
            str(Path(r"D:\LTX_Supervisor_Storage")).casefold(),
            str(self.store.database_path).casefold(),
        )


if __name__ == "__main__":
    unittest.main()
