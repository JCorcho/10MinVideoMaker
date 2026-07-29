from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.state_store import (
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
        self.store.set_scene_prompt_id(self.job.job_id, 1, "prompt-123")
        record = self.store.scene_records(self.job.job_id)[0]
        self.assertEqual(attempt, 1)
        self.assertEqual(record.t2i_attempts, 1)
        self.assertEqual(record.i2v_attempts, 0)
        self.assertEqual(record.prompt_id, "prompt-123")
        self.assertEqual(self.store.snapshot().active_scene_id, 1)

    def test_retry_job_preserves_attempts_and_requeues_unfinished_scene(self) -> None:
        self.store.claim_job(self.job)
        self.store.begin_scene_stage(
            self.job.job_id,
            1,
            PipelineState.RUNNING_T2I,
        )
        self.store.set_scene_prompt_id(self.job.job_id, 1, "prompt-123")
        self.store.set_scene_state(
            self.job.job_id,
            1,
            SceneState.FAILED,
            error="authentication required",
        )
        self.store.transition(
            PipelineState.ERROR,
            job_id=self.job.job_id,
            error="assets failed",
        )

        self.assertEqual(self.store.retry_job(self.job.job_id), [1])

        snapshot = self.store.snapshot()
        record = self.store.scene_records(self.job.job_id)[0]
        self.assertEqual(snapshot.state, PipelineState.DOWNLOADING_ASSETS)
        self.assertEqual(snapshot.job_id, self.job.job_id)
        self.assertEqual(record.state, SceneState.PENDING)
        self.assertIsNone(record.error)
        self.assertIsNone(record.prompt_id)
        self.assertEqual(record.t2i_attempts, 1)

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
        self.store.set_scene_prompt_id(self.job.job_id, 1, "prompt-123")
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


if __name__ == "__main__":
    unittest.main()
