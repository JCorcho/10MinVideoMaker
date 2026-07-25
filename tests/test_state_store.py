from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.state_store import PipelineState, PipelineStateStore, SceneState, StateTransitionError

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

    def test_job_cannot_be_claimed_twice(self) -> None:
        self.store.claim_job(self.job)
        with self.assertRaises(StateTransitionError):
            self.store.claim_job(self.job)

    def test_message_claim_is_idempotent(self) -> None:
        self.assertTrue(self.store.claim_message("imap:42"))
        self.assertFalse(self.store.claim_message("imap:42"))

    def test_inbound_job_claim_is_atomic_and_message_deduplicated(self) -> None:
        self.assertTrue(self.store.claim_inbound_job("imap:42", self.job))
        self.assertFalse(self.store.claim_inbound_job("imap:42", self.job))
        self.assertEqual(self.store.snapshot().state, PipelineState.DOWNLOADING_ASSETS)

    def test_previously_rejected_message_can_be_upgraded_to_a_job(self) -> None:
        self.assertTrue(self.store.claim_message("imap:42"))

        self.assertTrue(self.store.claim_inbound_job("imap:42", self.job))

        snapshot = self.store.snapshot()
        self.assertEqual(snapshot.state, PipelineState.DOWNLOADING_ASSETS)
        self.assertEqual(snapshot.job_id, self.job.job_id)

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
