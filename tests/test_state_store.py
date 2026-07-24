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


if __name__ == "__main__":
    unittest.main()
