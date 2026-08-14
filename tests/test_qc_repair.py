from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.qc_contracts import QcCandidateState, QcTier, derive_retry_seed
from tenminvideomaker.qc_repair import build_a1_document, schedule_a1_retry
from tenminvideomaker.review import scene_review_document
from tenminvideomaker.state_store import (
    PipelineStateStore,
    SceneState,
    StateTransitionError,
)
from tenminvideomaker.storage import StorageLayout
from test_contracts import payload


class A1RetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.layout = StorageLayout(root / "storage")
        self.store = PipelineStateStore(self.layout.database_path)
        self.job = parse_job_payload(payload())
        self.scene = self.job.scenes[0]
        self.document = scene_review_document(self.job, self.scene)
        self.store.claim_job(self.job)
        frame = self.layout.scene_frame_path(self.job.job_id, 1, 1)
        video = self.layout.scene_clip_path(self.job.job_id, 1, 1)
        frame.parent.mkdir(parents=True, exist_ok=True)
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
            parameters=self.document,
            frame_path=str(frame),
            video_path=str(video),
        )
        self.original = self.store.ensure_qc_candidate(
            candidate_id="candidate-original",
            job_id=self.job.job_id,
            scene_id=1,
            revision=1,
            tier=QcTier.ORIGINAL,
            parent_candidate_id=None,
            source_video_path=str(video),
            source_video_sha256="a" * 64,
            original_prompt=self.document["i2v"]["prompt"],
            current_prompt=self.document["i2v"]["prompt"],
            original_seed=int(self.document["i2v"]["seed"]),
            current_seed=int(self.document["i2v"]["seed"]),
            negative_prompt=self.document["i2v"]["negative"],
            negative_prompt_sha256="b" * 64,
            state=QcCandidateState.PENDING_QC,
            next_action="create_a1",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_a1_document_changes_only_seed_and_preserves_prompt_exactly(self) -> None:
        seed = 123456789
        revised = build_a1_document(self.document, seed=seed)

        self.assertEqual(revised["i2v"]["prompt"], self.document["i2v"]["prompt"])
        self.assertIsNot(revised, self.document)
        expected = deepcopy(self.document)
        expected["i2v"]["seed"] = str(seed)
        self.assertEqual(revised, expected)

    def test_a1_creates_one_video_only_revision_and_pending_candidate(self) -> None:
        retry = schedule_a1_retry(
            self.store,
            self.layout,
            original_job=self.job,
            source_candidate_id=self.original.candidate_id,
            source_document=self.document,
        )

        expected_seed = derive_retry_seed(
            job_id=self.job.job_id,
            scene_id=1,
            source_revision=1,
            original_seed=self.original.original_seed,
            tier=QcTier.A1,
            used_seeds=(self.original.current_seed,),
        )
        self.assertEqual(retry.seed, expected_seed)
        self.assertNotEqual(retry.seed, self.original.current_seed)
        self.assertEqual(retry.candidate.tier, QcTier.A1)
        self.assertEqual(retry.candidate.parent_candidate_id, self.original.candidate_id)
        self.assertEqual(retry.candidate.revision, 2)
        self.assertEqual(retry.candidate.state, QcCandidateState.PENDING_GENERATION)
        self.assertIsNone(retry.candidate.source_video_sha256)
        self.assertEqual(
            retry.candidate.source_video_path,
            str(self.layout.scene_clip_path(self.job.job_id, 1, 2)),
        )
        revisions = self.store.scene_revisions(self.job.job_id, 1)
        self.assertEqual(tuple(item.revision for item in revisions), (2, 1))
        self.assertEqual(revisions[0].remake_mode.value, "video_only")
        self.assertEqual(revisions[0].frame_path, revisions[1].frame_path)
        self.assertEqual(revisions[0].parameters, retry.document)

    def test_duplicate_a1_request_is_idempotent_and_does_not_create_retry_loop(self) -> None:
        first = schedule_a1_retry(
            self.store,
            self.layout,
            original_job=self.job,
            source_candidate_id=self.original.candidate_id,
            source_document=self.document,
        )
        second = schedule_a1_retry(
            PipelineStateStore(self.layout.database_path),
            self.layout,
            original_job=self.job,
            source_candidate_id=self.original.candidate_id,
            source_document=self.document,
        )

        self.assertEqual(second, first)
        self.assertEqual(
            len(self.store.qc_candidates(self.job.job_id, scene_id=1)), 2
        )
        self.assertEqual(len(self.store.scene_revisions(self.job.job_id, 1)), 2)

    def test_generation_completion_adds_hash_without_overwriting_original(self) -> None:
        retry = schedule_a1_retry(
            self.store,
            self.layout,
            original_job=self.job,
            source_candidate_id=self.original.candidate_id,
            source_document=self.document,
        )
        Path(retry.candidate.source_video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(retry.candidate.source_video_path).write_bytes(b"a1-video")

        a1_hash = hashlib.sha256(b"a1-video").hexdigest()
        completed = self.store.complete_qc_candidate_generation(
            retry.candidate.candidate_id,
            source_video_path=retry.candidate.source_video_path,
            source_video_sha256=a1_hash,
        )

        self.assertEqual(completed.state, QcCandidateState.PENDING_QC)
        self.assertEqual(completed.source_video_sha256, a1_hash)
        self.assertEqual(self.original.source_video_sha256, "a" * 64)
        with self.assertRaises(StateTransitionError):
            self.store.complete_qc_candidate_generation(
                retry.candidate.candidate_id,
                source_video_path=retry.candidate.source_video_path,
                source_video_sha256="d" * 64,
            )

    def test_b1_is_not_reachable_in_this_tranche(self) -> None:
        import tenminvideomaker.qc_repair as module

        self.assertFalse(hasattr(module, "build_b1_document"))
        self.assertFalse(hasattr(module, "parse_and_validate_b1_patch"))
        self.assertFalse(hasattr(module, "apply_b1_patch"))


if __name__ == "__main__":
    unittest.main()
