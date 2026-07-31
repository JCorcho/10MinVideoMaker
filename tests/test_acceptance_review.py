from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tenminvideomaker.acceptance_review import (
    AcceptanceReviewError,
    AcceptanceReviewProxyError,
    AcceptanceReviewService,
)
from tenminvideomaker.storage import StorageLayout


RUN_ID = "continuation-acceptance-20260731-065935"


class AcceptanceReviewServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.storage = StorageLayout(Path(self.temporary.name) / "storage")
        self.storage.ensure()
        self.run_root = self.storage.root / "acceptance" / RUN_ID
        self.run_root.mkdir(parents=True)
        self.raw_paths = {
            "common_base": self.storage.chunk_video_path(RUN_ID, 1, 1, 0, 1),
            "single_frame": self.storage.chunk_video_path(RUN_ID, 1, 2, 0, 1),
            "decoded_17_frame": self.storage.chunk_video_path(RUN_ID, 1, 3, 0, 1),
            "latent_overlap": self.storage.chunk_video_path(RUN_ID, 1, 1, 1, 1),
        }
        for raw_path in self.raw_paths.values():
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(b"unwatermarked-raw-window")
        self._write_run_document()
        self._write_required_stills()
        self.service = AcceptanceReviewService(self.storage)

    def _write_run_document(self) -> None:
        document = {
            "run_id": RUN_ID,
            "state": "awaiting_human_review",
            "case_order": [
                "common_base",
                "single_frame",
                "decoded_17_frame",
                "latent_overlap",
            ],
            "cases": {
                name: {"stage2": {"raw_video_path": str(path)}}
                for name, path in self.raw_paths.items()
            },
        }
        (self.run_root / "run.json").write_text(
            json.dumps(document), encoding="utf-8"
        )

    def _write_required_stills(self) -> None:
        stills = {
            "single_frame": [
                "base_0119.png",
                "base_0120.png",
                "case_0000.png",
                "case_0001.png",
            ],
            "decoded_17_frame": [
                "base_0096.png",
                "base_0111.png",
                "base_0112.png",
                "case_0000.png",
                "case_0016.png",
                "case_0017.png",
            ],
            "latent_overlap": [
                "base_0102.png",
                "base_0103.png",
                "case_0008.png",
                "case_0009.png",
            ],
        }
        for case_name, filenames in stills.items():
            directory = self.run_root / "metrics" / case_name
            directory.mkdir(parents=True, exist_ok=True)
            for filename in filenames:
                (directory / filename).write_bytes(b"png")

    def test_review_document_labels_boundaries_and_never_exposes_raw_paths(self) -> None:
        document = self.service.review_document(RUN_ID)

        self.assertEqual(document["run_id"], RUN_ID)
        self.assertEqual(document["base"]["role"], "base")
        self.assertEqual(
            document["cases"]["single_frame"]["boundary"],
            {"left": [119, 120], "right": [0, 1]},
        )
        self.assertEqual(
            document["cases"]["decoded_17_frame"]["boundary"],
            {"left": [111, 112], "right": [16, 17]},
        )
        self.assertEqual(
            document["cases"]["latent_overlap"]["boundary"],
            {"left": [102, 103], "right": [8, 9]},
        )
        self.assertIn("/api/acceptance-runs/", document["base"]["video_url"])
        self.assertNotIn(str(self.storage.root), json.dumps(document))
        self.assertNotIn("common_base", document["case_order"])

    def test_review_proxy_is_h264_atomic_unwatermarked_and_reused(self) -> None:
        source = self.raw_paths["common_base"]

        def fake_run(command, **_kwargs):
            destination = Path(command[-1])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"browser-safe-review-proxy")
            return subprocess.CompletedProcess(command, 0)

        with patch(
            "tenminvideomaker.acceptance_review.subprocess.run",
            side_effect=fake_run,
        ) as run:
            first = self.service.review_proxy_path(RUN_ID, "base")
            second = self.service.review_proxy_path(RUN_ID, "base")

        command = run.call_args.args[0]
        self.assertEqual(run.call_count, 1)
        self.assertEqual(first, second)
        self.assertTrue(first.is_file())
        self.assertEqual(first.read_bytes(), b"browser-safe-review-proxy")
        self.assertEqual(source.read_bytes(), b"unwatermarked-raw-window")
        self.assertIn("-c:v", command)
        self.assertIn("libx264", command)
        self.assertIn("-c:a", command)
        self.assertIn("aac", command)
        self.assertNotIn("watermark", " ".join(command).casefold())

    def test_review_document_describes_production_faithful_assembly(self) -> None:
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
        self.assertEqual(
            document["cases"]["single_frame"]["assembled_video_url"],
            f"/api/acceptance-runs/{RUN_ID}/assembled/single_frame",
        )
        self.assertEqual(
            document["cases"]["decoded_17_frame"]["assembly"],
            {
                "base_end_frame": 112,
                "continuation_start_frame": 17,
                "dropped_continuation_frames": [0, 16],
                "summary": "Base 0–112, then continuation 17 onward.",
            },
        )
        self.assertEqual(
            document["cases"]["latent_overlap"]["assembly"],
            {
                "base_end_frame": 103,
                "continuation_start_frame": 8,
                "dropped_continuation_frames": [0, 7],
                "summary": "Base 0–103, then continuation 8 onward.",
            },
        )

    def test_assembled_proxy_trims_overlap_and_concats_atomically(self) -> None:
        def fake_run(command, **_kwargs):
            destination = Path(command[-1])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"assembled-review-proxy")
            return subprocess.CompletedProcess(command, 0)

        with patch(
            "tenminvideomaker.acceptance_review.subprocess.run",
            side_effect=fake_run,
        ) as run:
            first = self.service.assembled_proxy_path(RUN_ID, "single_frame")
            second = self.service.assembled_proxy_path(RUN_ID, "single_frame")

        command = run.call_args.args[0]
        filter_graph = command[command.index("-filter_complex") + 1]
        self.assertEqual(run.call_count, 1)
        self.assertEqual(first, second)
        self.assertEqual(first.name, "assembled-single_frame.mp4")
        self.assertIn("trim=start_frame=0:end_frame=121", filter_graph)
        self.assertIn("trim=start_frame=1", filter_graph)
        self.assertIn("concat=n=2:v=1:a=0", filter_graph)
        self.assertEqual(command[command.index("-map") + 1], "[outv]")
        self.assertNotIn("-c:a", command)
        self.assertNotIn("watermark", " ".join(command).casefold())

    def test_assembled_proxy_rejects_unknown_case(self) -> None:
        with self.assertRaisesRegex(AcceptanceReviewError, "Unknown review case"):
            self.service.assembled_proxy_path(RUN_ID, "unknown")

    def test_assembled_proxy_failure_never_publishes_partial_media(self) -> None:
        with patch(
            "tenminvideomaker.acceptance_review.subprocess.run",
            return_value=subprocess.CompletedProcess(["ffmpeg"], 1, stderr="failed"),
        ):
            with self.assertRaisesRegex(AcceptanceReviewProxyError, "FFmpeg"):
                self.service.assembled_proxy_path(RUN_ID, "decoded_17_frame")

        review_root = self.run_root / "review"
        self.assertFalse((review_root / "assembled-decoded_17_frame.mp4").exists())
        self.assertFalse(
            (review_root / "assembled-decoded_17_frame.partial.mp4").exists()
        )

    def test_review_rejects_traversal_outside_storage_and_missing_artifacts(self) -> None:
        with self.assertRaisesRegex(AcceptanceReviewError, "run ID"):
            self.service.review_document("../../jobs/other")

        document_path = self.run_root / "run.json"
        document = json.loads(document_path.read_text(encoding="utf-8"))
        document["cases"]["single_frame"]["stage2"]["raw_video_path"] = str(
            Path(self.temporary.name) / "outside.mkv"
        )
        document_path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(AcceptanceReviewError, "outside project storage"):
            self.service.review_document(RUN_ID)

        self._write_run_document()
        self.raw_paths["decoded_17_frame"].unlink()
        with self.assertRaisesRegex(AcceptanceReviewError, "missing"):
            self.service.review_document(RUN_ID)

    def test_proxy_failure_never_publishes_partial_media(self) -> None:
        with patch(
            "tenminvideomaker.acceptance_review.subprocess.run",
            return_value=subprocess.CompletedProcess(["ffmpeg"], 1, stderr="failed"),
        ):
            with self.assertRaisesRegex(AcceptanceReviewProxyError, "FFmpeg"):
                self.service.review_proxy_path(RUN_ID, "single_frame")

        self.assertFalse(
            (self.run_root / "review" / "single_frame.mp4").exists()
        )


if __name__ == "__main__":
    unittest.main()
