from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.state_store import PipelineStateStore, SceneState
from tenminvideomaker.storage import StorageLayout, migrate_legacy_storage

from test_contracts import payload


class StorageTests(unittest.TestCase):
    def test_layout_versions_every_scene_artifact(self) -> None:
        layout = StorageLayout(Path(r"D:\LTX_Supervisor_Storage"))
        self.assertEqual(
            layout.scene_frame_path("job-1", 2, 3),
            Path(
                r"D:\LTX_Supervisor_Storage\jobs\job-1\scenes"
                r"\scene_0002\revisions\0003\frame.png"
            ),
        )
        self.assertEqual(
            layout.final_path("job-1"),
            Path(r"D:\LTX_Supervisor_Storage\finals\job-1_final.mp4"),
        )

    def test_migration_copies_database_payloads_and_media_without_deleting_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            layout = StorageLayout(root / "new-storage")
            legacy_output = root / "legacy-output"
            store = PipelineStateStore(project / "runtime" / "pipeline.sqlite3")
            job = parse_job_payload(payload())
            store.claim_job(job)
            old_frame = legacy_output / ".work" / job.job_id / "frames" / "scene_0001.png"
            old_clip = legacy_output / ".work" / job.job_id / "clips" / "scene_0001.mp4"
            old_final = legacy_output / f"{job.job_id}_final.mp4"
            for path, content in (
                (old_frame, b"frame"),
                (old_clip, b"clip"),
                (old_final, b"final"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            store.set_scene_state(
                job.job_id,
                1,
                SceneState.SUCCEEDED,
                frame_path=str(old_frame),
                video_path=str(old_clip),
            )
            (project / ".env").write_text(
                'TENMIN_GMAIL_USERNAME="owner@example.com"\n',
                encoding="utf-8",
            )
            (project / "runtime" / "secrets.json").write_text(
                '{"version": 1, "values": {}}\n',
                encoding="utf-8",
            )

            result = migrate_legacy_storage(
                project,
                layout=layout,
                legacy_output_root=legacy_output,
            )

            self.assertTrue(result["database"])
            self.assertEqual(layout.scene_frame_path(job.job_id, 1).read_bytes(), b"frame")
            self.assertEqual(layout.scene_clip_path(job.job_id, 1).read_bytes(), b"clip")
            self.assertEqual(layout.final_path(job.job_id).read_bytes(), b"final")
            self.assertEqual(
                json.loads(layout.source_payload_path(job.job_id).read_text(encoding="utf-8"))[
                    "job_id"
                ],
                job.job_id,
            )
            migrated = PipelineStateStore(layout.database_path).scene_records(job.job_id)[0]
            self.assertEqual(migrated.frame_path, str(layout.scene_frame_path(job.job_id, 1)))
            self.assertTrue(old_frame.exists())
            self.assertTrue(old_clip.exists())
            self.assertTrue(old_final.exists())
            self.assertTrue(layout.migration_marker.is_file())

    def test_empty_destination_database_cannot_block_legacy_history_migration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            layout = StorageLayout(root / "new-storage")
            legacy = PipelineStateStore(project / "runtime" / "pipeline.sqlite3")
            job = parse_job_payload(payload())
            legacy.claim_job(job)
            PipelineStateStore(layout.database_path).snapshot()

            result = migrate_legacy_storage(
                project,
                layout=layout,
                legacy_output_root=root / "legacy-output",
            )

            self.assertTrue(result["empty_database_preserved"])
            self.assertTrue(
                (layout.state_root / "pipeline.pre-migration-empty.sqlite3").is_file()
            )
            self.assertEqual(
                PipelineStateStore(layout.database_path).load_job(job.job_id).job_id,
                job.job_id,
            )


if __name__ == "__main__":
    unittest.main()
