from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

try:
    from fastapi.testclient import TestClient
except ImportError:  # System Python intentionally does not host the GUI.
    TestClient = None

from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.state_store import PipelineStateStore
from tenminvideomaker.storage import StorageLayout

from test_contracts import payload


@unittest.skipUnless(TestClient is not None, "FastAPI is supplied by the embedded Python")
class GuiAppTests(unittest.TestCase):
    def test_library_scene_editor_and_remake_draft_are_structured(self) -> None:
        from tenminvideomaker.gui_app import create_gui_app

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = StorageLayout(root / "storage")
            store = PipelineStateStore(storage.database_path)
            job = parse_job_payload(payload())
            store.claim_job(job, review_required=True)

            class FakeComfy:
                def object_info(self, node_type):
                    values = ["lcm", "euler"] if node_type == "KSamplerSelect" else ["euler"]
                    return {
                        node_type: {
                            "input": {
                                "required": {
                                    "sampler_name": [values],
                                    "scheduler": [["karras", "normal"]],
                                }
                            }
                        }
                    }

            controller = SimpleNamespace(
                store=store,
                supervisor=SimpleNamespace(comfy=FakeComfy()),
                active_render=lambda: False,
                status_document=lambda: {
                    "pipeline_state": "awaiting_review",
                    "job_id": job.job_id,
                },
                approve_job=store.approve_job,
                queue_batch=lambda batch_id, policy: store.queue_remake_batch(
                    batch_id, policy
                ),
            )
            app = create_gui_app(controller, storage, Path(__file__).parents[1])
            client = TestClient(app)

            jobs = client.get("/api/jobs").json()
            self.assertEqual(jobs[0]["job_id"], job.job_id)
            scene = client.get(f"/api/jobs/{job.job_id}/scenes/1").json()
            self.assertIn("first_pass", scene["parameters"]["i2v"])
            self.assertNotIn("payload_json", scene)
            draft = client.post(
                "/api/remake-batches",
                json={
                    "items": [
                        {
                            "job_id": job.job_id,
                            "scene_id": 1,
                            "remake_mode": "image_and_video",
                            "parameters": scene["parameters"],
                        }
                    ]
                },
            )
            self.assertEqual(draft.status_code, 200)
            self.assertFalse(draft.json()["active_render"])


if __name__ == "__main__":
    unittest.main()
