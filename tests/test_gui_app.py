from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

try:
    from fastapi.testclient import TestClient
except ImportError:  # System Python intentionally does not host the GUI.
    TestClient = None

from tenminvideomaker.contracts import parse_job_payload
from tenminvideomaker.state_store import PipelineStateStore
from tenminvideomaker.storage import StorageLayout

from test_contracts import payload


class GuiStaticAssetTests(unittest.TestCase):
    def test_project_and_scene_lists_have_independent_scroll_containers(self) -> None:
        web_root = Path(__file__).parents[1] / "web"
        styles = (web_root / "styles.css").read_text(encoding="utf-8")
        script = (web_root / "app.js").read_text(encoding="utf-8")

        self.assertRegex(
            styles,
            r"(?s)\.panel\s*\{[^}]*min-height:\s*0;[^}]*display:\s*flex;",
        )
        self.assertRegex(
            styles,
            r"(?s)\.scroll-list\s*\{[^}]*flex:\s*1 1 auto;"
            r"[^}]*min-height:\s*0;[^}]*overflow-y:\s*auto;",
        )
        self.assertIn("job.display_name || job.job_id", script)
        self.assertIn("summary?.display_name || job.job_id", script)
        self.assertIn("@media (max-width: 760px)", styles)
        self.assertIn("data-lora-picker", script)
        self.assertIn("scrollIntoView", script)


@unittest.skipUnless(TestClient is not None, "FastAPI is supplied by the embedded Python")
class GuiAppTests(unittest.TestCase):
    def test_gui_review_hold_flag_is_opt_in(self) -> None:
        from scripts.run_gui import _gui_binding, argument_parser

        parser = argument_parser()
        self.assertFalse(parser.parse_args([]).hold_new_jobs_for_review)
        self.assertTrue(
            parser.parse_args(["--hold-new-jobs-for-review"]).hold_new_jobs_for_review
        )
        self.assertTrue(parser.parse_args(["--lan"]).lan)
        with self.assertRaisesRegex(SystemExit, "12\\+ character password"):
            _gui_binding(parser.parse_args(["--lan"]), {})
        self.assertEqual(
            _gui_binding(
                parser.parse_args(["--lan"]),
                {"TENMIN_GUI_LAN_PASSWORD": "mobile-password"},
            ),
            ("0.0.0.0", "mobile-password"),
        )

    def test_gui_launcher_uses_shared_comfy_startup_guard(self) -> None:
        from scripts.run_gui import ensure_comfyui
        from scripts.setup_and_start import ensure_comfyui as shared_guard

        self.assertIs(ensure_comfyui, shared_guard)

    def test_launcher_restarts_only_for_stale_contract_and_empty_queue(self) -> None:
        from scripts.run_gui import _ensure_current_node_contract
        from tenminvideomaker.ownership import OwnershipError

        stale = Mock()
        stale.object_info.return_value = {
            "10MinVideoMaker_SaveSceneFrame": {
                "input": {"required": {"job_id": ["STRING"]}}
            }
        }
        stale.queue_counts.return_value = (1, 0)
        with self.assertRaisesRegex(OwnershipError, "queue is busy"):
            _ensure_current_node_contract(stale)

        stale.queue_counts.return_value = (0, 0)
        stale.object_info.side_effect = [
            {
                "10MinVideoMaker_SaveSceneFrame": {
                    "input": {"required": {"job_id": ["STRING"]}}
                }
            },
            {
                "10MinVideoMaker_SaveSceneFrame": {
                    "input": {
                        "required": {
                            "job_id": ["STRING"],
                            "revision": ["INT"],
                        }
                    }
                }
            },
        ]
        with patch("scripts.run_gui.restart_comfyui", return_value=True) as restart:
            _ensure_current_node_contract(stale)
        restart.assert_called_once_with()

    def test_library_scene_editor_and_remake_draft_are_structured(self) -> None:
        from tenminvideomaker.gui_app import create_gui_app

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = StorageLayout(root / "storage")
            store = PipelineStateStore(storage.database_path)
            source = payload()
            source["created_at"] = "2026-07-24T16:10:45Z"
            job = parse_job_payload(source)
            store.claim_job(job, review_required=True)

            class FakeComfy:
                def object_info(self, node_type):
                    if node_type == "LoraLoader":
                        return {
                            node_type: {
                                "input": {"required": {"lora_name": [["Elsa.safetensors"]]}}
                            }
                        }
                    if node_type == "LoraLoaderModelOnly":
                        return {
                            node_type: {
                                "input": {
                                    "required": {"lora_name": [["LTX/Dance.safetensors"]]}
                                }
                            }
                        }
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

            options = client.get("/api/options").json()
            self.assertEqual(options["t2i_loras"], ["Elsa.safetensors"])
            self.assertEqual(options["i2v_loras"], ["LTX/Dance.safetensors"])

            jobs = client.get("/api/jobs").json()
            self.assertEqual(jobs[0]["job_id"], job.job_id)
            self.assertEqual(jobs[0]["display_name"], "Elsa · 07/24/2026")
            job_detail = client.get(f"/api/jobs/{job.job_id}").json()
            self.assertEqual(job_detail["display_name"], "Elsa · 07/24/2026")
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

            secured_client = TestClient(
                create_gui_app(
                    controller,
                    storage,
                    Path(__file__).parents[1],
                    lan_password="mobile-password",
                )
            )
            self.assertEqual(secured_client.get("/api/status").status_code, 401)
            credentials = base64.b64encode(b"10min:mobile-password").decode("ascii")
            self.assertEqual(
                secured_client.get(
                    "/api/status",
                    headers={"Authorization": f"Basic {credentials}"},
                ).status_code,
                200,
            )


if __name__ == "__main__":
    unittest.main()
