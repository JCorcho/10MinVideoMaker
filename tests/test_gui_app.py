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
from tenminvideomaker.state_store import PipelineState, PipelineStateStore, SceneState
from tenminvideomaker.storage import StorageLayout

from test_contracts import payload


class GuiStaticAssetTests(unittest.TestCase):
    def test_project_and_scene_lists_have_independent_scroll_containers(self) -> None:
        web_root = Path(__file__).parents[1] / "web"
        styles = (web_root / "styles.css").read_text(encoding="utf-8")
        script = (web_root / "app.js").read_text(encoding="utf-8")
        markup = (web_root / "index.html").read_text(encoding="utf-8")

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
        self.assertIn("selectRevisionParameters", script)
        self.assertIn("revision?.parameters", script)
        self.assertIn('addEventListener("change", selectRevisionParameters)', script)
        self.assertIn("@media (max-width: 760px)", styles)
        self.assertIn("data-lora-picker", script)
        self.assertIn("scrollMobilePanel", script)

        self.assertIn('data-mobile-view="projects"', markup)
        self.assertIn('id="back-to-projects"', markup)
        self.assertIn('id="mobile-scene-switcher"', markup)
        self.assertIn('id="mobile-scene-select"', markup)
        self.assertIn('id="back-to-scenes"', markup)
        self.assertIn("backToProjects", script)
        self.assertIn("backToScenes", script)
        self.assertIn("renderMobileScenePicker", script)
        self.assertIn('id="render-project-final"', markup)
        self.assertIn('id="include-in-manual-final"', markup)
        self.assertIn('id="cancel-current-project"', markup)
        self.assertIn("queueManualFinal", script)
        self.assertIn("setManualFinalInclusion", script)
        self.assertIn("cancelCurrentProject", script)
        self.assertIn("/api/pipeline/cancel-current", script)
        self.assertIn("can_cancel_current_project", script)
        self.assertIn('body[data-mobile-view="projects"] .library-panel', styles)
        self.assertIn('body[data-mobile-view="scenes"] .scenes-panel', styles)
        self.assertIn('body[data-mobile-view="detail"] .detail-panel', styles)

        self.assertRegex(markup, r'<video[^>]*controls[^>]*playsinline[^>]*webkit-playsinline')
        self.assertRegex(
            styles,
            r"(?s)\.media-stage video\s*\{[^}]*width:\s*100%;[^}]*height:\s*100%;",
        )


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
            clip = storage.scene_clip_path(job.job_id, 1, 1)
            clip.parent.mkdir(parents=True, exist_ok=True)
            clip.write_bytes(b"mp4")
            store.set_scene_state(
                job.job_id,
                1,
                SceneState.SUCCEEDED,
                video_path=str(clip),
            )

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

            def cancel_current_project() -> dict[str, object]:
                cancelled = store.abandon_job(
                    job.job_id,
                    reason="test cancel",
                )
                return {
                    "job_id": job.job_id,
                    "cancelled_prompts": [],
                    "pipeline_state": "idle",
                    "cancelled_scenes": cancelled,
                }

            controller = SimpleNamespace(
                store=store,
                supervisor=SimpleNamespace(comfy=FakeComfy()),
                active_render=lambda: False,
                status_document=lambda: {
                    "pipeline_state": "awaiting_review",
                    "job_id": job.job_id,
                    "can_cancel_current_project": True,
                },
                approve_job=store.approve_job,
                cancel_current_project=cancel_current_project,
                queue_batch=lambda batch_id, policy: store.queue_remake_batch(
                    batch_id, policy
                ),
                queue_manual_final=store.queue_manual_final,
            )
            app = create_gui_app(controller, storage, Path(__file__).parents[1])
            client = TestClient(app)

            options = client.get("/api/options").json()
            self.assertEqual(options["t2i_loras"], ["Elsa.safetensors"])
            self.assertEqual(options["i2v_loras"], ["LTX/Dance.safetensors"])

            jobs = client.get("/api/jobs").json()
            self.assertEqual(jobs[0]["job_id"], job.job_id)
            self.assertEqual(jobs[0]["display_name"], "Elsa · 07/24/2026")
            status = client.get("/api/status").json()
            self.assertTrue(status["can_cancel_current_project"])
            cancelled = client.post("/api/pipeline/cancel-current")
            self.assertEqual(cancelled.status_code, 200)
            self.assertEqual(cancelled.json()["job_id"], job.job_id)
            self.assertTrue(cancelled.json()["cancelled"])
            self.assertEqual(store.snapshot().state, PipelineState.IDLE)
            # Restore scene success for the remake/manual-final assertions below.
            # The cancelled job history remains; only unfinished work was abandoned.
            store.set_scene_state(
                job.job_id,
                1,
                SceneState.SUCCEEDED,
                video_path=str(clip),
            )
            job_detail = client.get(f"/api/jobs/{job.job_id}").json()
            self.assertEqual(job_detail["display_name"], "Elsa · 07/24/2026")
            scene = client.get(f"/api/jobs/{job.job_id}/scenes/1").json()
            self.assertIn("first_pass", scene["parameters"]["i2v"])
            self.assertNotIn("payload_json", scene)
            original_prompt = scene["parameters"]["t2i"]["prompt"]
            remake_parameters = {
                **scene["parameters"],
                "t2i": {
                    **scene["parameters"]["t2i"],
                    "prompt": "A deliberately revised scene prompt.",
                },
                "i2v": {
                    **scene["parameters"]["i2v"],
                    "first_pass": {
                        **scene["parameters"]["i2v"]["first_pass"],
                        "sampler": "euler",
                    },
                },
            }
            draft = client.post(
                "/api/remake-batches",
                json={
                    "items": [
                        {
                            "job_id": job.job_id,
                            "scene_id": 1,
                            "remake_mode": "image_and_video",
                            "parameters": remake_parameters,
                        }
                    ]
                },
            )
            self.assertEqual(draft.status_code, 200)
            self.assertFalse(draft.json()["active_render"])
            scene_after_remake = client.get(
                f"/api/jobs/{job.job_id}/scenes/1"
            ).json()
            revisions = {
                item["revision"]: item
                for item in scene_after_remake["revisions"]
            }
            self.assertEqual(revisions[1]["parameters"]["t2i"]["prompt"], original_prompt)
            self.assertEqual(
                revisions[2]["parameters"]["t2i"]["prompt"],
                "A deliberately revised scene prompt.",
            )
            self.assertEqual(
                revisions[2]["parameters"]["i2v"]["first_pass"]["sampler"],
                "euler",
            )
            excluded = client.put(
                f"/api/jobs/{job.job_id}/scenes/1/manual-final-inclusion",
                json={"included": False},
            )
            self.assertEqual(excluded.status_code, 200)
            self.assertFalse(excluded.json()["include_in_manual_final"])
            unavailable_final = client.post(f"/api/jobs/{job.job_id}/manual-final")
            self.assertEqual(unavailable_final.status_code, 409)
            included = client.put(
                f"/api/jobs/{job.job_id}/scenes/1/manual-final-inclusion",
                json={"included": True},
            )
            self.assertEqual(included.status_code, 200)
            manual_final = client.post(f"/api/jobs/{job.job_id}/manual-final")
            self.assertEqual(manual_final.status_code, 200)
            self.assertEqual(manual_final.json()["state"], "queued")

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
