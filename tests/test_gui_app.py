from __future__ import annotations

import base64
import json
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
    def test_acceptance_review_page_shows_labeled_boundary_comparison(self) -> None:
        web_root = Path(__file__).parents[1] / "web"
        markup = (web_root / "acceptance-review.html").read_text(encoding="utf-8")
        script = (web_root / "acceptance-review.js").read_text(encoding="utf-8")
        styles = (web_root / "acceptance-review.css").read_text(encoding="utf-8")
        index = (web_root / "index.html").read_text(encoding="utf-8")

        self.assertRegex(
            markup,
            r'<video[^>]*id="base-video"[^>]*controls[^>]*playsinline[^>]*webkit-playsinline',
        )
        self.assertRegex(
            markup,
            r'<video[^>]*id="case-video"[^>]*controls[^>]*playsinline[^>]*webkit-playsinline',
        )
        self.assertRegex(
            markup,
            r'<video[^>]*id="assembled-video"[^>]*controls[^>]*playsinline[^>]*webkit-playsinline',
        )
        self.assertIn('id="show-seam"', markup)
        self.assertIn('id="visual-checklist"', markup)
        self.assertIn("Show exact seam", markup)
        self.assertIn("case_order", script)
        self.assertIn("assembled_video_url", script)
        self.assertIn("assembly.summary", script)
        self.assertIn("currentTime", script)
        self.assertIn("/api/acceptance-runs", script)
        self.assertIn(".assembled-video-card", styles)
        self.assertRegex(
            styles,
            r"(?s)\.comparison-grid\s*\{[^}]*grid-template-columns:\s*repeat\(2,",
        )
        self.assertRegex(
            styles,
            r"(?s)@media \(max-width: 760px\).*?"
            r"\.comparison-grid\s*\{[^}]*grid-template-columns:\s*1fr;",
        )
        self.assertRegex(
            styles,
            r"(?s)\.video-stage video\s*\{[^}]*width:\s*100%;[^}]*height:\s*auto;",
        )
        self.assertIn('id="acceptance-review-link"', index)

    def test_chunk_lineage_is_collapsible_human_readable_and_mobile_ready(self) -> None:
        web_root = Path(__file__).parents[1] / "web"
        styles = (web_root / "styles.css").read_text(encoding="utf-8")
        script = (web_root / "app.js").read_text(encoding="utf-8")
        markup = (web_root / "index.html").read_text(encoding="utf-8")

        self.assertIn('id="chunk-lineage"', markup)
        self.assertIn("function renderChunkLineage(progress)", script)
        self.assertIn("progress?.chunks", script)
        self.assertIn("chunk-lineage-card", script)
        self.assertIn("chunk-attempt-card", script)
        for label in (
            "Resolved prompt",
            "Resolved negative",
            "Accepted attempt",
            "Workflow hashes",
            "Artifact hashes",
            "Model files",
            "Node contracts",
        ):
            self.assertIn(label, script)
        self.assertNotIn("JSON.stringify(progress.chunks", script)
        self.assertIn(".chunk-lineage-grid", styles)
        self.assertRegex(
            styles,
            r"(?s)@media \(max-width: 760px\).*?"
            r"\.chunk-lineage-grid\s*\{[^}]*grid-template-columns:\s*1fr;",
        )

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
        self.assertIn("ltx23_exact_frame_handoff_v2", script)
        self.assertIn("Exact final-frame handoff", script)
        self.assertIn("120 new frames per continuation", script)
        self.assertIn("Force chunked continuation", script)
        self.assertIn("Ordered scene beats", script)
        self.assertIn('key === "seed_override"', script)
        self.assertIn("segment[key] = input.value.trim()", script)
        self.assertIn("normalizeSceneParameters", script)
        self.assertIn("document.i2v.segments = []", script)
        self.assertIn(
            "delete continuation.requested_duration_seconds",
            script,
        )
        self.assertIn("refreshChunkProgress", script)
        self.assertIn("exact final timeline frames", script)
        self.assertIn("LTX generation-master frames", script)
        self.assertIn("resolvedContinuationRoute", script)
        self.assertIn("Beat coverage", script)
        self.assertIn("data-segment-timing-mode", script)
        self.assertIn("LTX character LoRA (video)", script)
        self.assertIn('id="temporal-continuation-wrap"', markup)
        self.assertIn('id="continuation-status"', markup)
        self.assertIn('id="ltx-character-lora"', markup)
        self.assertIn(".continuation-segment-card", styles)
        self.assertIn("--mobile-scene-switcher-height", styles)
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
    def test_duplicate_gui_launch_opens_existing_instance_and_exits_cleanly(self) -> None:
        from scripts.run_gui import main
        from tenminvideomaker.ownership import OwnershipError

        class CollidingLock:
            def acquire(self) -> None:
                raise OwnershipError(
                    "Another 10MinVideoMaker controller is already running."
                )

            def __enter__(self):
                self.acquire()

            def __exit__(self, *_args):
                return None

        storage = SimpleNamespace(instance_lock_path=Path("supervisor.lock"))
        with (
            patch("scripts.run_gui.load_project_environment", return_value={}),
            patch("scripts.run_gui._gui_binding", return_value=("127.0.0.1", None)),
            patch("scripts.run_gui.StorageLayout.configured", return_value=storage),
            patch("scripts.run_gui.SupervisorInstanceLock", return_value=CollidingLock()),
            patch("scripts.run_gui.webbrowser.open") as open_browser,
        ):
            result = main([])

        self.assertEqual(result, 0)
        open_browser.assert_called_once_with("http://127.0.0.1:8765/")

    def test_duplicate_gui_launch_respects_no_browser(self) -> None:
        from scripts.run_gui import main
        from tenminvideomaker.ownership import OwnershipError

        class CollidingLock:
            def acquire(self) -> None:
                raise OwnershipError(
                    "Another 10MinVideoMaker controller is already running."
                )

            def __enter__(self):
                self.acquire()

            def __exit__(self, *_args):
                return None

        storage = SimpleNamespace(instance_lock_path=Path("supervisor.lock"))
        with (
            patch("scripts.run_gui.load_project_environment", return_value={}),
            patch("scripts.run_gui._gui_binding", return_value=("127.0.0.1", None)),
            patch("scripts.run_gui.StorageLayout.configured", return_value=storage),
            patch("scripts.run_gui.SupervisorInstanceLock", return_value=CollidingLock()),
            patch("scripts.run_gui.webbrowser.open") as open_browser,
        ):
            result = main(["--no-browser"])

        self.assertEqual(result, 0)
        open_browser.assert_not_called()

    def test_review_only_gui_serves_acceptance_page_without_supervisor(self) -> None:
        from tenminvideomaker.gui_app import create_acceptance_review_app

        run_id = "continuation-acceptance-20260731-065935"
        with tempfile.TemporaryDirectory() as directory:
            storage = StorageLayout(Path(directory) / "storage")
            storage.ensure()
            run_root = storage.root / "acceptance" / run_id
            run_root.mkdir(parents=True)
            (run_root / "run.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "state": "awaiting_human_review",
                    }
                ),
                encoding="utf-8",
            )

            client = TestClient(
                create_acceptance_review_app(storage, Path(__file__).parents[1])
            )
            root = client.get("/", follow_redirects=False)

            self.assertEqual(root.status_code, 307)
            self.assertEqual(root.headers["location"], "/acceptance-review.html")
            self.assertEqual(
                client.get("/api/acceptance-runs").json(),
                [{"run_id": run_id, "source_job_id": None, "source_scene_id": None}],
            )
            self.assertEqual(client.get("/api/status").status_code, 404)

    def test_launcher_uses_review_only_application_when_auto_rollout_is_locked(self) -> None:
        from scripts.run_gui import _build_gui_application
        from tenminvideomaker.continuation_validation import ContinuationRolloutError

        storage = Mock()
        review_only_app = object()
        with (
            patch(
                "scripts.run_gui.build_supervisor",
                side_effect=ContinuationRolloutError("approval required"),
            ),
            patch(
                "scripts.run_gui.create_acceptance_review_app",
                return_value=review_only_app,
            ) as create_review_app,
            patch("scripts.run_gui.create_gui_app") as create_gui,
        ):
            app, controller, auto_lock_error = _build_gui_application(
                storage,
                Path(__file__).parents[1],
                lan_password=None,
                require_human_review=False,
            )

        self.assertIs(app, review_only_app)
        self.assertIsNone(controller)
        self.assertEqual(auto_lock_error, "approval required")
        create_review_app.assert_called_once_with(
            storage,
            Path(__file__).parents[1],
            lan_password=None,
        )
        create_gui.assert_not_called()

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

    def test_launcher_accepts_all_current_continuation_node_contracts(self) -> None:
        from scripts.run_gui import _ensure_current_node_contract

        current = Mock()
        current.object_info.side_effect = lambda node_type: {
            node_type: {
                "input": {
                    "required": {
                        "revision": ["INT"],
                        "artifact_kind": [
                            [
                                "stage1_handoff",
                                "stage2_video",
                                "stage2_audio",
                            ],
                            {"default": "stage1_handoff"},
                        ],
                        "expected_temporal_tokens": ["INT"],
                        "conditioning": ["CONDITIONING"],
                        "model": ["MODEL"],
                        "scope": ["STRING"],
                        "ckpt_name": [["10Eros_v1.4_fp8mixed_learned.safetensors"]],
                    }
                }
            }
        }

        with patch("scripts.run_gui.restart_comfyui") as restart:
            _ensure_current_node_contract(current)

        self.assertEqual(
            [call.args[0] for call in current.object_info.call_args_list],
            [
                "10MinVideoMaker_SaveSceneFrame",
                "10MinVideoMaker_SaveChunkLatent",
                "10MinVideoMaker_LoadChunkLatent",
                "10MinVideoMaker_IsolateConditioning",
                "10MinVideoMaker_IsolateModel",
                "10MinVideoMaker_FreshCheckpoint",
            ],
        )
        current.queue_counts.assert_not_called()
        restart.assert_not_called()

    def test_launcher_does_not_restart_stale_contract_while_queue_is_busy(self) -> None:
        from scripts.run_gui import _ensure_current_node_contract
        from tenminvideomaker.ownership import OwnershipError

        stale = Mock()
        stale.object_info.side_effect = lambda node_type: {
            node_type: {
                "input": {
                    "required": {
                        "revision": ["INT"],
                        "artifact_kind": [["stage1_handoff"]],
                    }
                }
            }
        }
        stale.queue_counts.return_value = (1, 0)

        with self.assertRaisesRegex(OwnershipError, "queue is busy"):
            _ensure_current_node_contract(stale)

    def test_launcher_restarts_stale_contract_once_when_queue_is_empty(self) -> None:
        from scripts.run_gui import _ensure_current_node_contract

        stale = Mock()
        stale.queue_counts.return_value = (0, 0)
        current_after_restart = False

        def object_info(node_type):
            if node_type == "10MinVideoMaker_SaveSceneFrame":
                required = {"revision": ["INT"]}
            elif node_type == "10MinVideoMaker_SaveChunkLatent":
                required = {
                    "artifact_kind": (
                        [["stage1_handoff"]] if not current_after_restart else [
                            [
                                "stage1_handoff",
                                "stage2_video",
                                "stage2_audio",
                            ]
                        ]
                        )
                }
            elif node_type == "10MinVideoMaker_IsolateConditioning":
                required = {
                    "conditioning": ["CONDITIONING"],
                    "scope": ["STRING"],
                }
            elif node_type == "10MinVideoMaker_IsolateModel":
                required = {
                    "model": ["MODEL"],
                    "scope": ["STRING"],
                }
            elif node_type == "10MinVideoMaker_FreshCheckpoint":
                required = {
                    "ckpt_name": [["10Eros_v1.4_fp8mixed_learned.safetensors"]],
                    "scope": ["STRING"],
                }
            else:
                required = {
                    "artifact_kind": [
                        [
                            "stage1_handoff",
                            "stage2_video",
                            "stage2_audio",
                        ]
                    ],
                    "expected_temporal_tokens": ["INT"],
                }
            return {node_type: {"input": {"required": required}}}

        stale.object_info.side_effect = object_info

        def restart_comfyui():
            nonlocal current_after_restart
            current_after_restart = True
            return True

        with patch(
            "scripts.run_gui.restart_comfyui",
            side_effect=restart_comfyui,
        ) as restart:
            _ensure_current_node_contract(stale)

        restart.assert_called_once_with()

    def test_launcher_rejects_contract_that_remains_stale_after_restart(self) -> None:
        from scripts.run_gui import _ensure_current_node_contract
        from tenminvideomaker.ownership import OwnershipError

        stale = Mock()
        stale.object_info.side_effect = lambda node_type: {
            node_type: {"input": {"required": {"revision": ["INT"]}}}
        }
        stale.queue_counts.return_value = (0, 0)

        with (
            patch("scripts.run_gui.restart_comfyui", return_value=True) as restart,
            self.assertRaisesRegex(OwnershipError, "current continuation artifact"),
        ):
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
            source["ltxv_character_lora"] = {
                "name": "Elsa LTX",
                "download_url": "https://civitai.com/api/download/models/7654321",
                "weight": 0.55,
            }
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
                    "continuation_mode": "explicit",
                },
                approve_job=store.approve_job,
                cancel_current_project=cancel_current_project,
                chunk_progress_document=lambda _job, _scene, revision: {
                    "revision": revision
                },
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
            job_detail = client.get(f"/api/jobs/{job.job_id}").json()
            self.assertEqual(job_detail["display_name"], "Elsa · 07/24/2026")
            scene = client.get(f"/api/jobs/{job.job_id}/scenes/1").json()
            self.assertIn("first_pass", scene["parameters"]["i2v"])
            self.assertEqual(
                scene["parameters"]["character"]["ltx_character_lora"]["name"],
                "Elsa LTX",
            )
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
                "character": {
                    **scene["parameters"]["character"],
                    "ltx_character_lora": {
                        **scene["parameters"]["character"]["ltx_character_lora"],
                        "weight": 0.65,
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
            self.assertEqual(
                revisions[2]["parameters"]["character"]["ltx_character_lora"]["weight"],
                0.65,
            )
            latest_job_detail = client.get(f"/api/jobs/{job.job_id}").json()
            self.assertEqual(
                latest_job_detail["scenes"][0]["chunk_progress"]["revision"],
                2,
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

            cancelled = client.post("/api/pipeline/cancel-current")
            self.assertEqual(cancelled.status_code, 200)
            self.assertEqual(cancelled.json()["job_id"], job.job_id)
            self.assertTrue(cancelled.json()["cancelled"])
            self.assertEqual(store.snapshot().state, PipelineState.IDLE)

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

    def test_acceptance_review_routes_expose_only_validated_review_assets(self) -> None:
        from tenminvideomaker.gui_app import create_gui_app

        run_id = "continuation-acceptance-20260731-065935"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = StorageLayout(root / "storage")
            storage.ensure()
            run_root = storage.root / "acceptance" / run_id
            run_root.mkdir(parents=True)
            raw_paths = {
                "common_base": storage.chunk_video_path(run_id, 1, 1, 0, 1),
                "single_frame": storage.chunk_video_path(run_id, 1, 2, 0, 1),
                "decoded_17_frame": storage.chunk_video_path(run_id, 1, 3, 0, 1),
                "latent_overlap": storage.chunk_video_path(run_id, 1, 1, 1, 1),
            }
            for raw_path in raw_paths.values():
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_bytes(b"raw")
            (run_root / "review").mkdir()
            (run_root / "review" / "base.mp4").write_bytes(b"review")
            (run_root / "review" / "assembled-single_frame.mp4").write_bytes(
                b"assembled-review"
            )
            still = run_root / "metrics" / "single_frame" / "base_0119.png"
            still.parent.mkdir(parents=True)
            still.write_bytes(b"png")
            for filename in ("base_0120.png", "case_0000.png", "case_0001.png"):
                (still.parent / filename).write_bytes(b"png")
            for case_name, filenames in {
                "decoded_17_frame": (
                    "base_0096.png", "base_0111.png", "base_0112.png",
                    "case_0000.png", "case_0016.png", "case_0017.png",
                ),
                "latent_overlap": (
                    "base_0096.png", "base_0119.png", "base_0120.png",
                    "case_0000.png", "case_0024.png", "case_0025.png",
                ),
            }.items():
                metric_root = run_root / "metrics" / case_name
                metric_root.mkdir(parents=True)
                for filename in filenames:
                    (metric_root / filename).write_bytes(b"png")
            (run_root / "run.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "state": "awaiting_human_review",
                        "source_job_id": "20260730-0217",
                        "source_scene_id": 1,
                        "cases": {
                            name: {"stage2": {"raw_video_path": str(path)}}
                            for name, path in raw_paths.items()
                        },
                    }
                ),
                encoding="utf-8",
            )
            controller = SimpleNamespace(
                store=PipelineStateStore(storage.database_path),
            )
            client = TestClient(
                create_gui_app(controller, storage, Path(__file__).parents[1])
            )

            listed = client.get("/api/acceptance-runs")
            self.assertEqual(listed.status_code, 200)
            self.assertEqual(listed.json()[0]["run_id"], run_id)
            document = client.get(f"/api/acceptance-runs/{run_id}")
            self.assertEqual(document.status_code, 200)
            self.assertEqual(
                document.json()["cases"]["latent_overlap"]["boundary"]["right"],
                [24, 25],
            )
            self.assertNotIn(str(storage.root), document.text)
            self.assertEqual(
                client.get(f"/api/acceptance-runs/{run_id}/media/base").headers[
                    "content-type"
                ].split(";", 1)[0],
                "video/mp4",
            )
            self.assertEqual(
                client.get(
                    f"/api/acceptance-runs/{run_id}/assembled/single_frame"
                ).headers["content-type"].split(";", 1)[0],
                "video/mp4",
            )
            self.assertEqual(
                client.get(
                    f"/api/acceptance-runs/{run_id}/stills/single_frame/base_0119.png"
                ).headers["content-type"].split(";", 1)[0],
                "image/png",
            )
            self.assertEqual(
                client.get("/api/acceptance-runs/not-a-run").status_code,
                404,
            )

            secured_client = TestClient(
                create_gui_app(
                    controller,
                    storage,
                    Path(__file__).parents[1],
                    lan_password="mobile-password",
                )
            )
            self.assertEqual(secured_client.get("/api/acceptance-runs").status_code, 401)
            credentials = base64.b64encode(b"10min:mobile-password").decode("ascii")
            self.assertEqual(
                secured_client.get(
                    "/api/acceptance-runs",
                    headers={"Authorization": f"Basic {credentials}"},
                ).status_code,
                200,
            )


if __name__ == "__main__":
    unittest.main()
