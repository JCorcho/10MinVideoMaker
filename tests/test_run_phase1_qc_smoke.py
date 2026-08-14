from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import scripts.run_phase1_qc_smoke as smoke
from scripts.run_phase1_qc_smoke import (
    _numeric_compute_processes,
    ensure_nonproduction_smoke_paths,
    run_context_isolation_probe,
)
from tenminvideomaker.qc_backend import (
    RepairPlannerResponse,
    VisionJudgeEvaluation,
    load_repair_planner_prompt,
)
from tenminvideomaker.qc_contracts import parse_judge_response
from tenminvideomaker.qc_video import SampledFrame, SampledVideo, VideoMetadata


class Phase1QcSmokeHarnessTests(unittest.TestCase):
    def test_windows_display_rows_do_not_count_as_compute_owners(self) -> None:
        uuid = "GPU-stable"
        output = (
            f"100, explorer.exe, {uuid}, [N/A]\n"
            f"200, llama-server.exe, {uuid}, 9123\n"
            "300, other.exe, GPU-other, 100\n"
        )
        self.assertEqual(
            _numeric_compute_processes(output, uuid),
            [f"200, llama-server.exe, {uuid}, 9123"],
        )

    def test_context_probe_actually_sends_sentinel_in_a_but_not_b(self) -> None:
        requests = []

        class Backend:
            def plan_repair(self, request):
                requests.append(request)
                return RepairPlannerResponse(
                    raw_text=json.dumps(
                        {
                            "schema_version": 1,
                            "source": {
                                "candidate_id": request.source_identity["candidate_id"],
                                "candidate_sha256": request.source_identity["candidate_sha256"],
                                "evaluation_id": request.source_identity["evaluation_id"],
                                "repair_input_sha256": request.repair_input_sha256,
                                "source_revision": request.source_identity["source_revision"],
                                "source_document_sha256": request.source_identity["source_document_sha256"],
                            },
                            "patch": {"i2v": {"prompt": "bounded smoke prompt"}},
                            "summary": "smoke",
                        }
                    )
                )

        sentinel = "SENTINEL-ONLY-IN-REQUEST-A-7c1f"
        result = run_context_isolation_probe(
            Backend(),
            load_repair_planner_prompt(
                Path(__file__).parents[1] / "prompts" / "production_i2v_repair_v1.txt"
            ),
            sentinel,
        )

        self.assertEqual(len(requests), 2)
        self.assertIn(sentinel, json.dumps(requests[0].fixed_scene_facts))
        self.assertNotIn(sentinel, json.dumps(requests[1].fixed_scene_facts))
        self.assertTrue(result["request_a_payload_contains_sentinel"])
        self.assertFalse(result["request_b_payload_contains_sentinel"])
        self.assertFalse(result["response_b_contains_sentinel"])

    def test_smoke_paths_reject_production_storage_and_require_worktree_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "fixture.mp4"
            media.write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError, "evidence root"):
                ensure_nonproduction_smoke_paths(Path(directory) / "evidence", media)
            with self.assertRaisesRegex(ValueError, "production storage"):
                ensure_nonproduction_smoke_paths(
                    Path(__file__).parents[1] / "test-evidence" / "smoke",
                    Path(r"D:\LTX_Supervisor_Storage\jobs\subscriber.mp4"),
                )

    def test_preflight_without_execute_creates_no_evidence_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "fixture.mp4"
            media.write_bytes(b"fixture")
            evidence = root / "allowed" / "preflight-only"
            args = self._arguments(root, evidence, media)
            with (
                patch.object(smoke, "SMOKE_EVIDENCE_ROOT", root / "allowed"),
                patch.object(smoke, "hardware_preflight", return_value=self._preflight()),
                patch.object(
                    smoke.QualityControlSettings,
                    "validate_for_start",
                    return_value=None,
                ),
            ):
                result = smoke.main(args)

            self.assertEqual(result, 0)
            self.assertFalse(evidence.exists())

    def test_execute_main_traverses_owned_judge_sentinels_cleanup_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "fixture.mp4"
            media.write_bytes(b"fixture")
            evidence = root / "allowed" / "execute"
            events: list[str] = []
            sampled = SampledVideo(
                VideoMetadata(24.0, 4, 1.0),
                2.0,
                tuple(
                    SampledFrame(index, index / 2, root / f"{index}.jpg", b"jpeg")
                    for index in range(4)
                ),
            )
            identity = SimpleNamespace(
                backend_version="mock-llama",
                gpu_uuid="GPU-test",
                gpu_name="RTX test",
                device_telemetry=("mock-device",),
                owned_pid=4321,
                launch_id="launch-test",
                started_at="2026-08-13T00:00:00Z",
                effective_args=("--mock",),
                stdout_log_path="stdout.log",
                stderr_log_path="stderr.log",
            )

            class Process:
                _process = None

                def __init__(self, *_args):
                    events.append("owned-process-created")

            class Backend:
                def __init__(self, *_args):
                    self.plan_calls = 0

                def start(self):
                    events.append("backend-start")
                    return identity

                def evaluate(self, _request):
                    events.append("judge")
                    return VisionJudgeEvaluation(
                        parse_judge_response(
                            '{"decision":"PASS","confidence":0.99,'
                            '"summary":"clean","errors":[]}'
                        )
                    )

                def plan_repair(self, _request):
                    self.plan_calls += 1
                    events.append(f"sentinel-{self.plan_calls}")
                    return RepairPlannerResponse(raw_text=f"response-{self.plan_calls}")

                def close(self):
                    events.append("backend-close")

            args = self._arguments(root, evidence, media) + ["--execute"]
            with (
                patch.object(smoke, "SMOKE_EVIDENCE_ROOT", root / "allowed"),
                patch.object(smoke, "hardware_preflight", return_value=self._preflight()),
                patch.object(
                    smoke.QualityControlSettings,
                    "validate_for_start",
                    return_value=None,
                ),
                patch.object(smoke, "LlamaCppProcess", Process),
                patch.object(smoke, "LlamaCppHttpBackend", Backend),
                patch.object(smoke, "sample_video_frames", return_value=sampled),
                patch.object(smoke, "_gpu_line", return_value="GPU-test, RTX test, 100, 0, P8"),
                patch.object(
                    smoke,
                    "_gpu_snapshot",
                    return_value={
                        "line": "GPU-test, RTX test, 100, 0, P8",
                        "uuid": "GPU-test",
                        "name": "RTX test",
                        "memory_used_mib": 100,
                        "utilization_percent": 0,
                        "pstate": "P8",
                    },
                ),
                patch.object(smoke, "_pid_exists", return_value=False),
            ):
                result = smoke.main(args)

            self.assertEqual(result, 0)
            self.assertEqual(
                events,
                [
                    "owned-process-created",
                    "backend-start",
                    "judge",
                    "sentinel-1",
                    "sentinel-2",
                    "backend-close",
                ],
            )
            document = json.loads((evidence / "smoke-result.json").read_text())
            self.assertTrue(document["success"])
            self.assertTrue(document["owned_process_exited"])
            self.assertTrue(document["vram_returned"])
            self.assertEqual(document["qc"]["requests_sent"], 1)
            self.assertFalse(
                document["context_isolation"]["response_b_contains_sentinel"]
            )

    @staticmethod
    def _preflight() -> dict[str, object]:
        return {
            "safe": True,
            "gpu": {
                "line": "GPU-test, RTX test, 100, 0, P8",
                "uuid": "GPU-test",
                "name": "RTX test",
                "memory_used_mib": 100,
                "utilization_percent": 0,
                "pstate": "P8",
            },
            "matching_compute_processes": [],
            "loopback_port_open": False,
            "reasons": [],
        }

    @staticmethod
    def _arguments(root: Path, evidence: Path, media: Path) -> list[str]:
        return [
            "--evidence-root", str(evidence),
            "--media", str(media),
            "--sentinel", "SENTINEL-ONLY-IN-REQUEST-A-7c1f",
            "--executable", str(root / "llama-server.exe"),
            "--vendor-root", str(root / "vendor"),
            "--model", str(root / "model.gguf"),
            "--projector", str(root / "projector.gguf"),
            "--executable-sha256", "1" * 64,
            "--model-sha256", "2" * 64,
            "--projector-sha256", "3" * 64,
            "--gpu-uuid", "GPU-test",
            "--gpu-name", "RTX test",
        ]


if __name__ == "__main__":
    unittest.main()
