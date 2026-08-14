from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_phase1_qc_smoke import (
    _numeric_compute_processes,
    ensure_nonproduction_smoke_paths,
    run_context_isolation_probe,
)
from tenminvideomaker.qc_backend import RepairPlannerResponse, load_repair_planner_prompt


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


if __name__ == "__main__":
    unittest.main()
