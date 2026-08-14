from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import unittest

from tenminvideomaker.qc_backend import (
    HeadlessVideoEvaluator,
    LlamaCppHttpBackend,
    RepairPlannerRequest,
    VisionJudgeEvaluation,
    VisionJudgeRequest,
    build_repair_planner_payload,
    build_vision_judge_payload,
    load_repair_planner_prompt,
    load_production_rubric,
)
from tenminvideomaker.qc_config import QualityControlSettings
from tenminvideomaker.qc_contracts import QcDecision, parse_judge_response
from tenminvideomaker.qc_video import chronological_windows
from test_qc_video import sampled


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "production_ltx_video_qc_v1.txt"
REPAIR_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "production_i2v_repair_v1.txt"


def raw(decision: str, *, strong: bool = False) -> str:
    errors = []
    if strong:
        errors = [
            {
                "category": "topology",
                "severity": 3,
                "confidence": 0.9,
                "start_time_seconds": 0.5,
                "end_time_seconds": 1.0,
                "description": "visible fusion",
                "evidence": "two boundaries visibly merge",
            }
        ]
    return json.dumps(
        {
            "decision": decision,
            "confidence": 0.9,
            "summary": "result",
            "errors": errors,
        }
    )


class FakeBackend:
    def __init__(self, responses: list[str]):
        self.responses = deque(responses)
        self.requests: list[VisionJudgeRequest] = []

    def evaluate(self, request: VisionJudgeRequest) -> VisionJudgeEvaluation:
        self.requests.append(request)
        response = parse_judge_response(self.responses.popleft())
        return VisionJudgeEvaluation(response=response, input_tokens=100, output_tokens=20)


class QcBackendTests(unittest.TestCase):
    def test_production_prompt_preserves_validated_lab_bytes(self) -> None:
        rubric = load_production_rubric(PROMPT_PATH)

        self.assertEqual(rubric.version, "production_ltx_video_qc_v1")
        self.assertEqual(
            hashlib.sha256(rubric.text.encode("utf-8")).hexdigest(),
            "5e91bc1d45809d712a8848f915c4f2c797d117a37bddc49eeef0ab80d6534dd0",
        )

    def test_judge_payload_is_blind_fresh_and_tool_free(self) -> None:
        rubric = load_production_rubric(PROMPT_PATH)
        request = VisionJudgeRequest.from_window(
            chronological_windows(sampled(4))[0], rubric=rubric
        )

        payload = build_vision_judge_payload(request)
        serialized = json.dumps(payload)

        self.assertEqual(payload["temperature"], 0.0)
        self.assertFalse(payload["cache_prompt"])
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)
        self.assertNotIn("ORIGINAL", serialized)
        self.assertNotIn("A1", serialized)
        self.assertNotIn("B1", serialized)
        self.assertIn("0.000, 0.500, 1.000, 1.500", serialized)
        self.assertEqual(
            [item["type"] for item in payload["messages"][1]["content"]],
            ["text", "image_url", "image_url", "image_url", "image_url"],
        )

    def test_repair_planner_payload_is_separate_text_only_fresh_and_tool_free(self) -> None:
        prompt = load_repair_planner_prompt(REPAIR_PROMPT_PATH)
        request = RepairPlannerRequest(
            job_identity={"job_id": "job-1"},
            scene_identity={"scene_id": 2},
            source_identity={
                "candidate_id": "candidate-a1",
                "candidate_sha256": "a" * 64,
                "evaluation_id": "evaluation-a1",
            },
            current_i2v_prompt="A woman turns toward camera.",
            negative_prompt="no deformation",
            fixed_scene_facts={"character": "Ava", "required_action": "turns"},
            generation_config={"seed": "42", "width": 768, "height": 1344},
            normalized_qc={"decision": "FAIL", "summary": "hand fusion"},
            suspect_windows=({"start": 1.0, "end": 2.0},),
            previous_repairs=({"summary": "none"},),
            mutable_fields=("i2v.prompt",),
            locked_fields=("i2v.negative", "t2i", "character", "duration"),
            repair_input_sha256="b" * 64,
            prompt=prompt,
        )

        payload = build_repair_planner_payload(request)
        serialized = json.dumps(payload)

        self.assertEqual(payload["temperature"], 0.0)
        self.assertFalse(payload["cache_prompt"])
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)
        self.assertNotIn("image_url", serialized)
        self.assertNotIn("shell", serialized.casefold())
        self.assertNotIn("sql", serialized.casefold())
        self.assertIn("normalized_qc", serialized)
        self.assertIn("mutable_fields", serialized)
        self.assertIn("locked_fields", serialized)
        self.assertNotEqual(
            payload["messages"][0]["content"],
            build_vision_judge_payload(
                VisionJudgeRequest.from_window(
                    chronological_windows(sampled(4))[0],
                    rubric=load_production_rubric(PROMPT_PATH),
                )
            )["messages"][0]["content"],
        )

    def test_two_sequential_requests_are_self_contained_and_erase_slot_kv(self) -> None:
        seen = []

        class Response:
            status = 200

            def __init__(self, body: bytes = b"{}"):
                self.body = body

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return self.body

        def fake_urlopen(request, timeout):
            seen.append((request.full_url, request.data, timeout))
            if "/slots/" in request.full_url:
                return Response()
            body = json.dumps({
                "choices": [{"message": {"content": raw("PASS")}}],
                "usage": {},
            }).encode()
            return Response(body)

        backend = LlamaCppHttpBackend(
            QualityControlSettings(), object(), urlopen_factory=fake_urlopen
        )
        rubric = load_production_rubric(PROMPT_PATH)
        first = VisionJudgeRequest.from_window(
            chronological_windows(sampled(4))[0], rubric=rubric
        )
        second = VisionJudgeRequest.from_window(
            chronological_windows(sampled(5))[1], rubric=rubric
        )
        backend.evaluate(first)
        backend.evaluate(second)

        chats = [json.loads(data) for url, data, _ in seen if "/chat/completions" in url]
        resets = [url for url, _, _ in seen if "/slots/0?action=erase" in url]
        self.assertEqual(len(chats), 2)
        self.assertEqual(len(resets), 2)
        sentinel = "UNIQUE_SENTINEL_FROM_A"
        chats[0]["messages"][1]["content"][0]["text"] += sentinel
        self.assertNotIn(sentinel, json.dumps(chats[1]))
        self.assertEqual(len(chats[1]["messages"]), 2)

    def test_two_strong_normal_windows_end_evaluation_early(self) -> None:
        backend = FakeBackend(
            [raw("FAIL", strong=True), raw("FAIL", strong=True), raw("PASS")]
        )
        evaluator = HeadlessVideoEvaluator(backend, load_production_rubric(PROMPT_PATH))

        result = evaluator.evaluate_sampled(sampled(12))

        self.assertEqual(result.normalized.decision, QcDecision.FAIL)
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual(result.frame_accounting["processed_window_count"], 2)
        self.assertTrue(result.frame_accounting["early_exit_applied"])

    def test_lone_suspect_gets_one_fresh_shifted_confirmation(self) -> None:
        backend = FakeBackend(
            [raw("PASS"), raw("FAIL", strong=True), raw("PASS"), raw("PASS")]
        )
        evaluator = HeadlessVideoEvaluator(backend, load_production_rubric(PROMPT_PATH))

        result = evaluator.evaluate_sampled(sampled(12))

        self.assertEqual(result.normalized.decision, QcDecision.PASS)
        self.assertEqual(len(backend.requests), 4)
        self.assertTrue(backend.requests[-1].independent_confirmation)
        self.assertEqual(backend.requests[-1].window.confirmation_of_window, 2)
        self.assertEqual(result.frame_accounting["confirmation_frame_exposures"], 4)

    def test_malformed_response_cannot_become_pass(self) -> None:
        backend = FakeBackend(["I refuse", raw("PASS")])
        evaluator = HeadlessVideoEvaluator(backend, load_production_rubric(PROMPT_PATH))

        result = evaluator.evaluate_sampled(sampled(8))

        self.assertEqual(result.normalized.decision, QcDecision.UNCERTAIN)
        self.assertIn("I refuse", result.raw_result)


if __name__ == "__main__":
    unittest.main()
