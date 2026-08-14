from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_phase1_qc_microbenchmark import (
    BlindSample,
    LabeledSample,
    deterministic_order,
    load_manifest,
    run_microbenchmark,
    validate_benchmark_paths,
)
from tenminvideomaker.qc_backend import BackendIdentity, VisionJudgeEvaluation
from tenminvideomaker.qc_config import QualityControlSettings
from tenminvideomaker.qc_contracts import parse_judge_response
from tenminvideomaker.qc_video import SampledFrame, SampledVideo, VideoMetadata


class FakeSettings:
    def __init__(self):
        self.base = QualityControlSettings(quality_control_enabled=True)
        self.validations = 0

    def __getattr__(self, name):
        return getattr(self.base, name)

    def validate_for_start(self):
        self.validations += 1

    def effective_document(self):
        return self.base.effective_document()

    def effective_sha256(self):
        return self.base.effective_sha256()


def _response(decision: str, *, strong: bool = False) -> str:
    errors = []
    if strong:
        errors = [
            {
                "category": "topology",
                "severity": 4,
                "confidence": 0.95,
                "start_time_seconds": 0.0,
                "end_time_seconds": 1.5,
                "description": "visible defect",
                "evidence": "visible across the inspected frames",
            }
        ]
    return json.dumps(
        {
            "decision": decision,
            "confidence": 0.95,
            "summary": "structured result",
            "errors": errors,
        }
    )


class BlindBackend:
    def __init__(self):
        self.starts = 0
        self.closes = 0
        self.requests = []

    def start(self):
        self.starts += 1
        return BackendIdentity(
            evaluator_id="tenminvideomaker.production-vlm-qc",
            evaluator_version="phase1-v1",
            backend_family="llama.cpp",
            backend_version="2.28.2",
            executable_path="C:/blind/llama.exe",
            executable_sha256="1" * 64,
            model_path="C:/blind/model.gguf",
            model_sha256="2" * 64,
            model_id="Qwen3.6 27B Uncensored HauhauCS Balanced",
            quantization="IQ3_M GGUF",
            projector_path="C:/blind/mmproj.gguf",
            projector_sha256="3" * 64,
            projector_precision="FP16",
            gpu_uuid="GPU-test",
            gpu_name="RTX 4080 SUPER",
            effective_args=("--host", "127.0.0.1"),
            effective_config_sha256="4" * 64,
            owned_pid=123,
            stdout_log_path="stdout.log",
            stderr_log_path="stderr.log",
        )

    def evaluate(self, request):
        self.requests.append(request)
        raw = base64.b64decode(request.encoded_images[0].split(",", 1)[1]).decode()
        sample_id, frame_text = raw.rsplit("-frame-", 1)
        frame = int(frame_text)
        if sample_id == "sample-004":
            response = _response("PASS")
        elif sample_id == "sample-001":
            response = _response(
                "FAIL" if frame in {0, 2} else "PASS",
                strong=frame in {0, 2},
            )
        else:
            response = _response("FAIL", strong=True)
        return VisionJudgeEvaluation(parse_judge_response(response))

    def close(self):
        self.closes += 1


class Phase1QcMicrobenchmarkTests(unittest.TestCase):
    def _samples(self, root: Path):
        samples = []
        for index, label in enumerate(("BAD", "BAD", "BAD", "GOOD"), 1):
            path = root / f"clip-{index:03d}.mp4"
            path.write_bytes(f"opaque-media-{index}".encode())
            samples.append(
                LabeledSample(BlindSample(f"sample-{index:03d}", path), label)
            )
        return tuple(samples)

    @staticmethod
    def _sampler(settings):
        preprocessing = settings.effective_document()["sampling"]["preprocessing"]

        def sample(path, **_kwargs):
            sample_id = "sample-" + path.stem.rsplit("-", 1)[1]
            frames = tuple(
                SampledFrame(
                    index,
                    index / 2,
                    path.with_name(f"{sample_id}-{index}.jpg"),
                    f"{sample_id}-frame-{index}".encode(),
                )
                for index in range(8)
            )
            return SampledVideo(
                VideoMetadata(24.0, 8, 4.0),
                2.0,
                frames,
                preprocessing,
            )

        return sample

    def test_mocked_four_video_run_is_blind_and_uses_one_production_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = self._samples(root)
            settings = FakeSettings()
            backend = BlindBackend()

            report = run_microbenchmark(
                samples=samples,
                evidence_root=root / "evidence",
                benchmark_seed=731,
                settings=settings,
                backend_factory=lambda: backend,
                sampler=self._sampler(settings),
                environment={"TENMIN_STORAGE_ROOT": str(root / "production")},
            )

            self.assertTrue(report["score"]["passed"])
            self.assertEqual(report["score"]["bad_caught"], 3)
            self.assertEqual(report["score"]["good_accepted"], 1)
            self.assertEqual(backend.starts, 1)
            self.assertEqual(backend.closes, 1)
            self.assertEqual(settings.validations, 1)
            self.assertTrue(
                any(request.independent_confirmation for request in backend.requests)
            )
            sample_one = [
                request
                for request in backend.requests
                if "sample-001" in base64.b64decode(
                    request.encoded_images[0].split(",", 1)[1]
                ).decode()
            ]
            self.assertEqual(len(sample_one), 3)
            for request in backend.requests:
                serialized = json.dumps(
                    {
                        "rubric": request.rubric.text,
                        "images": request.encoded_images,
                    }
                )
                self.assertNotIn('"label"', serialized)
                self.assertNotIn("GOOD", serialized)
                self.assertNotIn("BAD", serialized)
            persisted = json.loads(
                (root / "evidence" / "benchmark-result.json").read_text()
            )
            self.assertEqual(persisted["benchmark_seed"], 731)
            self.assertEqual(persisted["blind_order"], report["blind_order"])

    def test_manifest_and_order_are_strict_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = self._samples(root)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "samples": [
                            {
                                "sample_id": item.blind.sample_id,
                                "path": str(item.blind.path),
                                "label": item.label,
                            }
                            for item in samples
                        ],
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_manifest(manifest)

            self.assertEqual(
                deterministic_order(loaded, 19), deterministic_order(loaded, 19)
            )
            self.assertNotEqual(
                [item.blind.sample_id for item in deterministic_order(loaded, 19)],
                [item.blind.sample_id for item in deterministic_order(loaded, 20)],
            )

    def test_media_and_evidence_below_production_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            production = root / "production"
            production.mkdir()
            samples = list(self._samples(root))
            forbidden_media = production / "live.mp4"
            forbidden_media.write_bytes(b"live")
            samples[0] = LabeledSample(
                BlindSample(samples[0].blind.sample_id, forbidden_media), "BAD"
            )
            environment = {"TENMIN_STORAGE_ROOT": str(production)}

            with self.assertRaisesRegex(ValueError, "inside production"):
                validate_benchmark_paths(
                    samples, root / "evidence", environment=environment
                )
            with self.assertRaisesRegex(ValueError, "outside production"):
                validate_benchmark_paths(
                    self._samples(root), production / "evidence", environment=environment
                )

    def test_backend_is_cleaned_up_when_one_sample_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = self._samples(root)
            settings = FakeSettings()
            backend = BlindBackend()
            normal_sampler = self._sampler(settings)

            def failing_sampler(path, **kwargs):
                if path.name == "clip-002.mp4":
                    raise RuntimeError("synthetic preprocessing failure")
                return normal_sampler(path, **kwargs)

            report = run_microbenchmark(
                samples=samples,
                evidence_root=root / "evidence",
                benchmark_seed=1,
                settings=settings,
                backend_factory=lambda: backend,
                sampler=failing_sampler,
                environment={"TENMIN_STORAGE_ROOT": str(root / "production")},
            )

            self.assertFalse(report["score"]["passed"])
            self.assertEqual(report["score"]["infrastructure_or_malformed"], 1)
            self.assertEqual(backend.starts, 1)
            self.assertEqual(backend.closes, 1)


if __name__ == "__main__":
    unittest.main()
