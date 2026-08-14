from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from tenminvideomaker.qc_config import QualityControlConfigurationError, QualityControlSettings


class QualityControlSettingsTests(unittest.TestCase):
    def test_rollout_defaults_are_disabled_and_human_gated(self) -> None:
        settings = QualityControlSettings.from_environment({})

        self.assertFalse(settings.quality_control_enabled)
        self.assertFalse(settings.auto_advance_pass)
        self.assertEqual(settings.sampling_fps, 2.0)
        self.assertEqual(settings.frames_per_window, 4)
        self.assertEqual(settings.image_min_tokens, 1024)
        self.assertEqual(
            settings.preprocessing_version,
            "vlm-qc-lab-f634ca2-image-v1",
        )
        self.assertEqual(settings.image_max_short_edge, 512)
        self.assertEqual(settings.image_max_pixels, 458752)
        self.assertEqual(settings.image_jpeg_quality, 88)
        self.assertEqual(settings.minimum_strong_windows, 2)
        with self.assertRaisesRegex(
            QualityControlConfigurationError, "kill switch is disabled"
        ):
            settings.validate_for_start()

    def test_boolean_environment_values_are_strict(self) -> None:
        enabled = QualityControlSettings.from_environment(
            {
                "TENMIN_QUALITY_CONTROL_ENABLED": "true",
                "TENMIN_QC_AUTO_ADVANCE_PASS": "1",
            }
        )
        self.assertTrue(enabled.quality_control_enabled)
        self.assertTrue(enabled.auto_advance_pass)

        for invalid in ("yes", "enabled", "2", ""):
            with self.subTest(invalid=invalid):
                with self.assertRaises(QualityControlConfigurationError):
                    QualityControlSettings.from_environment(
                        {"TENMIN_QUALITY_CONTROL_ENABLED": invalid}
                    )

    def test_effective_document_is_versionable_and_has_no_gpu_ordinal(self) -> None:
        settings = QualityControlSettings.from_environment(
            {
                "TENMIN_QC_GPU_UUID": "GPU-12345678-abcd-ef01-2345-6789abcdef01",
                "TENMIN_QC_GPU_NAME": "NVIDIA GeForce RTX 4080 SUPER",
            }
        )

        document = settings.effective_document()
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["gpu"]["expected_uuid"], settings.expected_gpu_uuid)
        self.assertEqual(document["gpu"]["expected_name"], settings.expected_gpu_name)
        self.assertEqual(
            document["sampling"]["preprocessing"],
            {
                "version": "vlm-qc-lab-f634ca2-image-v1",
                "validated_lab_commit": "f634ca2ab7ca95ddd9abde7fe840031eba0696f4",
                "decoder": "ffmpeg_selected_png_rgb24",
                "max_short_edge": 512,
                "max_pixels": 458752,
                "dimension_multiple": 16,
                "resize_interpolation": "opencv_inter_area",
                "jpeg_quality": 88,
                "orientation": "encoded_pixels_no_autorotate",
                "color_pipeline": "rgb24_to_opencv_bgr_to_jpeg",
            },
        )
        self.assertNotIn("ordinal", str(document).casefold())
        self.assertEqual(len(settings.effective_sha256()), 64)

    def test_runtime_validation_requires_stable_gpu_identity_and_exact_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "llama-server.exe"
            vendor = root / "vendor"
            model = root / "model.gguf"
            projector = root / "mmproj.gguf"
            executable.write_bytes(b"exe")
            vendor.mkdir()
            model.write_bytes(b"model")
            projector.write_bytes(b"projector")
            settings = QualityControlSettings.from_environment(
                {
                    "TENMIN_QUALITY_CONTROL_ENABLED": "true",
                    "TENMIN_QC_LLAMA_EXECUTABLE": str(executable),
                    "TENMIN_QC_LLAMA_VENDOR_ROOT": str(vendor),
                    "TENMIN_QC_MODEL_PATH": str(model),
                    "TENMIN_QC_PROJECTOR_PATH": str(projector),
                    "TENMIN_QC_LLAMA_SHA256": hashlib.sha256(b"exe").hexdigest(),
                    "TENMIN_QC_MODEL_SHA256": hashlib.sha256(b"model").hexdigest(),
                    "TENMIN_QC_PROJECTOR_SHA256": hashlib.sha256(b"projector").hexdigest(),
                    "TENMIN_QC_GPU_UUID": "GPU-12345678-abcd-ef01-2345-6789abcdef01",
                    "TENMIN_QC_GPU_NAME": "NVIDIA GeForce RTX 4080 SUPER",
                }
            )

            settings.validate_for_start()

            missing_uuid = QualityControlSettings.from_environment(
                {
                    "TENMIN_QUALITY_CONTROL_ENABLED": "true",
                    "TENMIN_QC_LLAMA_EXECUTABLE": str(executable),
                    "TENMIN_QC_LLAMA_VENDOR_ROOT": str(vendor),
                    "TENMIN_QC_MODEL_PATH": str(model),
                    "TENMIN_QC_PROJECTOR_PATH": str(projector),
                    "TENMIN_QC_GPU_NAME": "NVIDIA GeForce RTX 4080 SUPER",
                }
            )
            with self.assertRaises(QualityControlConfigurationError):
                missing_uuid.validate_for_start()

            changed_model = root / "changed.gguf"
            changed_model.write_bytes(b"different model")
            with self.assertRaises(QualityControlConfigurationError):
                QualityControlSettings.from_environment(
                    {
                        "TENMIN_QC_LLAMA_EXECUTABLE": str(executable),
                        "TENMIN_QC_LLAMA_VENDOR_ROOT": str(vendor),
                        "TENMIN_QC_MODEL_PATH": str(changed_model),
                        "TENMIN_QC_PROJECTOR_PATH": str(projector),
                        "TENMIN_QC_LLAMA_SHA256": hashlib.sha256(b"exe").hexdigest(),
                        "TENMIN_QC_MODEL_SHA256": hashlib.sha256(b"model").hexdigest(),
                        "TENMIN_QC_PROJECTOR_SHA256": hashlib.sha256(b"projector").hexdigest(),
                        "TENMIN_QC_GPU_UUID": "GPU-12345678-abcd-ef01-2345-6789abcdef01",
                        "TENMIN_QC_GPU_NAME": "NVIDIA GeForce RTX 4080 SUPER",
                    }
                ).validate_for_start()

            with self.assertRaises(QualityControlConfigurationError):
                replace(settings, sampling_fps=1.0).validate_for_start()
            with self.assertRaises(QualityControlConfigurationError):
                replace(settings, context_length=8192).validate_for_start()
            with self.assertRaises(QualityControlConfigurationError):
                replace(settings, image_max_pixels=262144).validate_for_start()


if __name__ == "__main__":
    unittest.main()
