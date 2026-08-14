from __future__ import annotations

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
        self.assertEqual(settings.minimum_strong_windows, 2)

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


if __name__ == "__main__":
    unittest.main()
