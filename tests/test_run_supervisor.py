from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.run_supervisor import (
    PROJECT_ROOT,
    _qc_owned_runtime_layout,
    _runtime_root_overlaps_disallowed_root,
    _require_auto_continuation_approval,
    build_supervisor,
)
from tenminvideomaker.storage import StorageLayout


class _Renderer:
    @staticmethod
    def implementation_sha256() -> str:
        return "1" * 64

    @staticmethod
    def runtime_contract_sha256() -> str:
        return "2" * 64


class RunSupervisorTests(unittest.TestCase):
    def test_explicit_mode_does_not_require_auto_approval(self) -> None:
        supervisor = SimpleNamespace(
            settings=SimpleNamespace(continuation_mode="explicit"),
            continuation_renderer=_Renderer(),
        )
        with patch(
            "scripts.run_supervisor.require_auto_rollout_approval"
        ) as require:
            _require_auto_continuation_approval(
                supervisor,
                StorageLayout(Path(r"D:\test")),
            )
        require.assert_not_called()

    def test_auto_mode_binds_approval_to_current_runtime_identity(self) -> None:
        storage = StorageLayout(Path(r"D:\test"))
        supervisor = SimpleNamespace(
            settings=SimpleNamespace(continuation_mode="auto"),
            continuation_renderer=_Renderer(),
        )
        with patch(
            "scripts.run_supervisor.require_auto_rollout_approval"
        ) as require:
            _require_auto_continuation_approval(supervisor, storage)
        require.assert_called_once_with(
            storage,
            implementation_sha256="1" * 64,
            node_contracts_sha256="2" * 64,
        )

    def test_qc_owned_runtime_layout_is_project_runtime_qc_owned(self) -> None:
        settings = SimpleNamespace(
            llama_vendor_root=Path(r"D:\vendor"),
            llama_executable=Path(r"D:\vendor") / "llama-server.exe",
            model_path=Path(r"D:\model.gguf"),
            projector_path=Path(r"D:\projector.gguf"),
        )
        layout = _qc_owned_runtime_layout(
            settings,
            StorageLayout(Path(r"D:\storage")),
        )
        self.assertEqual(layout.root, PROJECT_ROOT / "runtime" / "qc-owned")

    def test_qc_owned_runtime_layout_fails_for_overlapping_unsafe_root(self) -> None:
        settings = SimpleNamespace(
            llama_vendor_root=Path(r"D:\vendor"),
            llama_executable=Path(r"D:\vendor") / "llama-server.exe",
            model_path=Path(r"D:\model.gguf"),
            projector_path=Path(r"D:\projector.gguf"),
        )
        with self.assertRaises(RuntimeError):
            _qc_owned_runtime_layout(
                settings,
                StorageLayout(PROJECT_ROOT / "runtime"),
            )
        with self.assertRaises(RuntimeError):
            _qc_owned_runtime_layout(
                settings,
                StorageLayout(PROJECT_ROOT / "runtime" / "qc-owned" / "assets"),
            )

    def test_qc_owned_runtime_overlap_helper_checks_both_directions(self) -> None:
        candidate = PROJECT_ROOT / "runtime" / "qc-owned"
        disallowed_parent = PROJECT_ROOT / "runtime"
        disallowed_child = candidate / "assets"
        self.assertTrue(_runtime_root_overlaps_disallowed_root(candidate, disallowed_parent))
        self.assertTrue(_runtime_root_overlaps_disallowed_root(candidate, disallowed_child))

    def test_build_supervisor_routes_llama_backend_to_dedicated_runtime_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = StorageLayout(Path(directory) / "storage")
            qc_settings = SimpleNamespace(
                llama_vendor_root=storage.root / "vendor",
                llama_executable=(storage.root / "vendor") / "llama-server.exe",
                model_path=storage.root / "model.gguf",
                projector_path=storage.root / "projector.gguf",
            )
            captured: dict[str, object] = {}

            class FakeLlamaProcess:
                def __init__(self, qc_settings_arg, layout):
                    captured["llama_layout"] = layout

            class FakeBackend:
                def __init__(self, qc_settings_arg, backend):
                    captured["backend"] = backend

            class FakeQcController:
                def __init__(
                    self,
                    store,
                    layout,
                    settings,
                    backend_factory,
                    prompt_root,
                    ffmpeg_command,
                    ffprobe_command,
                ):
                    captured["controller_layout"] = layout
                    captured["backend_factory"] = backend_factory
                    self.store = store
                    self.layout = layout
                    self.settings = settings
                    self.backend = backend_factory()

            class FakeSupervisor:
                def __init__(self, **kwargs):
                    self.settings = kwargs["settings"]
                    self.continuation_renderer = None

            with patch(
                "scripts.run_supervisor.StorageLayout.configured",
                return_value=storage,
            ), patch(
                "scripts.run_supervisor.migrate_legacy_storage",
            ), patch(
                "scripts.run_supervisor.load_project_environment",
                return_value={},
            ), patch(
                "scripts.run_supervisor.SupervisorSettings.from_environment",
                return_value=SimpleNamespace(continuation_mode="explicit"),
            ), patch(
                "scripts.run_supervisor.QualityControlSettings.from_environment",
                return_value=qc_settings,
            ), patch(
                "scripts.run_supervisor.Phase1QcController",
                FakeQcController,
            ), patch(
                "scripts.run_supervisor.LlamaCppHttpBackend",
                FakeBackend,
            ), patch(
                "scripts.run_supervisor.LlamaCppProcess",
                FakeLlamaProcess,
            ), patch(
                "scripts.run_supervisor.PipelineSupervisor",
                FakeSupervisor,
            ), patch(
                "scripts.run_supervisor.ComfyHttpClient",
                lambda url: SimpleNamespace(url=url),
            ), patch(
                "scripts.run_supervisor.ComfyLoraAssetClient",
                lambda comfy: SimpleNamespace(comfy=comfy),
            ), patch(
                "scripts.run_supervisor.FfmpegAssembler",
                lambda *args, **kwargs: SimpleNamespace(),
            ), patch(
                "scripts.run_supervisor.SceneChunkAssembler",
                lambda *args, **kwargs: SimpleNamespace(),
            ), patch(
                "scripts.run_supervisor.GmailSettings.from_environment",
                lambda *args, **kwargs: SimpleNamespace(),
            ), patch(
                "scripts.run_supervisor.GmailClient",
                lambda settings: SimpleNamespace(),
            ), patch(
                "scripts.run_supervisor.DiscordDeliverySettings.from_environment",
                lambda *args, **kwargs: SimpleNamespace(),
            ), patch(
                "scripts.run_supervisor._require_auto_continuation_approval",
            ):
                build_supervisor(allow_restart=False)

            self.assertIs(captured["controller_layout"], storage)
            self.assertEqual(
                captured["llama_layout"].root,  # type: ignore[union-attr]
                PROJECT_ROOT / "runtime" / "qc-owned",
            )
            self.assertFalse(
                captured["llama_layout"].root.is_relative_to(storage.root)  # type: ignore[union-attr]
            )


if __name__ == "__main__":
    unittest.main()
