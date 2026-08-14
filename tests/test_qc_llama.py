from __future__ import annotations

from pathlib import Path
import hashlib
import subprocess
import tempfile
import unittest
from dataclasses import replace

from tenminvideomaker.qc_config import QualityControlSettings
from tenminvideomaker.qc_llama import (
    GpuIdentity,
    LlamaCppLifecycleError,
    LlamaCppProcess,
    build_llama_command,
    discover_matching_gpu,
)
from tenminvideomaker.storage import StorageLayout


class Completed:
    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = ""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


class FakeProcess:
    def __init__(self, *, wait_times_out: bool = False):
        self.pid = 4321
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.wait_times_out = wait_times_out

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.wait_times_out and not self.killed:
            raise subprocess.TimeoutExpired("llama-server", timeout)
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode


def settings(root: Path) -> QualityControlSettings:
    executable = root / "llama-server.exe"
    vendor = root / "vendor"
    model = root / "model.gguf"
    projector = root / "mmproj.gguf"
    executable.write_bytes(b"exe")
    vendor.mkdir()
    model.write_bytes(b"model")
    projector.write_bytes(b"projector")
    return QualityControlSettings(
        quality_control_enabled=True,
        llama_executable=executable,
        llama_vendor_root=vendor,
        model_path=model,
        projector_path=projector,
        expected_executable_sha256=hashlib.sha256(b"exe").hexdigest(),
        expected_model_sha256=hashlib.sha256(b"model").hexdigest(),
        expected_projector_sha256=hashlib.sha256(b"projector").hexdigest(),
        expected_gpu_uuid="GPU-12345678-abcd-ef01-2345-6789abcdef01",
        expected_gpu_name="NVIDIA GeForce RTX 4080 SUPER",
    )


class QcLlamaTests(unittest.TestCase):
    def test_gpu_is_matched_by_uuid_and_name_not_ordinal(self) -> None:
        configured = settings(Path(tempfile.mkdtemp()))

        gpu = discover_matching_gpu(
            configured,
            lambda *args, **kwargs: Completed(
                "GPU-other, NVIDIA GeForce RTX 5070 Ti\n"
                "GPU-12345678-abcd-ef01-2345-6789abcdef01, NVIDIA GeForce RTX 4080 SUPER\n"
            ),
        )

        self.assertEqual(gpu.uuid, configured.expected_gpu_uuid)
        self.assertEqual(gpu.name, configured.expected_gpu_name)

    def test_mismatched_gpu_fails_closed(self) -> None:
        configured = settings(Path(tempfile.mkdtemp()))
        with self.assertRaises(LlamaCppLifecycleError):
            discover_matching_gpu(
                configured,
                lambda *args, **kwargs: Completed(
                    "GPU-other, NVIDIA GeForce RTX 4080 SUPER\n"
                ),
            )

    def test_unexpected_backend_version_fails_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured = settings(root)
            launched = []

            def run_command(command, **kwargs):
                return Completed(
                    "llama.cpp version 9.0.0"
                    if "--version" in command
                    else f"{configured.expected_gpu_uuid}, {configured.expected_gpu_name}"
                )

            manager = LlamaCppProcess(
                configured,
                StorageLayout(root / "storage"),
                run_command=run_command,
                popen_factory=lambda *args, **kwargs: launched.append(args),
                port_open_probe=lambda: False,
            )
            with self.assertRaises(LlamaCppLifecycleError):
                manager.start()
            self.assertEqual(launched, [])

    def test_lmstudio_package_version_accepts_exact_hashed_upstream_self_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured = settings(root)
            package = root / "llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.28.2"
            package.mkdir()
            executable = package / "llama-server.exe"
            executable.write_bytes(configured.llama_executable.read_bytes())
            configured = replace(
                configured,
                llama_executable=executable,
                expected_executable_sha256=hashlib.sha256(
                    executable.read_bytes()
                ).hexdigest(),
            )
            manager = LlamaCppProcess(
                configured,
                StorageLayout(root / "storage"),
                run_command=lambda command, **kwargs: Completed(
                    "version: 1 (fe2adf0)"
                    if "--version" in command
                    else f"{configured.expected_gpu_uuid}, {configured.expected_gpu_name}"
                ),
                popen_factory=lambda command, **kwargs: FakeProcess(),
                health_probe=lambda: True,
                port_open_probe=lambda: False,
                telemetry_probe=lambda path, gpu: True,
            )

            identity = manager.start()
            manager.close()

            self.assertIn("package 2.28.2", identity.backend_version)
            self.assertIn("fe2adf0", identity.backend_version)

    def test_command_uses_validated_assets_and_fresh_single_slot_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configured = settings(Path(directory))
            slot_cache = Path(directory) / "slot-cache"
            command = build_llama_command(configured, slot_save_path=slot_cache)
            joined = " ".join(command)

            self.assertIn(str(configured.model_path), command)
            self.assertIn(str(configured.projector_path), command)
            self.assertIn("--alias production-vlm-qc", joined)
            self.assertIn("--host 127.0.0.1", joined)
            self.assertIn("--parallel 1", joined)
            self.assertIn("--slots", command)
            self.assertEqual(
                command[command.index("--slot-save-path") + 1], str(slot_cache)
            )
            self.assertIn("--image-min-tokens 1024", joined)
            self.assertIn("--no-cache-prompt", command)
            self.assertIn("--no-cache-prompt", command)
            self.assertEqual(command[command.index("--slot-prompt-similarity") + 1], "0")
            self.assertEqual(command[command.index("--split-mode") + 1], "none")
            self.assertIn("--no-webui", command)
            self.assertNotIn("--main-gpu", command)

    def test_owned_process_has_hidden_launch_identity_evidence_and_bounded_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured = settings(root)
            layout = StorageLayout(root / "storage")
            fake = FakeProcess(wait_times_out=True)
            launch: dict[str, object] = {}

            def run_command(command, **kwargs):
                if "--version" in command:
                    return Completed("llama.cpp version 2.28.2")
                return Completed(
                    f"{configured.expected_gpu_uuid}, {configured.expected_gpu_name}\n"
                )

            def popen_factory(command, **kwargs):
                launch["command"] = command
                launch["kwargs"] = kwargs
                return fake

            manager = LlamaCppProcess(
                configured,
                layout,
                run_command=run_command,
                popen_factory=popen_factory,
                health_probe=lambda: True,
                port_open_probe=lambda: False,
                telemetry_probe=lambda path, gpu: True,
                sleep=lambda seconds: None,
            )

            identity = manager.start()
            self.assertEqual(identity.owned_pid, 4321)
            self.assertEqual(identity.gpu_uuid, configured.expected_gpu_uuid)
            self.assertTrue(Path(identity.stdout_log_path).is_relative_to(layout.logs_root))
            self.assertEqual(
                manager.environment["CUDA_VISIBLE_DEVICES"], configured.expected_gpu_uuid
            )
            self.assertTrue(launch["kwargs"].get("creationflags", 0))

            manager.close()
            self.assertTrue(fake.terminated)
            self.assertTrue(fake.killed)

    def test_child_visible_device_telemetry_must_confirm_the_bound_device(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured = settings(root)
            manager = LlamaCppProcess(
                configured,
                StorageLayout(root / "storage"),
                run_command=lambda command, **kwargs: Completed(
                    "llama.cpp version 2.28.2"
                    if "--version" in command
                    else (
                        "CUDA0: NVIDIA GeForce RTX 5070 Ti"
                        if "--list-devices" in command
                        else f"{configured.expected_gpu_uuid}, {configured.expected_gpu_name}"
                    )
                ),
                popen_factory=lambda command, **kwargs: FakeProcess(),
                health_probe=lambda: True,
                port_open_probe=lambda: False,
                telemetry_probe=lambda path, gpu: False,
                sleep=lambda seconds: None,
            )

            with self.assertRaises(LlamaCppLifecycleError):
                manager.start()

    def test_readiness_timeout_terminates_only_the_owned_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured = replace(settings(root), startup_timeout_seconds=1)
            fake = FakeProcess()
            clock = [0.0]

            def run_command(command, **kwargs):
                return Completed(
                    "llama.cpp version 2.28.2"
                    if "--version" in command
                    else f"{configured.expected_gpu_uuid}, {configured.expected_gpu_name}"
                )

            manager = LlamaCppProcess(
                configured,
                StorageLayout(root / "storage"),
                run_command=run_command,
                popen_factory=lambda command, **kwargs: fake,
                health_probe=lambda: False,
                port_open_probe=lambda: False,
                telemetry_probe=lambda path, gpu: True,
                sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
                monotonic=lambda: clock[0],
            )

            with self.assertRaises(LlamaCppLifecycleError):
                manager.start()
            self.assertTrue(fake.terminated)
            self.assertFalse(fake.killed)

    def test_graceful_close_waits_for_port_to_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured = settings(root)
            fake = FakeProcess()
            port_states = iter((False, True, True, False))

            def run_command(command, **kwargs):
                return Completed(
                    "llama.cpp version 2.28.2"
                    if "--version" in command
                    else f"{configured.expected_gpu_uuid}, {configured.expected_gpu_name}"
                )

            manager = LlamaCppProcess(
                configured,
                StorageLayout(root / "storage"),
                run_command=run_command,
                popen_factory=lambda command, **kwargs: fake,
                health_probe=lambda: True,
                port_open_probe=lambda: next(port_states),
                telemetry_probe=lambda path, gpu: True,
                sleep=lambda seconds: None,
            )
            manager.start()

            manager.close()

            self.assertTrue(fake.terminated)
            self.assertFalse(fake.killed)


if __name__ == "__main__":
    unittest.main()
