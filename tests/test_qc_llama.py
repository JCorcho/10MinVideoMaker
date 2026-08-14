from __future__ import annotations

from pathlib import Path
import hashlib
import os
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace

from tenminvideomaker.qc_config import QualityControlSettings
from tenminvideomaker.qc_llama import (
    GpuIdentity,
    LlamaCppLifecycleError,
    LlamaCppProcess,
    WindowsKillOnCloseJob,
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


class FakeOwnershipBoundary:
    def __init__(self):
        self.assigned = []
        self.closed = False

    def assign(self, process):
        self.assigned.append(process)

    def close(self):
        self.closed = True


def successful_run(configured):
    def run(command, **kwargs):
        if "--version" in command:
            return Completed("llama.cpp version 2.28.2")
        if "--list-devices" in command:
            return Completed(f"CUDA0: {configured.expected_gpu_name}")
        return Completed(
            f"{configured.expected_gpu_uuid}, {configured.expected_gpu_name}"
        )

    return run


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
    def test_owned_child_is_assigned_to_lifetime_boundary_until_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured = settings(root)
            fake = FakeProcess()
            boundary = FakeOwnershipBoundary()
            manager = LlamaCppProcess(
                configured,
                StorageLayout(root / "storage"),
                run_command=successful_run(configured),
                popen_factory=lambda command, **kwargs: fake,
                health_probe=lambda: True,
                port_open_probe=lambda: False,
                telemetry_probe=lambda path, gpu: True,
                ownership_factory=lambda: boundary,
            )

            manager.start()

            self.assertEqual(boundary.assigned, [fake])
            self.assertFalse(boundary.closed)
            manager.close()
            self.assertTrue(boundary.closed)

    @unittest.skipUnless(os.name == "nt", "Windows Job Object integration test")
    def test_abrupt_parent_exit_kills_owned_child_but_not_unrelated_same_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child_pid_path = root / "owned-child.pid"
            parent_code = "\n".join(
                (
                    "import subprocess, sys, time",
                    "from pathlib import Path",
                    "from tenminvideomaker.qc_llama import WindowsKillOnCloseJob",
                    "owner = WindowsKillOnCloseJob()",
                    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])",
                    "owner.assign(child)",
                    "Path(sys.argv[1]).write_text(str(child.pid), encoding='ascii')",
                    "time.sleep(120)",
                )
            )
            unrelated = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(120)"],
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            parent = subprocess.Popen(
                [sys.executable, "-c", parent_code, str(child_pid_path)],
                cwd=str(Path(__file__).parents[1]),
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            try:
                deadline = time.monotonic() + 10
                while not child_pid_path.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(child_pid_path.exists())
                owned_pid = int(child_pid_path.read_text(encoding="ascii"))

                parent.kill()
                parent.wait(timeout=10)

                deadline = time.monotonic() + 10
                while WindowsKillOnCloseJob.pid_is_alive(owned_pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertFalse(WindowsKillOnCloseJob.pid_is_alive(owned_pid))
                self.assertIsNone(unrelated.poll())
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.wait(timeout=10)
                if unrelated.poll() is None:
                    unrelated.terminate()
                    unrelated.wait(timeout=10)
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

    def test_unknown_process_on_owned_port_fails_closed_without_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured = settings(root)
            launched = []
            manager = LlamaCppProcess(
                configured,
                StorageLayout(root / "storage"),
                popen_factory=lambda *args, **kwargs: launched.append(args),
                port_open_probe=lambda: True,
            )

            with self.assertRaisesRegex(LlamaCppLifecycleError, "unknown process"):
                manager.start()

            self.assertEqual(launched, [])

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
            command = build_llama_command(configured)
            joined = " ".join(command)

            self.assertIn(str(configured.model_path), command)
            self.assertIn(str(configured.projector_path), command)
            self.assertIn("--alias production-vlm-qc", joined)
            self.assertIn("--host 127.0.0.1", joined)
            self.assertIn("--parallel 1", joined)
            self.assertIn("--slots", command)
            self.assertNotIn("--slot-save-path", command)
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

    def test_post_launch_telemetry_runs_once_after_readiness_and_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured = settings(root)
            fake = FakeProcess()
            probes = []
            manager = LlamaCppProcess(
                configured,
                StorageLayout(root / "storage"),
                run_command=successful_run(configured),
                popen_factory=lambda command, **kwargs: fake,
                health_probe=lambda: True,
                port_open_probe=lambda: False,
                telemetry_probe=lambda path, gpu: probes.append((path, gpu)) or True,
            )

            identity = manager.start()
            manager.close()

            self.assertEqual(len(probes), 1)
            self.assertIn("post_launch_log_match=expected_gpu_name", identity.device_telemetry)

    def test_default_post_launch_probe_reads_llama_stderr_device_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stdout_path = root / "qc-llama-launch.stdout.log"
            stderr_path = root / "qc-llama-launch.stderr.log"
            stdout_path.write_text("server ready", encoding="utf-8")
            stderr_path.write_text(
                "load_tensors: offloading to NVIDIA GeForce RTX 4080 SUPER",
                encoding="utf-8",
            )

            self.assertTrue(
                LlamaCppProcess._default_telemetry_probe(
                    stdout_path,
                    GpuIdentity("GPU-expected", "NVIDIA GeForce RTX 4080 SUPER"),
                )
            )

    def test_post_launch_telemetry_mismatch_fails_before_backend_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured = settings(root)
            fake = FakeProcess()
            probes = []
            manager = LlamaCppProcess(
                configured,
                StorageLayout(root / "storage"),
                run_command=successful_run(configured),
                popen_factory=lambda command, **kwargs: fake,
                health_probe=lambda: True,
                port_open_probe=lambda: False,
                telemetry_probe=lambda path, gpu: probes.append((path, gpu)) or False,
            )

            with self.assertRaisesRegex(LlamaCppLifecycleError, "post-launch"):
                manager.start()

            self.assertEqual(len(probes), 1)
            self.assertTrue(fake.terminated)

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

    def test_failed_port_close_keeps_exact_owned_process_for_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured = replace(settings(root), shutdown_timeout_seconds=1)
            fake = FakeProcess()
            clock = [0.0]
            port_open = [False]

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
                port_open_probe=lambda: port_open[0],
                telemetry_probe=lambda path, gpu: True,
                sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
                monotonic=lambda: clock[0],
            )
            manager.start()
            port_open[0] = True

            with self.assertRaises(LlamaCppLifecycleError):
                manager.close()
            self.assertIs(manager._process, fake)

            port_open[0] = False
            manager.close()
            self.assertIsNone(manager._process)


if __name__ == "__main__":
    unittest.main()
