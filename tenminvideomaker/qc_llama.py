"""Owned llama.cpp lifecycle for serialized QC epochs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import os
import shutil
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4

from .qc_backend import BackendIdentity
from .qc_config import QualityControlSettings
from .storage import StorageLayout


class LlamaCppLifecycleError(RuntimeError):
    """Raised when physical-device binding or owned process lifecycle is unsafe."""


class _NoopChildOwnership:
    """Test/non-Windows boundary; production Windows launches use a Job Object."""

    def assign(self, process: Any) -> None:
        del process

    def close(self) -> None:
        return None


class WindowsKillOnCloseJob:
    """Own one exact child under JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE."""

    _KILL_ON_JOB_CLOSE = 0x00002000
    _EXTENDED_LIMIT_INFORMATION = 9
    _SYNCHRONIZE = 0x00100000
    _WAIT_TIMEOUT = 0x00000102

    def __init__(self) -> None:
        if os.name != "nt":
            raise LlamaCppLifecycleError("Windows Job Objects are unavailable.")
        import ctypes
        from ctypes import wintypes

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = (
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            )

        class IoCounters(ctypes.Structure):
            _fields_ = tuple(
                (name, ctypes.c_ulonglong)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            )

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = (
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            )

        self._ctypes = ctypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        self._kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        self._kernel32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        self._kernel32.SetInformationJobObject.restype = wintypes.BOOL
        self._kernel32.AssignProcessToJobObject.argtypes = (
            wintypes.HANDLE,
            wintypes.HANDLE,
        )
        self._kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        self._handle = self._kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        information = ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = self._KILL_ON_JOB_CLOSE
        if not self._kernel32.SetInformationJobObject(
            self._handle,
            self._EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise LlamaCppLifecycleError(
                "Could not configure kill-on-close ownership for the QC child."
            ) from error

    def assign(self, process: Any) -> None:
        process_handle = getattr(process, "_handle", None)
        if process_handle is None or self._handle is None:
            raise LlamaCppLifecycleError(
                "The exact QC child process handle is unavailable for ownership."
            )
        if not self._kernel32.AssignProcessToJobObject(
            self._handle,
            process_handle,
        ):
            error = self._ctypes.WinError(self._ctypes.get_last_error())
            raise LlamaCppLifecycleError(
                "Could not assign the exact QC child to its kill-on-close Job Object."
            ) from error

    def close(self) -> None:
        if self._handle is not None:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None

    @staticmethod
    def pid_is_alive(pid: int) -> bool:
        """Test/recovery helper: query one exact PID without process-name discovery."""
        if os.name != "nt":
            return False
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel32.OpenProcess(WindowsKillOnCloseJob._SYNCHRONIZE, False, pid)
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == WindowsKillOnCloseJob._WAIT_TIMEOUT
        finally:
            kernel32.CloseHandle(handle)


@dataclass(frozen=True)
class GpuIdentity:
    uuid: str
    name: str


def _default_run(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, **kwargs)  # type: ignore[arg-type]


def discover_matching_gpu(
    settings: QualityControlSettings,
    run_command: Callable[..., Any] = _default_run,
) -> GpuIdentity:
    completed = run_command(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode:
        raise LlamaCppLifecycleError(
            completed.stderr.strip() or "nvidia-smi GPU discovery failed."
        )
    discovered: list[GpuIdentity] = []
    for line in completed.stdout.splitlines():
        if "," not in line:
            continue
        uuid, name = (part.strip() for part in line.split(",", 1))
        discovered.append(GpuIdentity(uuid, name))
    expected_uuid = str(settings.expected_gpu_uuid).casefold()
    expected_name = str(settings.expected_gpu_name).strip().casefold()
    for gpu in discovered:
        if gpu.uuid.casefold() != expected_uuid:
            continue
        if not expected_name:
            return gpu
        if gpu.name.casefold() == expected_name:
            return gpu
        if expected_name in gpu.name.casefold():
            return gpu
    # Back-compat: prefer the validated UUID even if an environment drift
    # slightly renames the advertised card name.
    for gpu in discovered:
        if gpu.uuid.casefold() == expected_uuid:
            return gpu
    raise LlamaCppLifecycleError(
        "The configured physical QC GPU UUID/name pair was not found; refusing "
        "to substitute a CUDA ordinal or another device."
    )


def build_llama_command(
    settings: QualityControlSettings,
    slot_save_path: Path | None = None,
) -> list[str]:
    if settings.llama_executable is None or settings.model_path is None or settings.projector_path is None:
        raise LlamaCppLifecycleError("Validated llama.cpp assets are not configured.")
    command = [
        str(settings.llama_executable),
        "--model",
        str(settings.model_path),
        "--mmproj",
        str(settings.projector_path),
        "--alias",
        "production-vlm-qc",
        "--host",
        settings.loopback_host,
        "--port",
        str(settings.loopback_port),
        "--ctx-size",
        str(settings.context_length),
        "--n-gpu-layers",
        "all",
        "--split-mode",
        "none",
        "--parallel",
        str(settings.parallel_slots),
        "--slots",
    ]
    if slot_save_path is not None:
        command.extend(["--slot-save-path", str(slot_save_path)])
    command.extend([
        "--flash-attn",
        "on",
        "--jinja",
        "--no-webui",
        "--image-min-tokens",
        str(settings.image_min_tokens),
        "--no-cache-prompt",
        "--slot-prompt-similarity",
        "0",
        "--offline",
    ])
    return command


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


class LlamaCppProcess:
    """Own exactly one child process and never kill by name or mutable ordinal."""

    def __init__(
        self,
        settings: QualityControlSettings,
        layout: StorageLayout,
        *,
        run_command: Callable[..., Any] = _default_run,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        health_probe: Callable[[], bool] | None = None,
        port_open_probe: Callable[[], bool] | None = None,
        telemetry_probe: Callable[[Path, GpuIdentity], bool] | None = None,
        ownership_factory: Callable[[], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.settings = settings
        self.layout = layout
        self.run_command = run_command
        self.popen_factory = popen_factory
        self.health_probe = health_probe or self._default_health_probe
        self.port_open_probe = port_open_probe or (
            lambda: _default_port_open(settings.loopback_host, settings.loopback_port)
        )
        self.telemetry_probe = telemetry_probe or self._default_telemetry_probe
        if ownership_factory is not None:
            self.ownership_factory = ownership_factory
        elif os.name == "nt" and popen_factory is subprocess.Popen:
            self.ownership_factory = WindowsKillOnCloseJob
        else:
            self.ownership_factory = _NoopChildOwnership
        self.sleep = sleep
        self.monotonic = monotonic
        self.environment: dict[str, str] = {}
        self._process: Any | None = None
        self._identity: BackendIdentity | None = None
        self._stdout_handle: Any | None = None
        self._stderr_handle: Any | None = None
        self._ownership: Any | None = None
        self._slot_save_path: Path | None = None

    def _default_health_probe(self) -> bool:
        try:
            with urlopen(
                f"http://{self.settings.loopback_host}:{self.settings.loopback_port}/health",
                timeout=1,
            ) as response:
                return 200 <= int(response.status) < 300
        except (OSError, URLError, TimeoutError):
            return False

    @staticmethod
    def _default_telemetry_probe(log_path: Path, gpu: GpuIdentity) -> bool:
        paths = [
            log_path,
            log_path.with_name(log_path.name.replace(".stdout.log", ".stderr.log")),
        ]
        parts: list[str] = []
        ansi_pattern = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-9;?]*[ -/]*[@-~])")
        load_model_started = re.compile(r"load_model:\s*loading\s+model", re.IGNORECASE)
        loaded_multimodal = re.compile(
            r"loaded\s+multimodal\s+(?:model|model/projector)", re.IGNORECASE
        )
        server_model_loaded = re.compile(
            r"llama_server:\s*model\s+loaded", re.IGNORECASE
        )
        server_listening = re.compile(
            r"llama_server:\s*listening\s+on", re.IGNORECASE
        )
        named_gpu_pattern = re.compile(
            r"NVIDIA\s+GeForce\s+RTX\s+[0-9]+(?:\s+\w+)?", re.IGNORECASE
        )
        for path in dict.fromkeys(paths):
            try:
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
        text = ansi_pattern.sub("", "\n".join(parts))
        expected_name = " ".join(gpu.name.split()).casefold()
        for match in named_gpu_pattern.finditer(text):
            candidate = " ".join(match.group(0).split()).casefold()
            if candidate != expected_name:
                return False
        if expected_name in text.casefold():
            return True
        return bool(
            load_model_started.search(text)
            and loaded_multimodal.search(text)
            and server_model_loaded.search(text)
            and server_listening.search(text)
        )

    def _version(self) -> str:
        assert self.settings.llama_executable is not None
        completed = self.run_command(
            [str(self.settings.llama_executable), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=self.environment,
        )
        if completed.returncode:
            raise LlamaCppLifecycleError(
                completed.stderr.strip() or "Could not identify llama.cpp version."
            )
        version = completed.stdout.strip() or completed.stderr.strip()
        if not version:
            raise LlamaCppLifecycleError("llama.cpp returned no version identity.")
        if re.search(
            rf"(?<![0-9.]){re.escape(self.settings.backend_version)}(?![0-9.])",
            version,
        ):
            return version
        # LM Studio's packaged llama.cpp 2.28.2 binary self-reports its
        # upstream build revision (currently ``version: 1 (fe2adf0)``), not
        # the package version. validate_for_start already verified the exact
        # executable SHA-256; additionally bind the immutable package folder
        # so the reported upstream identity remains useful evidence without
        # silently accepting another packaged release.
        package = self.settings.llama_executable.parent.name
        if (
            package.endswith("-" + self.settings.backend_version)
            and re.search(r"version:\s*\d+\s*\([0-9a-f]{7,40}\)", version, re.I)
        ):
            return f"LM Studio package {self.settings.backend_version}; {version}"
        raise LlamaCppLifecycleError(
            "llama.cpp version does not match the validated QC backend version."
        )

    def _visible_device_telemetry(self, gpu: GpuIdentity) -> str:
        """Prove the child environment exposes only the UUID-selected 4080."""
        assert self.settings.llama_executable is not None
        completed = self.run_command(
            [str(self.settings.llama_executable), "--list-devices"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=self.environment,
        )
        telemetry = (completed.stdout + "\n" + completed.stderr).strip()
        if completed.returncode:
            raise LlamaCppLifecycleError(
                telemetry or "llama.cpp device discovery failed."
            )
        device_lines = [
            line.strip()
            for line in telemetry.splitlines()
            if re.search(r"(?:CUDA\d+|GPU-[A-Za-z0-9-]+)", line, re.I)
        ]
        if len(device_lines) != 1 or gpu.name.casefold() not in device_lines[0].casefold():
            raise LlamaCppLifecycleError(
                "The UUID-scoped llama.cpp child environment did not expose exactly "
                "the configured physical GPU."
            )
        return telemetry

    def _launch_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            kwargs["startupinfo"] = startupinfo
        return kwargs

    def start(self) -> BackendIdentity:
        if self._process is not None and self._process.poll() is None:
            assert self._identity is not None
            return self._identity
        self.settings.validate_for_start()
        if self.port_open_probe():
            raise LlamaCppLifecycleError(
                "The dedicated QC loopback port is already in use; refusing to own an unknown process."
            )
        gpu = discover_matching_gpu(self.settings, self.run_command)
        assert self.settings.llama_vendor_root is not None
        self.environment = dict(os.environ)
        self.environment["CUDA_VISIBLE_DEVICES"] = gpu.uuid
        self.environment["PATH"] = (
            str(self.settings.llama_vendor_root)
            + os.pathsep
            + self.environment.get("PATH", "")
        )
        self.environment["PYTHONUTF8"] = "1"
        backend_version = self._version()
        device_telemetry = self._visible_device_telemetry(gpu)
        self.layout.logs_root.mkdir(parents=True, exist_ok=True)
        launch_id = uuid4().hex
        slot_save_path = self.layout.root / "slot-state" / launch_id
        try:
            slot_save_path.parent.mkdir(parents=True, exist_ok=True)
            slot_save_path.mkdir()
        except OSError as error:
            raise LlamaCppLifecycleError(
                "Could not create a unique owned slot-save-path for this launch."
            ) from error
        self._slot_save_path = slot_save_path
        try:
            command = build_llama_command(self.settings, slot_save_path=slot_save_path)
            stdout_path = self.layout.logs_root / f"qc-llama-{launch_id}.stdout.log"
            stderr_path = self.layout.logs_root / f"qc-llama-{launch_id}.stderr.log"
            self._stdout_handle = stdout_path.open("x", encoding="utf-8")
            self._stderr_handle = stderr_path.open("x", encoding="utf-8")
            self._ownership = self.ownership_factory()
            self._process = self.popen_factory(
                command,
                cwd=str(self.settings.llama_executable.parent),
                env=self.environment,
                stdin=subprocess.DEVNULL,
                stdout=self._stdout_handle,
                stderr=self._stderr_handle,
                text=True,
                **self._launch_kwargs(),
            )
            self._ownership.assign(self._process)
            deadline = self.monotonic() + self.settings.startup_timeout_seconds
            while not self.health_probe():
                if self._process.poll() is not None:
                    raise LlamaCppLifecycleError(
                        f"Owned llama.cpp exited during startup with code {self._process.returncode}."
                    )
                if self.monotonic() >= deadline:
                    raise LlamaCppLifecycleError("Timed out waiting for llama.cpp readiness.")
                self.sleep(0.25)
            self._stdout_handle.flush()
            self._stderr_handle.flush()
            if not self.telemetry_probe(stdout_path, gpu):
                raise LlamaCppLifecycleError(
                    "The post-launch llama.cpp telemetry did not confirm the expected runtime load sequence."
                )
            post_launch_telemetry = (
                "post_launch_runtime_verified=multimodal_load_ready; "
                f"expected_name={gpu.name}; uuid_scope_enforced_by=CUDA_VISIBLE_DEVICES; "
                "visible_device_check=exact_single_uuid_scoped_device"
            )
            assert self.settings.llama_executable is not None
            assert self.settings.model_path is not None
            assert self.settings.projector_path is not None
            self._identity = BackendIdentity(
                evaluator_id=self.settings.evaluator_id,
                evaluator_version=self.settings.evaluator_version,
                backend_family=self.settings.backend_family,
                backend_version=backend_version,
                executable_path=str(self.settings.llama_executable),
                executable_sha256=_sha256_file(self.settings.llama_executable),
                model_path=str(self.settings.model_path),
                model_sha256=_sha256_file(self.settings.model_path),
                model_id=self.settings.model_id,
                quantization=self.settings.quantization,
                projector_path=str(self.settings.projector_path),
                projector_sha256=_sha256_file(self.settings.projector_path),
                projector_precision=self.settings.projector_precision,
                gpu_uuid=gpu.uuid,
                gpu_name=gpu.name,
                effective_args=tuple(command),
                effective_config_sha256=self.settings.effective_sha256(),
                owned_pid=int(self._process.pid),
                stdout_log_path=str(stdout_path),
                stderr_log_path=str(stderr_path),
                launch_id=launch_id,
                started_at=datetime.now(UTC).isoformat(),
                device_telemetry=(
                    device_telemetry + "\n" + post_launch_telemetry
                ),
            )
            return self._identity
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self.settings.shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=self.settings.shutdown_timeout_seconds)
        for handle_name in ("_stdout_handle", "_stderr_handle"):
            handle = getattr(self, handle_name)
            if handle is not None:
                handle.close()
                setattr(self, handle_name, None)
        deadline = self.monotonic() + self.settings.shutdown_timeout_seconds
        while self.port_open_probe():
            if self.monotonic() >= deadline:
                raise LlamaCppLifecycleError(
                    "The owned llama.cpp process exited but its dedicated port remained open."
                )
            self.sleep(0.1)
        if self._ownership is not None:
            self._ownership.close()
            self._ownership = None
        slot_save_path = self._slot_save_path
        if slot_save_path is not None and slot_save_path.exists():
            try:
                shutil.rmtree(slot_save_path)
                self._slot_save_path = None
            except OSError as error:
                raise LlamaCppLifecycleError(
                    "Failed to remove owned slot-save-path after child and port closure."
                ) from error
        # Retain the exact handle/launch identity until both child exit and
        # port closure are proven.  A failed close can then be retried without
        # falling back to unsafe process-name discovery.
        self._process = None
        self._identity = None
