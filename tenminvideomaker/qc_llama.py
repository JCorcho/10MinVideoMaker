"""Owned llama.cpp lifecycle for serialized QC epochs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
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
    for gpu in discovered:
        if (
            gpu.uuid.casefold() == str(settings.expected_gpu_uuid).casefold()
            and gpu.name.casefold() == str(settings.expected_gpu_name).casefold()
        ):
            return gpu
    raise LlamaCppLifecycleError(
        "The configured physical QC GPU UUID/name pair was not found; refusing "
        "to substitute a CUDA ordinal or another device."
    )


def build_llama_command(settings: QualityControlSettings) -> list[str]:
    if settings.llama_executable is None or settings.model_path is None or settings.projector_path is None:
        raise LlamaCppLifecycleError("Validated llama.cpp assets are not configured.")
    return [
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
        "--parallel",
        str(settings.parallel_slots),
        "--flash-attn",
        "on",
        "--jinja",
        "--no-webui",
        "--image-min-tokens",
        str(settings.image_min_tokens),
        "--no-cache-prompt",
        "--cache-ram",
        "0",
        "--no-cache-idle-slots",
        "--slot-prompt-similarity",
        "0",
        "--offline",
    ]


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
        self.sleep = sleep
        self.monotonic = monotonic
        self.environment: dict[str, str] = {}
        self._process: Any | None = None
        self._identity: BackendIdentity | None = None
        self._stdout_handle: Any | None = None
        self._stderr_handle: Any | None = None

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
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return gpu.name.casefold() in text.casefold()

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
        if not re.search(
            rf"(?<![0-9.]){re.escape(self.settings.backend_version)}(?![0-9.])",
            version,
        ):
            raise LlamaCppLifecycleError(
                "llama.cpp version does not match the validated QC backend version."
            )
        return version

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
        command = build_llama_command(self.settings)
        self.layout.logs_root.mkdir(parents=True, exist_ok=True)
        launch_id = uuid4().hex
        stdout_path = self.layout.logs_root / f"qc-llama-{launch_id}.stdout.log"
        stderr_path = self.layout.logs_root / f"qc-llama-{launch_id}.stderr.log"
        self._stdout_handle = stdout_path.open("x", encoding="utf-8")
        self._stderr_handle = stderr_path.open("x", encoding="utf-8")
        try:
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
            if not (
                self.telemetry_probe(stdout_path, gpu)
                or self.telemetry_probe(stderr_path, gpu)
            ):
                raise LlamaCppLifecycleError(
                    "llama.cpp startup telemetry did not confirm the configured physical GPU."
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
            )
            return self._identity
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        process = self._process
        self._process = None
        self._identity = None
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
