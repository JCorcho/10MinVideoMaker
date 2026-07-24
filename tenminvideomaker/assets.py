"""Safe, deterministic LoRA discovery and download services."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Callable, Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

from .contracts import LoraSpec

_FILENAME_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class AssetDownloadError(RuntimeError):
    """Raised when a remote asset cannot be downloaded after all retries."""


@dataclass(frozen=True)
class LocalLoraRequirement:
    filename: str
    weight: float


@dataclass(frozen=True)
class AssetResolution:
    name: str
    path: Path | None
    downloaded: bool
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.path is not None and self.error is None


def predictable_lora_filename(name: str) -> str:
    """Return a portable safe filename without trusting a remote URL path."""
    sanitized = _FILENAME_UNSAFE.sub("_", name.strip()).strip("._")
    if not sanitized:
        raise ValueError("LoRA name does not contain a usable filename.")
    return sanitized if sanitized.casefold().endswith(".safetensors") else f"{sanitized}.safetensors"


class LoraAssetManager:
    """Resolves each asset independently so one failed scene does not block another."""

    def __init__(
        self,
        lora_directories: Iterable[str | Path],
        manifest_path: str | Path,
        *,
        retries: int = 3,
        retry_delay_seconds: float = 1.0,
        downloader: Callable[[str, Path], None] | None = None,
    ):
        self.lora_directories = tuple(Path(directory) for directory in lora_directories)
        if not self.lora_directories:
            raise ValueError("At least one ComfyUI LoRA directory is required.")
        self.manifest_path = Path(manifest_path)
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self._downloader = downloader or self._download_with_urllib

    def resolve_or_download(self, lora: LoraSpec) -> AssetResolution:
        filename = predictable_lora_filename(lora.name)
        existing = self._find_existing(lora.name, filename)
        if existing:
            return AssetResolution(lora.name, existing, downloaded=False)
        destination = self.lora_directories[0] / filename
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._download_with_retries(lora.download_url, destination)
            self._record_manifest(lora.name, destination)
            return AssetResolution(lora.name, destination, downloaded=True)
        except (AssetDownloadError, OSError) as error:
            return AssetResolution(lora.name, None, downloaded=False, error=str(error))

    def resolve_many(self, loras: Iterable[LoraSpec]) -> tuple[AssetResolution, ...]:
        return tuple(self.resolve_or_download(lora) for lora in loras)

    def require_local(self, requirement: LocalLoraRequirement) -> AssetResolution:
        existing = self._find_existing(requirement.filename, requirement.filename)
        if existing:
            return AssetResolution(requirement.filename, existing, downloaded=False)
        return AssetResolution(
            requirement.filename,
            None,
            downloaded=False,
            error=f"Required local LoRA is missing: {requirement.filename}",
        )

    def _find_existing(self, lora_name: str, expected_filename: str) -> Path | None:
        manifest = self._read_manifest()
        manifest_name = manifest.get(lora_name)
        if manifest_name:
            candidate = Path(manifest_name)
            if self._inside_allowed_directory(candidate) and candidate.is_file():
                return candidate
        for directory in self.lora_directories:
            candidate = directory / expected_filename
            if candidate.is_file():
                return candidate
        return None

    def _inside_allowed_directory(self, candidate: Path) -> bool:
        resolved_candidate = candidate.resolve(strict=False)
        return any(
            resolved_candidate.is_relative_to(directory.resolve(strict=False)) for directory in self.lora_directories
        )

    def _read_manifest(self) -> dict[str, str]:
        if not self.manifest_path.is_file():
            return {}
        try:
            parsed = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) and all(isinstance(key, str) and isinstance(value, str) for key, value in parsed.items()) else {}

    def _record_manifest(self, name: str, path: Path) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = self._read_manifest()
        manifest[name] = str(path.resolve())
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.manifest_path)

    def _download_with_retries(self, url: str, destination: Path) -> None:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                self._downloader(url, destination)
                if not destination.is_file() or destination.stat().st_size == 0:
                    raise AssetDownloadError("Downloader did not produce a non-empty file.")
                return
            except (AssetDownloadError, OSError, URLError) as error:
                last_error = error
                partial = destination.with_suffix(destination.suffix + ".part")
                partial.unlink(missing_ok=True)
                destination.unlink(missing_ok=True)
                if attempt < self.retries:
                    time.sleep(self.retry_delay_seconds * attempt)
        raise AssetDownloadError(f"Download failed after {self.retries} attempts: {last_error}")

    @staticmethod
    def _download_with_urllib(url: str, destination: Path) -> None:
        """Download through urllib, which follows standard HTTP redirects by default."""
        temporary = destination.with_suffix(destination.suffix + ".part")
        request = Request(url, headers={"User-Agent": "10MinVideoMaker/0.1"})
        with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        os.replace(temporary, destination)
