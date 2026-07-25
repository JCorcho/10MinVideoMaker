"""Safe, deterministic LoRA discovery and download services."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

from .constants import I2V_DYNAMIC_BASE_MODEL
from .contracts import LoraSpec, civitai_version_id, lora_identity

_FILENAME_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_CIVITAI_HOSTS = frozenset({"civitai.com", "www.civitai.com"})


class AssetDownloadError(RuntimeError):
    """Raised when a remote asset cannot be downloaded safely."""


class AssetAuthenticationRequired(AssetDownloadError):
    """Raised when Civitai requires an account token for file transfer."""


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
    local_filename: str | None = None
    base_model: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.path is not None and self.error is None


@dataclass(frozen=True)
class CivitaiLoraMetadata:
    version_id: int
    model_id: int
    filename: str
    download_url: str
    sha256: str | None
    size_bytes: int | None
    base_model: str | None = None


def predictable_lora_filename(name: str) -> str:
    """Return a portable safe filename without trusting a remote URL path."""
    sanitized = _FILENAME_UNSAFE.sub("_", name.strip()).strip("._")
    if not sanitized:
        raise ValueError("LoRA name does not contain a usable filename.")
    return sanitized if sanitized.casefold().endswith(".safetensors") else f"{sanitized}.safetensors"


def lora_to_mapping(lora: LoraSpec) -> dict[str, Any]:
    return {
        "name": lora.name,
        "download_url": lora.download_url,
        "weight": lora.weight,
        "model_id": lora.model_id,
        "version_id": lora.version_id,
    }


def asset_resolution_from_mapping(value: Mapping[str, Any]) -> AssetResolution:
    path_value = value.get("path")
    return AssetResolution(
        name=str(value.get("name") or ""),
        path=Path(path_value) if isinstance(path_value, str) and path_value else None,
        downloaded=bool(value.get("downloaded")),
        error=str(value["error"]) if value.get("error") else None,
        local_filename=(
            str(value["local_filename"]) if value.get("local_filename") else None
        ),
        base_model=str(value["base_model"]) if value.get("base_model") else None,
    )


class ComfyLoraAssetClient:
    """Resolve LoRAs inside the live ComfyUI process over its loopback API."""

    def __init__(self, comfy_client: Any):
        self.comfy_client = comfy_client

    def resolve_or_download(
        self,
        lora: LoraSpec,
        *,
        expected_base_model: str | None = None,
    ) -> AssetResolution:
        response = self.comfy_client.resolve_lora_asset(
            {
                "kind": "dynamic",
                "lora": lora_to_mapping(lora),
                "expected_base_model": expected_base_model,
            }
        )
        return asset_resolution_from_mapping(response)

    def require_local(self, requirement: LocalLoraRequirement) -> AssetResolution:
        response = self.comfy_client.resolve_lora_asset(
            {
                "kind": "required",
                "filename": requirement.filename,
                "weight": requirement.weight,
            }
        )
        return asset_resolution_from_mapping(response)


class LoraAssetManager:
    """Resolve assets against the active ComfyUI LoRA roots."""

    def __init__(
        self,
        lora_directories: Iterable[str | Path],
        manifest_path: str | Path,
        *,
        visible_lora_names: Iterable[str] = (),
        civitai_token: str = "",
        retries: int = 3,
        retry_delay_seconds: float = 1.0,
        downloader: Callable[[str, Path], None] | None = None,
        metadata_fetcher: Callable[[LoraSpec], CivitaiLoraMetadata | None] | None = None,
    ):
        self.lora_directories = tuple(Path(directory) for directory in lora_directories)
        if not self.lora_directories:
            raise ValueError("At least one ComfyUI LoRA directory is required.")
        self.manifest_path = Path(manifest_path)
        self.visible_lora_names = tuple(
            name for name in visible_lora_names if isinstance(name, str) and name
        )
        self.civitai_token = civitai_token.strip()
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self._downloader = downloader or self._download_with_urllib
        self._metadata_fetcher = metadata_fetcher or self._fetch_civitai_metadata

    def resolve_or_download(
        self,
        lora: LoraSpec,
        *,
        expected_base_model: str | None = None,
    ) -> AssetResolution:
        filename = predictable_lora_filename(lora.name)
        try:
            metadata = None
            if expected_base_model is not None:
                if expected_base_model != I2V_DYNAMIC_BASE_MODEL:
                    raise AssetDownloadError(
                        f"Unsupported required LoRA base model: {expected_base_model}."
                    )
                metadata = self._metadata_fetcher(lora)
                if metadata is None or not metadata.base_model:
                    raise AssetDownloadError(
                        f"{lora.name} cannot be verified as an {expected_base_model} LoRA; "
                        "dynamic I2V LoRAs must have Civitai version metadata."
                    )
                if not _base_models_match(metadata.base_model, expected_base_model):
                    raise AssetDownloadError(
                        f"{lora.name} is a {metadata.base_model} LoRA, not "
                        f"{expected_base_model}; it was blocked from the LTX video model."
                    )
            else:
                existing = self._find_existing(lora, (filename,))
                if existing:
                    path, local_filename = existing
                    return AssetResolution(
                        lora.name,
                        path,
                        downloaded=False,
                        local_filename=local_filename,
                    )

            if metadata is None:
                metadata = self._metadata_fetcher(lora)
            alternatives = (metadata.filename,) if metadata else ()
            existing = self._find_existing(lora, (filename, *alternatives))
            if existing:
                path, local_filename = existing
                self._record_manifest(lora, path)
                return AssetResolution(
                    lora.name,
                    path,
                    downloaded=False,
                    local_filename=local_filename,
                    base_model=metadata.base_model if metadata else None,
                )

            destination = self.lora_directories[0] / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._check_disk_space(destination, metadata.size_bytes if metadata else None)
            self._download_with_retries(
                metadata.download_url if metadata else lora.download_url,
                destination,
                expected_sha256=metadata.sha256 if metadata else None,
            )
            self._record_manifest(lora, destination)
            return AssetResolution(
                lora.name,
                destination,
                downloaded=True,
                local_filename=self._local_filename(destination),
                base_model=metadata.base_model if metadata else None,
            )
        except (AssetDownloadError, OSError) as error:
            return AssetResolution(
                lora.name,
                None,
                downloaded=False,
                error=str(error),
            )

    def resolve_many(
        self,
        loras: Iterable[LoraSpec],
        *,
        expected_base_model: str | None = None,
    ) -> tuple[AssetResolution, ...]:
        return tuple(
            self.resolve_or_download(
                lora,
                expected_base_model=expected_base_model,
            )
            for lora in loras
        )

    def require_local(self, requirement: LocalLoraRequirement) -> AssetResolution:
        placeholder = LoraSpec(
            requirement.filename,
            "https://local.invalid/not-downloaded",
            requirement.weight,
        )
        existing = self._find_existing(placeholder, (requirement.filename,))
        if existing:
            path, local_filename = existing
            return AssetResolution(
                requirement.filename,
                path,
                downloaded=False,
                local_filename=local_filename,
            )
        return AssetResolution(
            requirement.filename,
            None,
            downloaded=False,
            error=f"Required local LoRA is missing: {requirement.filename}",
        )

    def _find_existing(
        self,
        lora: LoraSpec,
        expected_filenames: Iterable[str],
    ) -> tuple[Path, str] | None:
        manifest = self._read_manifest()
        manifest_name = manifest.get(lora_identity(lora)) or manifest.get(lora.name)
        if manifest_name:
            candidate = Path(manifest_name)
            if self._inside_allowed_directory(candidate) and candidate.is_file():
                return candidate, self._local_filename(candidate)

        expected = tuple(dict.fromkeys(expected_filenames))
        for directory in self.lora_directories:
            for filename in expected:
                candidate = directory / filename
                if self._inside_allowed_directory(candidate) and candidate.is_file():
                    return candidate, self._local_filename(candidate)

        visible_by_relative: dict[str, str] = {}
        visible_by_basename: dict[str, str] = {}
        for name in self.visible_lora_names:
            normalized = name.replace("/", os.sep).replace("\\", os.sep)
            visible_by_relative.setdefault(normalized.casefold(), normalized)
            visible_by_basename.setdefault(Path(normalized).name.casefold(), normalized)
        for filename in expected:
            visible_name = (
                visible_by_relative.get(filename.replace("/", os.sep).casefold())
                or visible_by_basename.get(Path(filename).name.casefold())
            )
            if not visible_name:
                continue
            for directory in self.lora_directories:
                candidate = directory / visible_name
                if self._inside_allowed_directory(candidate) and candidate.is_file():
                    return candidate, visible_name
        return None

    def _inside_allowed_directory(self, candidate: Path) -> bool:
        resolved_candidate = candidate.resolve(strict=False)
        return any(
            resolved_candidate.is_relative_to(directory.resolve(strict=False))
            for directory in self.lora_directories
        )

    def _local_filename(self, path: Path) -> str:
        resolved = path.resolve(strict=False)
        for directory in self.lora_directories:
            root = directory.resolve(strict=False)
            if resolved.is_relative_to(root):
                return str(resolved.relative_to(root))
        return path.name

    def _read_manifest(self) -> dict[str, str]:
        if not self.manifest_path.is_file():
            return {}
        try:
            parsed = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return (
            parsed
            if isinstance(parsed, dict)
            and all(isinstance(key, str) and isinstance(value, str) for key, value in parsed.items())
            else {}
        )

    def _record_manifest(self, lora: LoraSpec, path: Path) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = self._read_manifest()
        resolved = str(path.resolve())
        manifest[lora_identity(lora)] = resolved
        manifest[lora.name] = resolved
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.manifest_path)

    @staticmethod
    def _check_disk_space(destination: Path, required_bytes: int | None) -> None:
        if not required_bytes:
            return
        free_bytes = shutil.disk_usage(destination.parent).free
        reserve = max(64 * 1024 * 1024, required_bytes // 20)
        if free_bytes < required_bytes + reserve:
            raise AssetDownloadError(
                f"Not enough free space for {destination.name}: "
                f"need at least {required_bytes + reserve} bytes, have {free_bytes}."
            )

    def _download_with_retries(
        self,
        url: str,
        destination: Path,
        *,
        expected_sha256: str | None,
    ) -> None:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                self._downloader(url, destination)
                if not destination.is_file() or destination.stat().st_size == 0:
                    raise AssetDownloadError("Downloader did not produce a non-empty file.")
                if expected_sha256:
                    actual = _sha256(destination)
                    if actual.casefold() != expected_sha256.casefold():
                        raise AssetDownloadError(
                            f"SHA-256 mismatch for {destination.name}; refusing the file."
                        )
                return
            except AssetAuthenticationRequired:
                self._remove_partial_download(destination)
                raise
            except (AssetDownloadError, OSError, URLError) as error:
                last_error = error
                self._remove_partial_download(destination)
                if attempt < self.retries:
                    time.sleep(self.retry_delay_seconds * attempt)
        raise AssetDownloadError(
            f"Download failed after {self.retries} attempts: {last_error}"
        )

    @staticmethod
    def _remove_partial_download(destination: Path) -> None:
        destination.with_suffix(destination.suffix + ".part").unlink(missing_ok=True)
        destination.unlink(missing_ok=True)

    def _download_with_urllib(self, url: str, destination: Path) -> None:
        temporary = destination.with_suffix(destination.suffix + ".part")
        request_url = self._authenticated_download_url(url)
        request = Request(
            request_url,
            headers={
                "User-Agent": "10MinVideoMaker/0.2",
                "Accept": "application/octet-stream",
            },
        )
        try:
            with urlopen(request, timeout=120) as response:
                final_url = response.geturl()
                content_type = (response.headers.get_content_type() or "").casefold()
                if _is_civitai_login_url(final_url) or content_type == "text/html":
                    raise AssetAuthenticationRequired(_civitai_authentication_message())
                with temporary.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            os.replace(temporary, destination)
        except HTTPError as error:
            if error.code in {401, 403} or _is_civitai_login_url(error.geturl()):
                raise AssetAuthenticationRequired(
                    _civitai_authentication_message()
                ) from None
            raise AssetDownloadError(
                f"Download request failed with HTTP {error.code}."
            ) from None

    def _authenticated_download_url(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.hostname not in _CIVITAI_HOSTS or not self.civitai_token:
            return url
        query = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.casefold() != "token"
        ]
        query.append(("token", self.civitai_token))
        return urlunparse(parsed._replace(query=urlencode(query)))

    @staticmethod
    def _fetch_civitai_metadata(lora: LoraSpec) -> CivitaiLoraMetadata | None:
        version_id = lora.version_id or civitai_version_id(lora.download_url)
        if version_id is None:
            return None
        metadata_url = f"https://civitai.com/api/v1/model-versions/{version_id}"
        request = Request(
            metadata_url,
            headers={
                "User-Agent": "10MinVideoMaker/0.2",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                document = json.load(response)
        except HTTPError as error:
            raise AssetDownloadError(
                f"Civitai metadata request for version {version_id} failed with HTTP {error.code}."
            ) from None
        except (OSError, URLError, json.JSONDecodeError) as error:
            raise AssetDownloadError(
                f"Civitai metadata request for version {version_id} failed: {error}"
            ) from error

        if not isinstance(document, Mapping):
            raise AssetDownloadError(
                f"Civitai metadata for version {version_id} was not an object."
            )
        model = document.get("model")
        if not isinstance(model, Mapping) or str(model.get("type", "")).casefold() != "lora":
            raise AssetDownloadError(f"Civitai version {version_id} is not a LoRA.")
        if model.get("mode") in {"Archived", "TakenDown"}:
            raise AssetDownloadError(
                f"Civitai version {version_id} is {model.get('mode')}."
            )
        model_id = document.get("modelId")
        if isinstance(model_id, bool) or not isinstance(model_id, int):
            raise AssetDownloadError(
                f"Civitai metadata for version {version_id} has no model id."
            )
        if lora.model_id is not None and lora.model_id != model_id:
            raise AssetDownloadError(
                f"Civitai version {version_id} does not belong to model {lora.model_id}."
            )

        files = document.get("files")
        if not isinstance(files, list):
            raise AssetDownloadError(f"Civitai version {version_id} has no files.")
        candidates = [
            item
            for item in files
            if isinstance(item, Mapping)
            and str(item.get("type", "")).casefold() == "model"
            and str((item.get("metadata") or {}).get("format", "")).casefold()
            == "safetensor"
            and str(item.get("virusScanResult", "")).casefold() == "success"
        ]
        if not candidates:
            raise AssetDownloadError(
                f"Civitai version {version_id} has no virus-scanned SafeTensor model file."
            )
        selected = next(
            (item for item in candidates if item.get("primary") is True),
            candidates[0],
        )
        filename = selected.get("name")
        download_url = selected.get("downloadUrl") or document.get("downloadUrl")
        if not isinstance(filename, str) or not filename.casefold().endswith(".safetensors"):
            raise AssetDownloadError(
                f"Civitai version {version_id} has an invalid model filename."
            )
        if not isinstance(download_url, str) or urlparse(download_url).scheme != "https":
            raise AssetDownloadError(
                f"Civitai version {version_id} has an invalid download URL."
            )
        hashes = selected.get("hashes")
        sha256 = hashes.get("SHA256") if isinstance(hashes, Mapping) else None
        size_kb = selected.get("sizeKB")
        size_bytes = (
            int(float(size_kb) * 1024)
            if isinstance(size_kb, (int, float)) and not isinstance(size_kb, bool)
            else None
        )
        return CivitaiLoraMetadata(
            version_id=version_id,
            model_id=model_id,
            filename=filename,
            download_url=download_url,
            sha256=sha256 if isinstance(sha256, str) and sha256 else None,
            size_bytes=size_bytes,
            base_model=(
                str(document["baseModel"]).strip()
                if isinstance(document.get("baseModel"), str)
                and str(document["baseModel"]).strip()
                else None
            ),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _base_models_match(actual: str, expected: str) -> bool:
    """Compare the small set of known Civitai labels for the same LTX 2.3 family."""
    aliases = {
        "ltxv23": I2V_DYNAMIC_BASE_MODEL,
        "ltxvideo23": I2V_DYNAMIC_BASE_MODEL,
        "ltx23": I2V_DYNAMIC_BASE_MODEL,
    }

    def canonical(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "", value.casefold())
        return aliases.get(normalized, normalized)

    return canonical(actual) == canonical(expected)


def _is_civitai_login_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.hostname in _CIVITAI_HOSTS and parsed.path.casefold().startswith("/login")


def _civitai_authentication_message() -> str:
    return (
        "Civitai requires authentication to download this LoRA. "
        "Run Start 10MinVideoMaker.bat and configure a Civitai API token."
    )
