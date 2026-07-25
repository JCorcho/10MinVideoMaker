"""Loopback-only ComfyUI routes for active model-path asset resolution."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from .assets import AssetResolution, LocalLoraRequirement, LoraAssetManager
from .configuration import load_project_environment
from .constants import I2V_DYNAMIC_BASE_MODEL
from .contracts import LoraSpec, civitai_version_id

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
ASSET_RESOLVE_ROUTE = "/10minvideomaker/assets/resolve"

_REGISTERED = False


def _asset_resolution_mapping(result: AssetResolution) -> dict[str, Any]:
    return {
        "name": result.name,
        "path": str(result.path) if result.path else None,
        "downloaded": result.downloaded,
        "error": result.error,
        "local_filename": result.local_filename,
        "base_model": result.base_model,
        "succeeded": result.succeeded,
    }


def _positive_optional_integer(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer when provided.")
    return value


def _lora_from_request(value: Any) -> LoraSpec:
    if not isinstance(value, Mapping):
        raise ValueError("lora must be an object.")
    name = value.get("name")
    download_url = value.get("download_url")
    weight = value.get("weight")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("lora.name must be non-empty text.")
    if not isinstance(download_url, str):
        raise ValueError("lora.download_url must be an HTTPS URL.")
    parsed = urlparse(download_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("lora.download_url must be an HTTPS URL.")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError("lora.weight must be numeric.")
    model_id = _positive_optional_integer(value.get("model_id"), "lora.model_id")
    version_id = _positive_optional_integer(value.get("version_id"), "lora.version_id")
    url_version_id = civitai_version_id(download_url)
    if version_id and url_version_id and version_id != url_version_id:
        raise ValueError("lora.version_id does not match lora.download_url.")
    return LoraSpec(
        name=name.strip(),
        download_url=download_url,
        weight=float(weight),
        model_id=model_id,
        version_id=version_id or url_version_id,
    )


def resolve_asset_request(
    document: Mapping[str, Any],
    manager: LoraAssetManager,
) -> dict[str, Any]:
    kind = document.get("kind")
    if kind == "dynamic":
        expected_base_model = document.get("expected_base_model")
        if expected_base_model is not None and expected_base_model != I2V_DYNAMIC_BASE_MODEL:
            raise ValueError(
                f"expected_base_model must be {I2V_DYNAMIC_BASE_MODEL} when provided."
            )
        result = manager.resolve_or_download(
            _lora_from_request(document.get("lora")),
            expected_base_model=expected_base_model,
        )
    elif kind == "required":
        filename = document.get("filename")
        weight = document.get("weight")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must be non-empty text.")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("weight must be numeric.")
        result = manager.require_local(
            LocalLoraRequirement(filename.strip(), float(weight))
        )
    else:
        raise ValueError("kind must be dynamic or required.")
    return _asset_resolution_mapping(result)


def register_routes() -> bool:
    """Register once when imported by a running ComfyUI server."""
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        import folder_paths
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return False
    if PromptServer.instance is None:
        return False

    @PromptServer.instance.routes.post(ASSET_RESOLVE_ROUTE)
    async def resolve_asset(request):
        if request.remote not in {"127.0.0.1", "::1", "::ffff:127.0.0.1"}:
            return web.json_response(
                {"error": "This project route accepts loopback requests only."},
                status=403,
            )
        try:
            document = await request.json()
            if not isinstance(document, Mapping):
                raise ValueError("Request body must be a JSON object.")
            environment = load_project_environment(PROJECT_ROOT)
            manager = LoraAssetManager(
                folder_paths.get_folder_paths("loras"),
                RUNTIME_ROOT / "asset_manifest.json",
                visible_lora_names=folder_paths.get_filename_list("loras"),
                civitai_token=environment.get("TENMIN_CIVITAI_TOKEN", ""),
            )
            result = await asyncio.to_thread(resolve_asset_request, document, manager)
            return web.json_response(result)
        except (ValueError, OSError) as error:
            return web.json_response({"error": str(error)}, status=400)

    _REGISTERED = True
    return True
