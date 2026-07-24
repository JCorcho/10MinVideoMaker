"""Small standard-library client for project-scoped ComfyUI prompt execution."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class ComfyHttpError(RuntimeError):
    """Raised when ComfyUI rejects, fails, or times out a project prompt."""


class ComfyHttpClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        *,
        client_id: str = "10MinVideoMaker-supervisor",
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self._sleep = sleep

    def _json_request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout: float = 30,
    ) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                content = response.read()
        except OSError as error:
            raise ComfyHttpError(f"ComfyUI {method} {path} failed: {error}") from error
        if not content:
            return {}
        try:
            return json.loads(content)
        except json.JSONDecodeError as error:
            raise ComfyHttpError(f"ComfyUI returned invalid JSON for {path}.") from error

    def alive(self) -> bool:
        try:
            self._json_request("GET", "/system_stats", timeout=3)
            return True
        except ComfyHttpError:
            return False

    def queue_prompt(self, workflow: Mapping[str, Any]) -> str:
        response = self._json_request(
            "POST",
            "/prompt",
            {"prompt": workflow, "client_id": self.client_id},
        )
        prompt_id = response.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            node_errors = response.get("node_errors")
            raise ComfyHttpError(f"ComfyUI did not accept the prompt: {node_errors or response}")
        return prompt_id

    def wait_for_prompt(
        self,
        prompt_id: str,
        *,
        timeout_seconds: float,
        poll_seconds: float = 2.0,
    ) -> Mapping[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            response = self._json_request("GET", f"/history/{prompt_id}", timeout=10)
            record = response.get(prompt_id) if isinstance(response, Mapping) else None
            if isinstance(record, Mapping):
                status = record.get("status", {})
                if status.get("completed") and status.get("status_str") == "success":
                    return record
                if status.get("status_str") in {"error", "failed"}:
                    raise ComfyHttpError(_history_error(record, prompt_id))
            self._sleep(poll_seconds)
        self.cancel_prompt(prompt_id)
        raise ComfyHttpError(f"ComfyUI prompt {prompt_id} exceeded {timeout_seconds:g} seconds.")

    def cancel_prompt(self, prompt_id: str) -> None:
        queue = self._json_request("GET", "/queue", timeout=10)
        pending = queue.get("queue_pending", []) if isinstance(queue, Mapping) else []
        running = queue.get("queue_running", []) if isinstance(queue, Mapping) else []
        if any(_queue_prompt_id(item) == prompt_id for item in pending):
            self._json_request("POST", "/queue", {"delete": [prompt_id]}, timeout=10)
        if any(_queue_prompt_id(item) == prompt_id for item in running):
            self._json_request("POST", "/interrupt", {}, timeout=10)

    def free_memory(self) -> None:
        self._json_request(
            "POST",
            "/free",
            {"unload_models": True, "free_memory": True},
            timeout=30,
        )

    def resolve_lora_asset(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        response = self._json_request(
            "POST",
            "/10minvideomaker/assets/resolve",
            payload,
            timeout=1800,
        )
        if not isinstance(response, Mapping):
            raise ComfyHttpError("ComfyUI returned an invalid LoRA asset response.")
        return response

    def download_output(self, metadata: Mapping[str, Any], destination: str | Path) -> Path:
        filename = metadata.get("filename")
        if not isinstance(filename, str) or not filename:
            raise ComfyHttpError("ComfyUI output metadata has no filename.")
        query = urlencode(
            {
                "filename": filename,
                "subfolder": str(metadata.get("subfolder") or ""),
                "type": str(metadata.get("type") or "output"),
            }
        )
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with urlopen(f"{self.base_url}/view?{query}", timeout=300) as response, temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            if temporary.stat().st_size == 0:
                raise ComfyHttpError("Downloaded ComfyUI output is empty.")
            temporary.replace(destination)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise ComfyHttpError(f"Could not download ComfyUI output {filename}: {error}") from error
        return destination


def _queue_prompt_id(item: Any) -> str | None:
    if isinstance(item, list) and len(item) > 1 and isinstance(item[1], str):
        return item[1]
    return None


def _history_error(record: Mapping[str, Any], prompt_id: str) -> str:
    messages = record.get("status", {}).get("messages", [])
    for event in reversed(messages if isinstance(messages, list) else []):
        if (
            isinstance(event, list)
            and len(event) == 2
            and event[0] == "execution_error"
            and isinstance(event[1], Mapping)
        ):
            detail = event[1].get("exception_message") or event[1].get("exception_type")
            return f"ComfyUI prompt {prompt_id} failed: {detail or event[1]}"
    return f"ComfyUI prompt {prompt_id} failed."


def find_video_output(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Find the saved or temporary MP4 metadata returned by VHS_VideoCombine."""
    stack: list[Any] = [record.get("outputs", {})]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            filename = value.get("filename")
            if isinstance(filename, str) and filename.casefold().endswith(".mp4"):
                return value
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    raise ComfyHttpError("ComfyUI completed I2V but returned no MP4 output metadata.")
