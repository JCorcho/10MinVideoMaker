"""Small standard-library client for project-scoped ComfyUI prompt execution."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class ComfyHttpError(RuntimeError):
    """Raised when ComfyUI rejects, fails, or times out a project prompt."""


class ComfyPromptRejectedError(ComfyHttpError):
    """The /prompt response proves that ComfyUI did not accept the workflow."""


class ComfyPromptDispatchAmbiguousError(ComfyHttpError):
    """Transport failed without proving whether /prompt accepted the workflow."""


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
        except HTTPError as error:
            if method == "POST" and path == "/prompt" and 400 <= error.code < 500:
                raise ComfyPromptRejectedError(
                    f"ComfyUI rejected POST /prompt with HTTP {error.code}."
                ) from error
            if method == "POST" and path == "/prompt":
                raise ComfyPromptDispatchAmbiguousError(
                    f"ComfyUI POST /prompt may have been accepted: {error}"
                ) from error
            raise ComfyHttpError(f"ComfyUI {method} {path} failed: {error}") from error
        except OSError as error:
            if method == "POST" and path == "/prompt":
                raise ComfyPromptDispatchAmbiguousError(
                    f"ComfyUI POST /prompt may have been accepted: {error}"
                ) from error
            raise ComfyHttpError(f"ComfyUI {method} {path} failed: {error}") from error
        if not content:
            return {}
        try:
            return json.loads(content)
        except json.JSONDecodeError as error:
            if method == "POST" and path == "/prompt":
                raise ComfyPromptDispatchAmbiguousError(
                    "ComfyUI POST /prompt returned invalid JSON after dispatch."
                ) from error
            raise ComfyHttpError(f"ComfyUI returned invalid JSON for {path}.") from error

    def alive(self) -> bool:
        try:
            self._json_request("GET", "/system_stats", timeout=3)
            return True
        except ComfyHttpError:
            return False

    def system_stats(self) -> Mapping[str, Any]:
        """Return live ComfyUI system statistics for project telemetry."""
        response = self._json_request("GET", "/system_stats", timeout=10)
        if not isinstance(response, Mapping):
            raise ComfyHttpError("ComfyUI returned invalid system statistics.")
        return response

    def queue_counts(self) -> tuple[int, int]:
        """Return running and pending prompt counts without exposing workflow contents."""
        queue = self._json_request("GET", "/queue", timeout=10)
        if not isinstance(queue, Mapping):
            raise ComfyHttpError("ComfyUI returned an invalid queue response.")
        running = queue.get("queue_running", [])
        pending = queue.get("queue_pending", [])
        if not isinstance(running, list) or not isinstance(pending, list):
            raise ComfyHttpError("ComfyUI returned invalid queue lists.")
        return len(running), len(pending)

    def object_info(self, node_type: str) -> Mapping[str, Any]:
        if not isinstance(node_type, str) or not node_type:
            raise ComfyHttpError("node_type must be non-empty text.")
        response = self._json_request("GET", f"/object_info/{node_type}", timeout=10)
        if not isinstance(response, Mapping):
            raise ComfyHttpError("ComfyUI returned invalid node information.")
        return response

    def queue_prompt(self, workflow: Mapping[str, Any]) -> str:
        response = self._json_request(
            "POST", "/prompt", {"prompt": workflow, "client_id": self.client_id}
        )
        prompt_id = response.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            node_errors = response.get("node_errors")
            raise ComfyPromptRejectedError(
                f"ComfyUI did not accept the prompt: {node_errors or response}"
            )
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

    def completed_prompt(self, prompt_id: str) -> Mapping[str, Any] | None:
        """Return a successful history record without waiting.

        A persisted prompt ID lets a restarted supervisor reclaim work that
        ComfyUI finished while the Python process was unavailable.
        """
        response = self._json_request("GET", f"/history/{prompt_id}", timeout=10)
        record = response.get(prompt_id) if isinstance(response, Mapping) else None
        if not isinstance(record, Mapping):
            return None
        status = record.get("status", {})
        if status.get("completed") and status.get("status_str") == "success":
            return record
        if status.get("status_str") in {"error", "failed"}:
            raise ComfyHttpError(_history_error(record, prompt_id))
        return None

    def prompt_is_queued(self, prompt_id: str) -> bool:
        """Return whether this exact project-owned prompt is pending or running."""
        queue = self._json_request("GET", "/queue", timeout=10)
        if not isinstance(queue, Mapping):
            raise ComfyHttpError("ComfyUI returned an invalid queue response.")
        pending = queue.get("queue_pending", [])
        running = queue.get("queue_running", [])
        if not isinstance(pending, list) or not isinstance(running, list):
            raise ComfyHttpError("ComfyUI returned invalid queue lists.")
        return any(
            _queue_prompt_id(item) == prompt_id
            and _queue_client_id(item) == self.client_id
            for item in (*pending, *running)
        )

    def cancel_prompt(self, prompt_id: str) -> None:
        """Best-effort compatibility wrapper for an owned prompt cancellation."""
        self.cancel_owned_prompt(prompt_id)

    def cancel_owned_prompt(self, prompt_id: str) -> bool:
        """Cancel this exact prompt only when the queue confirms project ownership."""
        queue = self._json_request("GET", "/queue", timeout=10)
        pending = queue.get("queue_pending", []) if isinstance(queue, Mapping) else []
        running = queue.get("queue_running", []) if isinstance(queue, Mapping) else []
        pending_owned = any(
            _queue_prompt_id(item) == prompt_id
            and _queue_client_id(item) == self.client_id
            for item in pending
        )
        running_owned = any(
            _queue_prompt_id(item) == prompt_id
            and _queue_client_id(item) == self.client_id
            for item in running
        )
        if pending_owned:
            self._json_request("POST", "/queue", {"delete": [prompt_id]}, timeout=10)
        if running_owned:
            fresh = self._json_request("GET", "/queue", timeout=10)
            fresh_running = (
                fresh.get("queue_running", []) if isinstance(fresh, Mapping) else []
            )
            still_owned = any(
                _queue_prompt_id(item) == prompt_id
                and _queue_client_id(item) == self.client_id
                for item in fresh_running
            )
            safe_to_interrupt = bool(fresh_running) and all(
                _queue_client_id(item) == self.client_id for item in fresh_running
            )
            if still_owned and safe_to_interrupt:
                self._json_request("POST", "/interrupt", {}, timeout=10)
                return True
        return pending_owned

    def cancel_project_prompts(self) -> tuple[str, ...]:
        """Cancel queued/running prompts owned by this project client only."""
        queue = self._json_request("GET", "/queue", timeout=10)
        pending = queue.get("queue_pending", []) if isinstance(queue, Mapping) else []
        running = queue.get("queue_running", []) if isinstance(queue, Mapping) else []
        pending_ids = [
            prompt_id
            for item in pending
            if _queue_client_id(item) == self.client_id
            if (prompt_id := _queue_prompt_id(item)) is not None
        ]
        running_ids = [
            prompt_id
            for item in running
            if _queue_client_id(item) == self.client_id
            if (prompt_id := _queue_prompt_id(item)) is not None
        ]
        if pending_ids:
            self._json_request("POST", "/queue", {"delete": pending_ids}, timeout=10)
        cancelled_running_ids: list[str] = []
        if running_ids:
            fresh = self._json_request("GET", "/queue", timeout=10)
            fresh_running = (
                fresh.get("queue_running", []) if isinstance(fresh, Mapping) else []
            )
            if fresh_running and all(
                _queue_client_id(item) == self.client_id for item in fresh_running
            ):
                cancelled_running_ids = [
                    prompt_id
                    for item in fresh_running
                    if (prompt_id := _queue_prompt_id(item)) is not None
                ]
                self._json_request("POST", "/interrupt", {}, timeout=10)
        return tuple(dict.fromkeys((*pending_ids, *cancelled_running_ids)))

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


def _queue_client_id(item: Any) -> str | None:
    if (
        isinstance(item, list)
        and len(item) > 3
        and isinstance(item[3], Mapping)
        and isinstance(item[3].get("client_id"), str)
    ):
        return item[3]["client_id"]
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


def find_video_output(
    record: Mapping[str, Any],
    output_node_id: str,
    *,
    expected_suffixes: tuple[str, ...] = (".mp4",),
) -> Mapping[str, Any]:
    """Find an expected video only beneath the designated raw VHS output node.

    A workflow can contain additional video-producing delivery nodes. Their
    output must never be chosen for the durable clean scene clip.
    """
    if not isinstance(output_node_id, str) or not output_node_id:
        raise ComfyHttpError("The designated raw video output node ID is required.")
    if (
        not isinstance(expected_suffixes, tuple)
        or not expected_suffixes
        or any(
            not isinstance(suffix, str)
            or not suffix.startswith(".")
            or suffix != suffix.casefold()
            for suffix in expected_suffixes
        )
    ):
        raise ComfyHttpError(
            "Expected video suffixes must be a non-empty tuple of lowercase extensions."
        )
    outputs = record.get("outputs", {})
    if not isinstance(outputs, Mapping):
        raise ComfyHttpError("ComfyUI completed I2V with invalid output metadata.")
    selected_output = outputs.get(output_node_id)
    if selected_output is None:
        raise ComfyHttpError(
            f"ComfyUI completed I2V without output from raw video node {output_node_id}."
        )
    stack: list[Any] = [selected_output]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            filename = value.get("filename")
            if isinstance(filename, str) and filename.casefold().endswith(
                expected_suffixes
            ):
                return value
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    raise ComfyHttpError(
        f"ComfyUI raw video node {output_node_id} returned no "
        f"{'/'.join(expected_suffixes)} output metadata."
    )
