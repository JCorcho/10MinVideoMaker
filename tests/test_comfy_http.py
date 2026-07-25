from __future__ import annotations

import unittest
from unittest.mock import patch

from tenminvideomaker.comfy_http import ComfyHttpClient, ComfyHttpError, find_video_output


class ComfyHttpTests(unittest.TestCase):
    def test_finds_nested_vhs_video_metadata(self) -> None:
        metadata = {
            "filename": "scene_0001_00001-audio.mp4",
            "subfolder": "10MinVideoMaker/job/clips",
            "type": "temp",
        }
        record = {"outputs": {"36": {"gifs": [metadata]}}}
        self.assertIs(find_video_output(record), metadata)

    def test_missing_video_metadata_is_an_error(self) -> None:
        with self.assertRaises(ComfyHttpError):
            find_video_output({"outputs": {"1": {"images": []}}})

    def test_queue_counts_returns_only_redacted_totals(self) -> None:
        client = ComfyHttpClient()
        with patch.object(
            client,
            "_json_request",
            return_value={
                "queue_running": [[1, "secret-prompt", {"workflow": "hidden"}]],
                "queue_pending": [
                    [2, "another-secret", {"workflow": "hidden"}],
                    [3, "third-secret", {"workflow": "hidden"}],
                ],
            },
        ) as request:
            self.assertEqual(client.queue_counts(), (1, 2))
        request.assert_called_once_with("GET", "/queue", timeout=10)

    def test_queue_counts_rejects_invalid_queue_shape(self) -> None:
        client = ComfyHttpClient()
        with patch.object(
            client,
            "_json_request",
            return_value={"queue_running": "not-a-list", "queue_pending": []},
        ):
            with self.assertRaisesRegex(ComfyHttpError, "invalid queue lists"):
                client.queue_counts()

    def test_lora_resolution_uses_project_route_and_long_download_timeout(self) -> None:
        client = ComfyHttpClient()
        payload = {"kind": "required", "filename": "DMD.safetensors", "weight": 1.0}
        with patch.object(
            client,
            "_json_request",
            return_value={"succeeded": True, "path": "DMD.safetensors"},
        ) as request:
            response = client.resolve_lora_asset(payload)
        request.assert_called_once_with(
            "POST",
            "/10minvideomaker/assets/resolve",
            payload,
            timeout=1800,
        )
        self.assertTrue(response["succeeded"])

    def test_cancel_project_prompts_leaves_other_clients_untouched(self) -> None:
        client = ComfyHttpClient(client_id="10MinVideoMaker-supervisor")
        queue = {
            "queue_pending": [
                [1, "project-pending", {}, {"client_id": client.client_id}, []],
                [2, "other-pending", {}, {"client_id": "another-client"}, []],
            ],
            "queue_running": [
                [3, "project-running", {}, {"client_id": client.client_id}, []],
                [4, "other-running", {}, {"client_id": "another-client"}, []],
            ],
        }
        with patch.object(
            client,
            "_json_request",
            side_effect=[queue, {}, {}],
        ) as request:
            cancelled = client.cancel_project_prompts()

        self.assertEqual(cancelled, ("project-pending", "project-running"))
        self.assertEqual(
            request.call_args_list,
            [
                unittest.mock.call("GET", "/queue", timeout=10),
                unittest.mock.call(
                    "POST",
                    "/queue",
                    {"delete": ["project-pending"]},
                    timeout=10,
                ),
                unittest.mock.call("POST", "/interrupt", {}, timeout=10),
            ],
        )

    def test_cancel_project_prompts_does_nothing_for_other_clients(self) -> None:
        client = ComfyHttpClient(client_id="10MinVideoMaker-supervisor")
        queue = {
            "queue_pending": [
                [1, "other-pending", {}, {"client_id": "another-client"}, []]
            ],
            "queue_running": [
                [2, "other-running", {}, {"client_id": "another-client"}, []]
            ],
        }
        with patch.object(client, "_json_request", return_value=queue) as request:
            self.assertEqual(client.cancel_project_prompts(), ())
        request.assert_called_once_with("GET", "/queue", timeout=10)


if __name__ == "__main__":
    unittest.main()
