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
        self.assertIs(find_video_output(record, "36"), metadata)

    def test_finds_lossless_continuation_mkv_only_when_requested(self) -> None:
        metadata = {
            "filename": "window_00001-audio.mkv",
            "subfolder": "10MinVideoMaker/job/continuation",
            "type": "temp",
        }
        record = {"outputs": {"raw-window": {"gifs": [metadata]}}}

        with self.assertRaises(ComfyHttpError):
            find_video_output(record, "raw-window")
        self.assertIs(
            find_video_output(
                record,
                "raw-window",
                expected_suffixes=(".mkv",),
            ),
            metadata,
        )

    def test_ignores_watermarked_delivery_mp4_from_another_output_node(self) -> None:
        raw_metadata = {
            "filename": "scene_0001_raw.mp4",
            "subfolder": "10MinVideoMaker/job/clips",
            "type": "temp",
        }
        watermarked_metadata = {
            "filename": "discord-watermarked.mp4",
            "subfolder": "output",
            "type": "output",
        }
        record = {
            "outputs": {
                "raw-vhs": {"gifs": [raw_metadata]},
                "discord-delivery": {"gifs": [watermarked_metadata]},
            }
        }

        self.assertIs(find_video_output(record, "raw-vhs"), raw_metadata)

    def test_missing_video_metadata_is_an_error(self) -> None:
        with self.assertRaises(ComfyHttpError):
            find_video_output({"outputs": {"1": {"images": []}}}, "1")

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

    def test_system_stats_returns_live_mapping(self) -> None:
        client = ComfyHttpClient()
        document = {"devices": [{"vram_total": 16, "vram_free": 12}]}
        with patch.object(client, "_json_request", return_value=document) as request:
            self.assertIs(client.system_stats(), document)
        request.assert_called_once_with("GET", "/system_stats", timeout=10)

    def test_system_stats_rejects_non_mapping_response(self) -> None:
        client = ComfyHttpClient()
        with patch.object(client, "_json_request", return_value=[]):
            with self.assertRaisesRegex(ComfyHttpError, "invalid system statistics"):
                client.system_stats()

    def test_persisted_prompt_can_be_reclaimed_from_history_or_owned_queue(self) -> None:
        client = ComfyHttpClient(client_id="10MinVideoMaker-supervisor")
        successful = {
            "prompt-1": {
                "status": {"completed": True, "status_str": "success"},
                "outputs": {},
            }
        }
        with patch.object(client, "_json_request", return_value=successful):
            self.assertIs(client.completed_prompt("prompt-1"), successful["prompt-1"])

        queue = {
            "queue_pending": [
                [1, "prompt-2", {}, {"client_id": client.client_id}, []],
                [2, "other", {}, {"client_id": "another-client"}, []],
            ],
            "queue_running": [],
        }
        with patch.object(client, "_json_request", return_value=queue):
            self.assertTrue(client.prompt_is_queued("prompt-2"))
            self.assertFalse(client.prompt_is_queued("other"))

    def test_missing_or_failed_persisted_prompt_is_reported(self) -> None:
        client = ComfyHttpClient()
        with patch.object(client, "_json_request", return_value={}):
            self.assertIsNone(client.completed_prompt("missing"))
        failed = {
            "failed": {
                "status": {
                    "completed": True,
                    "status_str": "error",
                    "messages": [],
                }
            }
        }
        with patch.object(client, "_json_request", return_value=failed):
            with self.assertRaises(ComfyHttpError):
                client.completed_prompt("failed")

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
            side_effect=[queue, {}, queue],
        ) as request:
            cancelled = client.cancel_project_prompts()

        self.assertEqual(cancelled, ("project-pending",))
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
                unittest.mock.call("GET", "/queue", timeout=10),
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

    def test_cancel_owned_prompt_refuses_same_id_from_another_client(self) -> None:
        client = ComfyHttpClient(client_id="10MinVideoMaker-supervisor")
        queue = {
            "queue_pending": [
                [1, "delivery-prompt", {}, {"client_id": "another-client"}, []]
            ],
            "queue_running": [],
        }
        with patch.object(client, "_json_request", return_value=queue) as request:
            self.assertFalse(client.cancel_owned_prompt("delivery-prompt"))
        request.assert_called_once_with("GET", "/queue", timeout=10)

    def test_cancel_owned_prompt_deletes_exact_project_prompt(self) -> None:
        client = ComfyHttpClient(client_id="10MinVideoMaker-supervisor")
        queue = {
            "queue_pending": [
                [1, "delivery-prompt", {}, {"client_id": client.client_id}, []],
                [2, "other", {}, {"client_id": client.client_id}, []],
            ],
            "queue_running": [],
        }
        with patch.object(
            client,
            "_json_request",
            side_effect=[queue, {}],
        ) as request:
            self.assertTrue(client.cancel_owned_prompt("delivery-prompt"))
        self.assertEqual(
            request.call_args_list,
            [
                unittest.mock.call("GET", "/queue", timeout=10),
                unittest.mock.call(
                    "POST",
                    "/queue",
                    {"delete": ["delivery-prompt"]},
                    timeout=10,
                ),
            ],
        )

    def test_cancel_owned_running_prompt_rechecks_exclusive_ownership(self) -> None:
        client = ComfyHttpClient(client_id="10MinVideoMaker-supervisor")
        queue = {
            "queue_pending": [],
            "queue_running": [
                [1, "owned-running", {}, {"client_id": client.client_id}, []]
            ],
        }
        with patch.object(
            client,
            "_json_request",
            side_effect=[queue, queue, {}],
        ) as request:
            self.assertTrue(client.cancel_owned_prompt("owned-running"))
        self.assertEqual(
            request.call_args_list,
            [
                unittest.mock.call("GET", "/queue", timeout=10),
                unittest.mock.call("GET", "/queue", timeout=10),
                unittest.mock.call("POST", "/interrupt", {}, timeout=10),
            ],
        )


if __name__ == "__main__":
    unittest.main()
