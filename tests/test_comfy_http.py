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


if __name__ == "__main__":
    unittest.main()
