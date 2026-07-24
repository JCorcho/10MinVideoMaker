from __future__ import annotations

import unittest

from tenminvideomaker.comfy_http import ComfyHttpError, find_video_output


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


if __name__ == "__main__":
    unittest.main()
