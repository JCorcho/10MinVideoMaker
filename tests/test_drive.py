from __future__ import annotations

import json
from urllib.error import HTTPError
import unittest

from tenminvideomaker.drive import (
    DriveDownloadError,
    download_google_drive_json,
    google_drive_file_urls,
    parse_google_drive_file_url,
    validate_google_drive_access,
)


FILE_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz"
SHARE_URL = f"https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing"


class _Response:
    def __init__(
        self,
        content: bytes,
        *,
        url: str,
        content_type: str = "application/json",
    ):
        self.content = content
        self.url = url
        self.headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(content)),
        }

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def geturl(self) -> str:
        return self.url

    def read(self, limit: int = -1) -> bytes:
        return self.content if limit < 0 else self.content[:limit]


class DriveTests(unittest.TestCase):
    def test_supported_share_urls_extract_file_id_and_reject_folders(self) -> None:
        self.assertEqual(parse_google_drive_file_url(SHARE_URL).file_id, FILE_ID)
        self.assertEqual(
            parse_google_drive_file_url(f"https://drive.google.com/open?id={FILE_ID}").file_id,
            FILE_ID,
        )
        self.assertIsNone(
            parse_google_drive_file_url(f"https://drive.google.com/drive/folders/{FILE_ID}")
        )
        self.assertIsNone(parse_google_drive_file_url("https://example.com/job.json"))

    def test_html_and_plain_text_links_are_extracted_and_deduplicated(self) -> None:
        text = f'Open <a href="{SHARE_URL}">scenes.json</a> or {SHARE_URL}.'
        self.assertEqual(google_drive_file_urls(text), (SHARE_URL,))

    def test_public_file_download_is_bounded_and_does_not_send_bearer_token(self) -> None:
        captured = {}

        def opener(request, *, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            captured["timeout"] = timeout
            return _Response(
                b'{"job_id":"public","scenes":[]}',
                url=request.full_url,
            )

        text = download_google_drive_json(SHARE_URL, opener=opener)
        self.assertEqual(json.loads(text)["job_id"], "public")
        self.assertIn("drive.usercontent.google.com/download", captured["url"])
        self.assertIsNone(captured["authorization"])
        self.assertEqual(captured["timeout"], 30)

    def test_private_file_falls_back_to_authenticated_drive_api(self) -> None:
        requests = []

        def opener(request, *, timeout):
            requests.append(request)
            if len(requests) == 1:
                return _Response(
                    b"<html>Sign in</html>",
                    url=request.full_url,
                    content_type="text/html",
                )
            return _Response(
                b'{"job_id":"private","scenes":[]}',
                url=request.full_url,
            )

        text = download_google_drive_json(
            SHARE_URL,
            access_token="access-token",
            opener=opener,
        )
        self.assertEqual(json.loads(text)["job_id"], "private")
        self.assertEqual(
            requests[1].get_header("Authorization"),
            "Bearer access-token",
        )
        self.assertIn(f"/drive/v3/files/{FILE_ID}?alt=media", requests[1].full_url)

    def test_private_file_without_oauth_has_actionable_error(self) -> None:
        def opener(request, *, timeout):
            raise HTTPError(request.full_url, 403, "Forbidden", {}, None)

        with self.assertRaisesRegex(DriveDownloadError, "Anyone with the link"):
            download_google_drive_json(SHARE_URL, opener=opener)

    def test_download_rejects_oversized_json(self) -> None:
        def opener(request, *, timeout):
            return _Response(b"123456", url=request.full_url)

        with self.assertRaisesRegex(DriveDownloadError, "safety limit"):
            download_google_drive_json(SHARE_URL, opener=opener, max_bytes=5)

    def test_oauth_validation_uses_drive_about_endpoint(self) -> None:
        captured = {}

        def opener(request, *, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            return _Response(
                b'{"user":{"emailAddress":"owner@example.com"}}',
                url=request.full_url,
            )

        validate_google_drive_access("access-token", opener=opener)
        self.assertEqual(
            captured["url"],
            "https://www.googleapis.com/drive/v3/about?fields=user",
        )
        self.assertEqual(captured["authorization"], "Bearer access-token")


if __name__ == "__main__":
    unittest.main()
