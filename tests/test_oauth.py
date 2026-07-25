from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse
import unittest

from tenminvideomaker.oauth import (
    GMAIL_IMAP_SMTP_SCOPE,
    GOOGLE_DRIVE_READONLY_SCOPE,
    build_authorization_url,
    refresh_access_token,
)


class _Response:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class OAuthTests(unittest.TestCase):
    def test_authorization_url_uses_offline_pkce_mail_and_drive_scopes(self) -> None:
        url = build_authorization_url(
            client_id="client-id",
            redirect_uri="http://127.0.0.1:54321/oauth2callback",
            state="state-token",
            code_challenge="challenge-token",
            login_hint="owner@example.com",
        )
        query = parse_qs(urlparse(url).query)
        self.assertEqual(
            frozenset(query["scope"][0].split()),
            frozenset({GMAIL_IMAP_SMTP_SCOPE, GOOGLE_DRIVE_READONLY_SCOPE}),
        )
        self.assertEqual(query["access_type"], ["offline"])
        self.assertEqual(query["prompt"], ["consent"])
        self.assertEqual(query["state"], ["state-token"])
        self.assertEqual(query["code_challenge_method"], ["S256"])
        self.assertEqual(query["login_hint"], ["owner@example.com"])

    def test_refresh_access_token_posts_refresh_grant(self) -> None:
        captured = {}

        def opener(request, *, timeout):
            captured["url"] = request.full_url
            captured["body"] = parse_qs(request.data.decode("ascii"))
            captured["timeout"] = timeout
            return _Response({"access_token": "fresh-access", "expires_in": 1800})

        token, expires_in = refresh_access_token(
            client_id="client-id",
            client_secret="client-secret",
            refresh_token="refresh-token",
            opener=opener,
        )
        self.assertEqual(token, "fresh-access")
        self.assertEqual(expires_in, 1800)
        self.assertEqual(captured["body"]["grant_type"], ["refresh_token"])
        self.assertEqual(captured["body"]["refresh_token"], ["refresh-token"])
        self.assertEqual(captured["timeout"], 30)


if __name__ == "__main__":
    unittest.main()
