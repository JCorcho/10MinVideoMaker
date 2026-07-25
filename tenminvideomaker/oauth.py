"""Google OAuth2 desktop authorization and access-token refresh helpers."""

from __future__ import annotations

import base64
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import secrets
import time
from typing import Any, Callable, Mapping
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen
import webbrowser

GOOGLE_AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GMAIL_IMAP_SMTP_SCOPE = "https://mail.google.com/"
GOOGLE_DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
GOOGLE_OAUTH_SCOPES = (GMAIL_IMAP_SMTP_SCOPE, GOOGLE_DRIVE_READONLY_SCOPE)
GOOGLE_OAUTH_SCOPE_VALUE = " ".join(GOOGLE_OAUTH_SCOPES)
GOOGLE_CREDENTIALS_URL = "https://console.cloud.google.com/apis/credentials"


class OAuthError(RuntimeError):
    """Raised when Google OAuth setup or token refresh fails."""


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def build_authorization_url(
    *,
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str,
    login_hint: str = "",
) -> str:
    parameters = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": GOOGLE_OAUTH_SCOPE_VALUE,
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    if login_hint:
        parameters["login_hint"] = login_hint
    return f"{GOOGLE_AUTHORIZATION_ENDPOINT}?{urlencode(parameters)}"


def _token_request(
    parameters: Mapping[str, str],
    *,
    opener: Callable[..., Any] = urlopen,
) -> Mapping[str, Any]:
    request = Request(
        GOOGLE_TOKEN_ENDPOINT,
        data=urlencode(parameters).encode("ascii"),
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with opener(request, timeout=30) as response:
            payload = json.loads(response.read())
    except HTTPError as error:
        try:
            payload = json.loads(error.read())
        except (OSError, json.JSONDecodeError):
            raise OAuthError(f"Google token request failed with HTTP {error.code}.") from error
    except (OSError, json.JSONDecodeError) as error:
        raise OAuthError(f"Google token request failed: {error}") from error
    if not isinstance(payload, Mapping):
        raise OAuthError("Google token endpoint returned an invalid response.")
    if "error" in payload:
        detail = payload.get("error_description") or payload["error"]
        raise OAuthError(f"Google rejected the token request: {detail}")
    return payload


def exchange_authorization_code(
    *,
    client_id: str,
    client_secret: str,
    code: str,
    code_verifier: str,
    redirect_uri: str,
    opener: Callable[..., Any] = urlopen,
) -> Mapping[str, Any]:
    parameters = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "code_verifier": code_verifier,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    }
    return _token_request(parameters, opener=opener)


def refresh_access_token(
    *,
    client_id: str,
    client_secret: str,
    refresh_token: str,
    opener: Callable[..., Any] = urlopen,
) -> tuple[str, int]:
    payload = _token_request(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        opener=opener,
    )
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise OAuthError("Google refresh response did not contain an access token.")
    expires_in = payload.get("expires_in", 3600)
    try:
        expires_in = int(expires_in)
    except (TypeError, ValueError):
        expires_in = 3600
    return access_token, max(expires_in, 60)


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    result: dict[str, str] = {}

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        parameters = parse_qs(urlparse(self.path).query)
        self.__class__.result = {
            key: values[0] for key, values in parameters.items() if values
        }
        success = "code" in self.__class__.result
        title = "Authorization received" if success else "Authorization failed"
        message = (
            "You may close this window and return to 10MinVideoMaker."
            if success
            else "Google did not return an authorization code. Return to the launcher for details."
        )
        body = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{title}</title></head><body><h1>{title}</h1><p>{message}</p></body></html>"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def authorize_desktop_app(
    *,
    client_id: str,
    client_secret: str,
    login_hint: str = "",
    timeout_seconds: int = 300,
    open_browser: Callable[[str], Any] = webbrowser.open,
    notify: Callable[[str], None] = print,
) -> str:
    """Open Google consent in the system browser and return a persistent refresh token."""
    code_verifier = secrets.token_urlsafe(72)[:128]
    code_challenge = _base64url(sha256(code_verifier.encode("ascii")).digest())
    state = secrets.token_urlsafe(32)
    _OAuthCallbackHandler.result = {}
    server = HTTPServer(("127.0.0.1", 0), _OAuthCallbackHandler)
    server.timeout = timeout_seconds
    redirect_uri = f"http://127.0.0.1:{server.server_port}/oauth2callback"
    authorization_url = build_authorization_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
        login_hint=login_hint,
    )
    notify("\nOpen this Google authorization link:")
    notify(authorization_url)
    notify("\nWaiting for Google to return authorization to this computer...")
    open_browser(authorization_url)
    started = time.monotonic()
    try:
        server.handle_request()
    finally:
        server.server_close()
    if time.monotonic() - started >= timeout_seconds and not _OAuthCallbackHandler.result:
        raise OAuthError("Timed out waiting for Google authorization.")
    result = _OAuthCallbackHandler.result
    if result.get("state") != state:
        raise OAuthError("OAuth state validation failed; authorization was not accepted.")
    if "error" in result:
        raise OAuthError(f"Google authorization failed: {result['error']}")
    code = result.get("code")
    if not code:
        raise OAuthError("Google did not return an authorization code.")
    payload = exchange_authorization_code(
        client_id=client_id,
        client_secret=client_secret,
        code=code,
        code_verifier=code_verifier,
        redirect_uri=redirect_uri,
    )
    refresh_token = payload.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise OAuthError(
            "Google returned no refresh token. Revoke the prior grant and run OAuth setup again."
        )
    return refresh_token
