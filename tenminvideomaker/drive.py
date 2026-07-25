"""Restricted Google Drive file-link parsing and bounded JSON downloads."""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import json
import re
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlencode, urlparse
from urllib.request import Request, urlopen

GOOGLE_DRIVE_API_ENABLE_URL = (
    "https://console.cloud.google.com/apis/library/drive.googleapis.com"
)
MAX_DRIVE_JSON_BYTES = 5 * 1024 * 1024
_URL_PATTERN = re.compile(r"https://[^\s<>\"']+", re.IGNORECASE)
_FILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{10,}$")


class DriveDownloadError(RuntimeError):
    """Raised when a Drive handoff cannot be downloaded safely."""


class _DriveAccessError(DriveDownloadError):
    """Internal signal that authenticated access may succeed."""


@dataclass(frozen=True)
class DriveFileReference:
    file_id: str
    resource_key: str = ""


def parse_google_drive_file_url(url: str) -> DriveFileReference | None:
    """Return a file reference for supported Drive file URLs, never folder URLs."""
    try:
        parsed = urlparse(unescape(url).rstrip(".,;:!?)]}"))
    except ValueError:
        return None
    if parsed.scheme.casefold() != "https" or (parsed.hostname or "").casefold() != "drive.google.com":
        return None
    query = parse_qs(parsed.query)
    parts = [part for part in parsed.path.split("/") if part]
    file_id = ""
    if len(parts) >= 3 and parts[0].casefold() == "file" and parts[1].casefold() == "d":
        file_id = parts[2]
    elif parts and parts[0].casefold() in {"open", "uc"}:
        file_id = (query.get("id") or [""])[0]
    if not _FILE_ID_PATTERN.fullmatch(file_id):
        return None
    resource_key = (query.get("resourcekey") or [""])[0]
    if resource_key and not _FILE_ID_PATTERN.fullmatch(resource_key):
        resource_key = ""
    return DriveFileReference(file_id=file_id, resource_key=resource_key)


def google_drive_file_urls(text: str) -> tuple[str, ...]:
    """Extract unique, supported Google Drive file links from plain text or HTML."""
    result: list[str] = []
    seen: set[str] = set()
    for raw_url in _URL_PATTERN.findall(unescape(text)):
        url = raw_url.rstrip(".,;:!?)]}")
        reference = parse_google_drive_file_url(url)
        if reference is None:
            continue
        identity = f"{reference.file_id}:{reference.resource_key}"
        if identity not in seen:
            seen.add(identity)
            result.append(url)
    return tuple(result)


def _allowed_download_host(hostname: str) -> bool:
    host = hostname.casefold().rstrip(".")
    return host in {
        "drive.google.com",
        "drive.usercontent.google.com",
        "www.googleapis.com",
    } or host.endswith(".googleusercontent.com")


def _read_response(response: Any, *, max_bytes: int) -> tuple[bytes, str]:
    final_url = response.geturl()
    parsed = urlparse(final_url)
    if parsed.scheme != "https" or not _allowed_download_host(parsed.hostname or ""):
        raise DriveDownloadError("Google Drive redirected the download outside an approved Google host.")
    length = response.headers.get("Content-Length") if response.headers else None
    try:
        if length is not None and int(length) > max_bytes:
            raise DriveDownloadError(
                f"Google Drive JSON exceeds the {max_bytes // (1024 * 1024)} MiB safety limit."
            )
    except ValueError:
        pass
    content = response.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise DriveDownloadError(
            f"Google Drive JSON exceeds the {max_bytes // (1024 * 1024)} MiB safety limit."
        )
    content_type = response.headers.get("Content-Type", "") if response.headers else ""
    return content, content_type.casefold()


def _open_request(
    request: Request,
    *,
    opener: Callable[..., Any],
    max_bytes: int,
) -> tuple[bytes, str]:
    try:
        with opener(request, timeout=30) as response:
            return _read_response(response, max_bytes=max_bytes)
    except HTTPError as error:
        raise _DriveAccessError(f"Google Drive returned HTTP {error.code}.") from error
    except (OSError, URLError, ValueError) as error:
        raise _DriveAccessError(f"Google Drive download failed: {error}") from error


def _decode_download(content: bytes, content_type: str) -> str:
    if "text/html" in content_type or content.lstrip().lower().startswith(
        (b"<!doctype html", b"<html")
    ):
        raise _DriveAccessError("Google Drive returned a sign-in or sharing page instead of JSON.")
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise DriveDownloadError("Google Drive file is not UTF-8 JSON text.") from error


def download_google_drive_json(
    share_url: str,
    *,
    access_token: str = "",
    opener: Callable[..., Any] = urlopen,
    max_bytes: int = MAX_DRIVE_JSON_BYTES,
) -> str:
    """Download one Drive file, trying public sharing before authenticated Drive API."""
    reference = parse_google_drive_file_url(share_url)
    if reference is None:
        raise DriveDownloadError("The email does not contain a supported Google Drive file link.")
    public_parameters = {
        "export": "download",
        "confirm": "t",
        "id": reference.file_id,
    }
    if reference.resource_key:
        public_parameters["resourcekey"] = reference.resource_key
    public_url = "https://drive.usercontent.google.com/download?" + urlencode(public_parameters)
    public_request = Request(public_url, headers={"User-Agent": "10MinVideoMaker/1.0"})
    public_error: _DriveAccessError | None = None
    try:
        content, content_type = _open_request(
            public_request,
            opener=opener,
            max_bytes=max_bytes,
        )
        return _decode_download(content, content_type)
    except _DriveAccessError as error:
        public_error = error

    if not access_token:
        raise DriveDownloadError(
            "Google Drive could not download the file anonymously. Share it with "
            "'Anyone with the link' or use OAuth2 with Drive read-only access."
        ) from public_error

    api_url = (
        "https://www.googleapis.com/drive/v3/files/"
        f"{quote(reference.file_id, safe='')}?alt=media"
    )
    authenticated_request = Request(
        api_url,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json, application/octet-stream",
            "User-Agent": "10MinVideoMaker/1.0",
        },
    )
    try:
        content, content_type = _open_request(
            authenticated_request,
            opener=opener,
            max_bytes=max_bytes,
        )
        return _decode_download(content, content_type)
    except DriveDownloadError as error:
        raise DriveDownloadError(
            "Authenticated Google Drive download failed. Enable the Google Drive API in the OAuth "
            "project, share the file with this Gmail account, and reauthorize Drive read-only access."
        ) from error


def validate_google_drive_access(
    access_token: str,
    *,
    opener: Callable[..., Any] = urlopen,
) -> None:
    """Confirm the OAuth grant and Cloud project can call the read-only Drive API."""
    request = Request(
        "https://www.googleapis.com/drive/v3/about?fields=user",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "User-Agent": "10MinVideoMaker/1.0",
        },
    )
    try:
        content, _ = _open_request(request, opener=opener, max_bytes=1024 * 1024)
        document = json.loads(content)
    except (DriveDownloadError, json.JSONDecodeError) as error:
        raise DriveDownloadError(
            "Google Drive OAuth validation failed. Enable the Google Drive API and reauthorize "
            "10MinVideoMaker with Drive read-only access."
        ) from error
    if not isinstance(document, Mapping) or not isinstance(document.get("user"), Mapping):
        raise DriveDownloadError("Google Drive OAuth validation returned an invalid response.")
