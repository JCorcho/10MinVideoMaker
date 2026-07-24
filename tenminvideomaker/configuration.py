"""Project-local environment configuration with Windows DPAPI secret storage."""

from __future__ import annotations

import base64
import binascii
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
from typing import Callable, Mapping

SECRET_KEYS = frozenset(
    {
        "TENMIN_GMAIL_APP_PASSWORD",
        "TENMIN_GMAIL_OAUTH2_TOKEN",
        "TENMIN_GMAIL_OAUTH_CLIENT_SECRET",
        "TENMIN_GMAIL_OAUTH_REFRESH_TOKEN",
    }
)

CONFIG_KEYS = frozenset(
    {
        "TENMIN_GMAIL_USERNAME",
        "TENMIN_GMAIL_RECIPIENT",
        "TENMIN_GMAIL_ALLOWED_SENDERS",
        "TENMIN_GMAIL_AUTH_MODE",
        "TENMIN_GMAIL_OAUTH_CLIENT_ID",
        "TENMIN_COMFY_URL",
        "TENMIN_POLL_SECONDS",
        "TENMIN_T2I_TIMEOUT_SECONDS",
        "TENMIN_I2V_TIMEOUT_SECONDS",
        "TENMIN_MAX_STAGE_ATTEMPTS",
        "TENMIN_FFMPEG",
        "TENMIN_FFPROBE",
        "TENMIN_LOG_LEVEL",
    }
)


class ConfigurationError(RuntimeError):
    """Raised when local configuration or encrypted secrets are invalid."""


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _protect_dpapi(value: bytes) -> bytes:
    if os.name != "nt":
        raise ConfigurationError("Windows DPAPI secret storage is only available on Windows.")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        wintypes.LPCWSTR,
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    buffer = ctypes.create_string_buffer(value)
    source = _DataBlob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
    protected = _DataBlob()
    if not crypt32.CryptProtectData(
        ctypes.byref(source),
        "10MinVideoMaker",
        None,
        None,
        None,
        0x1,
        ctypes.byref(protected),
    ):
        raise ConfigurationError(f"Windows DPAPI encryption failed: {ctypes.get_last_error()}")
    try:
        return ctypes.string_at(protected.pbData, protected.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(protected.pbData, ctypes.c_void_p))


def _unprotect_dpapi(value: bytes) -> bytes:
    if os.name != "nt":
        raise ConfigurationError("Windows DPAPI secret storage is only available on Windows.")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    buffer = ctypes.create_string_buffer(value)
    source = _DataBlob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
    clear = _DataBlob()
    if not crypt32.CryptUnprotectData(
        ctypes.byref(source),
        None,
        None,
        None,
        None,
        0x1,
        ctypes.byref(clear),
    ):
        raise ConfigurationError(f"Windows DPAPI decryption failed: {ctypes.get_last_error()}")
    try:
        return ctypes.string_at(clear.pbData, clear.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(clear.pbData, ctypes.c_void_p))


class SecretStore:
    def __init__(
        self,
        path: str | Path,
        *,
        protect: Callable[[bytes], bytes] = _protect_dpapi,
        unprotect: Callable[[bytes], bytes] = _unprotect_dpapi,
    ):
        self.path = Path(path)
        self._protect = protect
        self._unprotect = unprotect

    def load(self) -> dict[str, str]:
        if not self.path.is_file():
            return {}
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
            values = document["values"]
            if document.get("version") != 1 or not isinstance(values, Mapping):
                raise ValueError("unsupported secret-store format")
            result = {}
            for key, encoded in values.items():
                if key not in SECRET_KEYS or not isinstance(encoded, str):
                    continue
                encrypted = base64.b64decode(encoded, validate=True)
                result[key] = self._unprotect(encrypted).decode("utf-8")
            return result
        except (
            OSError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
            UnicodeError,
            binascii.Error,
        ) as error:
            raise ConfigurationError(f"Could not read encrypted secret store: {error}") from error

    def save(self, values: Mapping[str, str]) -> None:
        serialized = {
            key: base64.b64encode(self._protect(value.encode("utf-8"))).decode("ascii")
            for key, value in values.items()
            if key in SECRET_KEYS and value
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"version": 1, "values": serialized}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


def read_env_file(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigurationError(f"{path.name}:{line_number} is not KEY=VALUE.")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key not in CONFIG_KEYS:
            continue
        raw_value = raw_value.strip()
        try:
            value = json.loads(raw_value) if raw_value.startswith('"') else raw_value
        except json.JSONDecodeError as error:
            raise ConfigurationError(f"{path.name}:{line_number} has invalid quoting.") from error
        if not isinstance(value, str):
            raise ConfigurationError(f"{path.name}:{line_number} value must be text.")
        result[key] = value
    return result


def write_env_file(path: str | Path, values: Mapping[str, str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 10MinVideoMaker local configuration. Secrets are stored separately with Windows DPAPI.",
        *[
            f"{key}={json.dumps(value, ensure_ascii=False)}"
            for key, value in sorted(values.items())
            if key in CONFIG_KEYS and value
        ],
        "",
    ]
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def load_project_environment(
    project_root: str | Path,
    *,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Merge local config and DPAPI secrets, with explicit OS variables taking precedence."""
    project_root = Path(project_root)
    local = read_env_file(project_root / ".env")
    secrets_values = SecretStore(project_root / "runtime" / "secrets.json").load()
    merged = {**local, **secrets_values}
    merged.update(dict(os.environ if base_environment is None else base_environment))
    return merged


def save_project_environment(project_root: str | Path, values: Mapping[str, str]) -> None:
    project_root = Path(project_root)
    write_env_file(
        project_root / ".env",
        {key: value for key, value in values.items() if key in CONFIG_KEYS},
    )
    SecretStore(project_root / "runtime" / "secrets.json").save(
        {key: value for key, value in values.items() if key in SECRET_KEYS}
    )
