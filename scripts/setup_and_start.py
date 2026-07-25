"""Interactive one-click setup and launch for 10MinVideoMaker."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from getpass import getpass
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable
from urllib.parse import urlparse
import webbrowser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tenminvideomaker.comfy_http import ComfyHttpClient
from tenminvideomaker.configuration import (
    ConfigurationError,
    SECRET_KEYS,
    load_project_environment,
    save_project_environment,
)
from tenminvideomaker.drive import GOOGLE_DRIVE_API_ENABLE_URL
from tenminvideomaker.mail import (
    GmailClient,
    GmailSettings,
    MailConfigurationError,
    MailTransportError,
)
from tenminvideomaker.oauth import (
    GOOGLE_CREDENTIALS_URL,
    GOOGLE_OAUTH_SCOPES,
    GOOGLE_OAUTH_SCOPE_VALUE,
    OAuthError,
    authorize_desktop_app,
)
from tenminvideomaker.state_store import PipelineState, PipelineStateStore, SceneState

APP_PASSWORD_URL = "https://myaccount.google.com/apppasswords"
CIVITAI_API_KEYS_URL = "https://civitai.com/user/account"


@dataclass(frozen=True)
class OptionalSetting:
    key: str
    label: str
    default: str
    description: str
    validator: Callable[[str], bool]


def _positive_number(value: str) -> bool:
    try:
        return float(value) > 0
    except ValueError:
        return False


def _positive_integer(value: str) -> bool:
    return value.isdigit() and int(value) > 0


def _nonempty(value: str) -> bool:
    return bool(value.strip())


def _log_level(value: str) -> bool:
    return value.upper() in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _valid_email(value: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value.strip()))


def _valid_sender_list(value: str) -> bool:
    senders = [sender.strip() for sender in value.split(",") if sender.strip()]
    return bool(senders) and all(_valid_email(sender) for sender in senders)


def _local_comfy_url(value: str) -> bool:
    parsed = urlparse(value)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname in {"127.0.0.1", "localhost"}
        and not parsed.username
        and not parsed.password
    )


OPTIONAL_SETTINGS = (
    OptionalSetting(
        "TENMIN_GMAIL_RECIPIENT",
        "Request recipient",
        "",
        "Address that receives the Grok request; blank defaults to the Gmail username.",
        _valid_email,
    ),
    OptionalSetting(
        "TENMIN_GMAIL_ALLOWED_SENDERS",
        "Allowed response senders",
        "",
        "Comma-separated sender addresses; blank defaults to the Gmail username.",
        _valid_sender_list,
    ),
    OptionalSetting(
        "TENMIN_COMFY_URL",
        "ComfyUI URL",
        "http://127.0.0.1:8188",
        "Local ComfyUI HTTP endpoint.",
        _local_comfy_url,
    ),
    OptionalSetting(
        "TENMIN_POLL_SECONDS",
        "Gmail polling seconds",
        "300",
        "Delay between mailbox checks.",
        _positive_number,
    ),
    OptionalSetting(
        "TENMIN_T2I_TIMEOUT_SECONDS",
        "T2I timeout seconds",
        "3600",
        "Maximum time for one T2I prompt.",
        _positive_number,
    ),
    OptionalSetting(
        "TENMIN_I2V_TIMEOUT_SECONDS",
        "I2V timeout seconds",
        "21600",
        "Maximum time for one I2V prompt.",
        _positive_number,
    ),
    OptionalSetting(
        "TENMIN_MAX_STAGE_ATTEMPTS",
        "Maximum stage attempts",
        "2",
        "Persistent retry budget for each scene stage.",
        _positive_integer,
    ),
    OptionalSetting(
        "TENMIN_FFMPEG",
        "FFmpeg command/path",
        "ffmpeg",
        "Executable used for final concat.",
        _nonempty,
    ),
    OptionalSetting(
        "TENMIN_FFPROBE",
        "FFprobe command/path",
        "ffprobe",
        "Executable used for clip preflight.",
        _nonempty,
    ),
    OptionalSetting(
        "TENMIN_LOG_LEVEL",
        "Log level",
        "INFO",
        "DEBUG, INFO, WARNING, ERROR, or CRITICAL.",
        _log_level,
    ),
)


def _yes_no(
    prompt: str,
    *,
    default: bool,
    input_func: Callable[[str], str] = input,
) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        response = input_func(prompt + suffix).strip().casefold()
        if not response:
            return default
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print("Please enter y or n.")


def required_gmail_ready(environment: dict[str, str]) -> bool:
    username = environment.get("TENMIN_GMAIL_USERNAME", "").strip()
    mode = environment.get("TENMIN_GMAIL_AUTH_MODE", "app_password").strip().casefold()
    if not _valid_email(username) or mode not in {"app_password", "oauth2"}:
        return False
    if mode == "app_password":
        return bool(environment.get("TENMIN_GMAIL_APP_PASSWORD", "").strip())
    legacy_token = environment.get("TENMIN_GMAIL_OAUTH2_TOKEN", "").strip()
    refresh_ready = all(
        environment.get(key, "").strip()
        for key in (
            "TENMIN_GMAIL_OAUTH_CLIENT_ID",
            "TENMIN_GMAIL_OAUTH_CLIENT_SECRET",
            "TENMIN_GMAIL_OAUTH_REFRESH_TOKEN",
        )
    )
    return bool(legacy_token or refresh_ready) and oauth_drive_scopes_ready(environment)


def oauth_drive_scopes_ready(environment: dict[str, str]) -> bool:
    configured = frozenset(environment.get("TENMIN_GMAIL_OAUTH_SCOPES", "").split())
    return set(GOOGLE_OAUTH_SCOPES).issubset(configured)


def _prompt_email(
    prompt: str,
    *,
    default: str = "",
    input_func: Callable[[str], str] = input,
) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        value = input_func(f"{prompt}{suffix}: ").strip() or default
        if _valid_email(value):
            return value
        print("Enter a valid email address.")


def _clear_authentication(environment: dict[str, str]) -> None:
    for key in SECRET_KEYS | {
        "TENMIN_GMAIL_OAUTH_CLIENT_ID",
        "TENMIN_GMAIL_OAUTH_SCOPES",
    }:
        environment.pop(key, None)


def configure_gmail(
    environment: dict[str, str],
    *,
    input_func: Callable[[str], str] = input,
    secret_input: Callable[[str], str] = getpass,
    open_url: Callable[[str], object] = webbrowser.open,
    oauth_authorize: Callable[..., str] = authorize_desktop_app,
    force_authentication: bool = False,
) -> None:
    print("\nGmail setup")
    print("-----------")
    username = environment.get("TENMIN_GMAIL_USERNAME", "").strip()
    if force_authentication or not _valid_email(username):
        username = _prompt_email(
            "Gmail address",
            default=username if _valid_email(username) else "",
            input_func=input_func,
        )
    environment["TENMIN_GMAIL_USERNAME"] = username
    environment.setdefault("TENMIN_GMAIL_RECIPIENT", username)
    environment.setdefault("TENMIN_GMAIL_ALLOWED_SENDERS", username)

    current_mode = environment.get("TENMIN_GMAIL_AUTH_MODE", "").casefold()
    if force_authentication or current_mode not in {"app_password", "oauth2"}:
        print("\nChoose Gmail authentication:")
        print("  1. Google App Password")
        print("  2. OAuth2 browser login (persistent refresh token)")
        while True:
            selection = input_func("Selection [1]: ").strip() or "1"
            if selection in {"1", "2"}:
                break
            print("Choose 1 or 2.")
        current_mode = "app_password" if selection == "1" else "oauth2"
        _clear_authentication(environment)
        environment["TENMIN_GMAIL_AUTH_MODE"] = current_mode

    if current_mode == "app_password":
        if force_authentication or not environment.get("TENMIN_GMAIL_APP_PASSWORD", "").strip():
            print("\nGoogle App Password page:")
            print(APP_PASSWORD_URL)
            print("App Passwords require 2-Step Verification on the Google account.")
            if _yes_no("Open the App Password page now?", default=True, input_func=input_func):
                open_url(APP_PASSWORD_URL)
            while True:
                password = secret_input("Paste the 16-character App Password: ").replace(" ", "").strip()
                if len(password) == 16:
                    break
                print("An App Password should contain 16 characters.")
            environment["TENMIN_GMAIL_APP_PASSWORD"] = password
        return

    refresh_ready = all(
        environment.get(key, "").strip()
        for key in (
            "TENMIN_GMAIL_OAUTH_CLIENT_ID",
            "TENMIN_GMAIL_OAUTH_CLIENT_SECRET",
            "TENMIN_GMAIL_OAUTH_REFRESH_TOKEN",
        )
    )
    scopes_ready = oauth_drive_scopes_ready(environment)
    if force_authentication or not refresh_ready or not scopes_ready:
        if refresh_ready and not scopes_ready and not force_authentication:
            print(
                "\nThe saved OAuth grant predates Google Drive handoffs. "
                "A one-time browser reauthorization is required."
            )
            client_id = environment["TENMIN_GMAIL_OAUTH_CLIENT_ID"].strip()
            client_secret = environment["TENMIN_GMAIL_OAUTH_CLIENT_SECRET"].strip()
        else:
            print("\nOAuth2 needs a Google Cloud OAuth client of type Desktop app.")
            print("Google Cloud credentials page:")
            print(GOOGLE_CREDENTIALS_URL)
            if _yes_no("Open Google Cloud credentials now?", default=True, input_func=input_func):
                open_url(GOOGLE_CREDENTIALS_URL)
            print(
                "Create/select a project, configure the OAuth consent screen, add your Gmail as a test user "
                "when the app is in testing, then create a Desktop app OAuth client."
            )
            print(
                "For unattended 24/7 use, publish the consent screen to In production. "
                "External apps left in Testing receive refresh tokens that expire after 7 days."
            )
            client_id = input_func("Paste the Desktop OAuth client ID: ").strip()
            client_secret = secret_input("Paste the Desktop OAuth client secret: ").strip()
        if not client_id or not client_secret:
            raise MailConfigurationError("OAuth client ID and client secret are required.")
        print("\nGoogle Drive API page:")
        print(GOOGLE_DRIVE_API_ENABLE_URL)
        print("The Google Drive API must be enabled in the same project as the Desktop OAuth client.")
        print(
            "The OAuth consent screen's Data Access list must also allow "
            "https://www.googleapis.com/auth/drive.readonly."
        )
        if _yes_no("Open the Google Drive API page now?", default=True, input_func=input_func):
            open_url(GOOGLE_DRIVE_API_ENABLE_URL)
        refresh_token = oauth_authorize(
            client_id=client_id,
            client_secret=client_secret,
            login_hint=username,
        )
        environment["TENMIN_GMAIL_OAUTH_CLIENT_ID"] = client_id
        environment["TENMIN_GMAIL_OAUTH_CLIENT_SECRET"] = client_secret
        environment["TENMIN_GMAIL_OAUTH_REFRESH_TOKEN"] = refresh_token
        environment["TENMIN_GMAIL_OAUTH_SCOPES"] = GOOGLE_OAUTH_SCOPE_VALUE
        environment.pop("TENMIN_GMAIL_OAUTH2_TOKEN", None)


def configure_civitai(
    environment: dict[str, str],
    *,
    input_func: Callable[[str], str] = input,
    secret_input: Callable[[str], str] = getpass,
    open_url: Callable[[str], object] = webbrowser.open,
) -> None:
    print("\nCivitai download authentication")
    print("-------------------------------")
    print(
        "Civitai model metadata is public, but the file endpoint currently requires "
        "account authentication. The token is encrypted with Windows DPAPI."
    )
    print("Civitai account/API keys page:")
    print(CIVITAI_API_KEYS_URL)
    if _yes_no("Open the Civitai account page now?", default=True, input_func=input_func):
        open_url(CIVITAI_API_KEYS_URL)
    print(
        "Sign in, open Account Settings, find API Keys, create a key, then paste it here. "
        "Do not paste the token into chat."
    )
    token = secret_input("Paste the Civitai API token: ").strip()
    if not token:
        raise ConfigurationError("A non-empty Civitai API token is required.")
    environment["TENMIN_CIVITAI_TOKEN"] = token


def edit_optional_settings(
    environment: dict[str, str],
    *,
    input_func: Callable[[str], str] = input,
    secret_input: Callable[[str], str] = getpass,
    open_url: Callable[[str], object] = webbrowser.open,
    oauth_authorize: Callable[..., str] = authorize_desktop_app,
) -> None:
    while True:
        username = environment.get("TENMIN_GMAIL_USERNAME", "")
        print("\nOptional settings")
        print("-----------------")
        for index, setting in enumerate(OPTIONAL_SETTINGS, 1):
            default = username if setting.key in {
                "TENMIN_GMAIL_RECIPIENT",
                "TENMIN_GMAIL_ALLOWED_SENDERS",
            } else setting.default
            current = environment.get(setting.key, "").strip() or default
            print(f"  {index:2}. {setting.label}: {current}")
            print(f"      {setting.description}")
        auth_index = len(OPTIONAL_SETTINGS) + 1
        civitai_index = auth_index + 1
        print(f"  {auth_index:2}. Reconfigure Gmail authentication")
        token_status = (
            "configured"
            if environment.get("TENMIN_CIVITAI_TOKEN", "").strip()
            else "not configured"
        )
        print(f"  {civitai_index:2}. Configure/change Civitai API token ({token_status})")
        print("   0. Save and continue")
        choice = input_func("Choose a setting to change [0]: ").strip() or "0"
        if choice == "0":
            return
        if not choice.isdigit():
            print("Enter one of the listed numbers.")
            continue
        number = int(choice)
        if number == auth_index:
            configure_gmail(
                environment,
                input_func=input_func,
                secret_input=secret_input,
                open_url=open_url,
                oauth_authorize=oauth_authorize,
                force_authentication=True,
            )
            continue
        if number == civitai_index:
            configure_civitai(
                environment,
                input_func=input_func,
                secret_input=secret_input,
                open_url=open_url,
            )
            continue
        if not 1 <= number <= len(OPTIONAL_SETTINGS):
            print("Enter one of the listed numbers.")
            continue
        setting = OPTIONAL_SETTINGS[number - 1]
        username_default = username if setting.key in {
            "TENMIN_GMAIL_RECIPIENT",
            "TENMIN_GMAIL_ALLOWED_SENDERS",
        } else setting.default
        current = environment.get(setting.key, "").strip() or username_default
        value = input_func(f"{setting.label} [{current}]: ").strip() or current
        if not setting.validator(value):
            print(f"Invalid value for {setting.label}.")
            continue
        environment[setting.key] = value.upper() if setting.key == "TENMIN_LOG_LEVEL" else value


def _validate_gmail(environment: dict[str, str]) -> None:
    settings = GmailSettings.from_environment(environment)
    if settings.auth_mode == "oauth2":
        print("\nChecking Gmail SMTP/IMAP and Google Drive access (no email will be sent)...")
    else:
        print("\nChecking Gmail SMTP and IMAP login (no email will be sent)...")
    GmailClient(settings).validate_credentials()
    if settings.auth_mode == "oauth2":
        print("Gmail and Google Drive authentication succeeded.")
    else:
        print("Gmail authentication succeeded. Public Google Drive links are supported.")


def _ensure_comfyui(environment: dict[str, str]) -> None:
    comfy_url = environment.get("TENMIN_COMFY_URL", "http://127.0.0.1:8188")
    client = ComfyHttpClient(comfy_url)
    if client.alive():
        print(f"ComfyUI is healthy at {comfy_url}.")
        return
    parsed = urlparse(comfy_url)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise RuntimeError(f"Remote ComfyUI is unavailable at {comfy_url}; automatic restart is local-only.")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    easy_install_root = PROJECT_ROOT.parents[2]
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(PROJECT_ROOT / "scripts" / "restart_comfyui.ps1"),
        "-EasyInstallRoot",
        str(easy_install_root),
        "-ProjectRuntimeRoot",
        str(PROJECT_ROOT / "runtime"),
        "-Port",
        str(port),
    ]
    print(f"ComfyUI is down at {comfy_url}; starting the verified local server...")
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=240)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Timed out while starting the local ComfyUI server.") from error
    if completed.returncode != 0:
        raise RuntimeError(
            "ComfyUI could not be started: "
            + (completed.stderr.strip() or completed.stdout.strip() or "unknown restart error")
        )
    if not client.alive():
        raise RuntimeError(f"ComfyUI did not become healthy at {comfy_url}.")
    print("ComfyUI started successfully.")


def setup_environment(
    *,
    input_func: Callable[[str], str] = input,
    secret_input: Callable[[str], str] = getpass,
    open_url: Callable[[str], object] = webbrowser.open,
    oauth_authorize: Callable[..., str] = authorize_desktop_app,
) -> dict[str, str]:
    environment = load_project_environment(PROJECT_ROOT)
    had_required = required_gmail_ready(environment)
    if had_required:
        print("Required Gmail environment details are already configured.")
    else:
        print("Some required Gmail environment details are missing.")
        configure_gmail(
            environment,
            input_func=input_func,
            secret_input=secret_input,
            open_url=open_url,
            oauth_authorize=oauth_authorize,
        )

    if environment.get("TENMIN_CIVITAI_TOKEN", "").strip():
        print("Civitai API token is configured.")
    else:
        print(
            "No Civitai API token is configured. It is needed when a job references "
            "a LoRA that is not already installed."
        )
        if _yes_no(
            "Configure a Civitai API token now?",
            default=True,
            input_func=input_func,
        ):
            configure_civitai(
                environment,
                input_func=input_func,
                secret_input=secret_input,
                open_url=open_url,
            )

    if _yes_no(
        "Change optional environment settings before starting?",
        default=False,
        input_func=input_func,
    ):
        edit_optional_settings(
            environment,
            input_func=input_func,
            secret_input=secret_input,
            open_url=open_url,
            oauth_authorize=oauth_authorize,
        )

    username = environment["TENMIN_GMAIL_USERNAME"].strip()
    environment["TENMIN_GMAIL_RECIPIENT"] = (
        environment.get("TENMIN_GMAIL_RECIPIENT", "").strip() or username
    )
    environment["TENMIN_GMAIL_ALLOWED_SENDERS"] = (
        environment.get("TENMIN_GMAIL_ALLOWED_SENDERS", "").strip() or username
    )
    save_project_environment(PROJECT_ROOT, environment)
    return environment


def offer_saved_job_retry(
    *,
    input_func: Callable[[str], str] = input,
) -> str | None:
    store = PipelineStateStore(PROJECT_ROOT / "runtime" / "pipeline.sqlite3")
    snapshot = store.snapshot()
    if not snapshot.job_id or snapshot.state not in {
        PipelineState.ERROR,
        PipelineState.WAITING_FOR_GROK,
    }:
        return None
    records = store.scene_records(snapshot.job_id)
    unfinished = [record for record in records if record.state != SceneState.SUCCEEDED]
    if not unfinished:
        return None
    print(
        f"\nSaved job {snapshot.job_id} has {len(unfinished)} unfinished scene(s)."
    )
    if not _yes_no(
        "Retry this saved job before accepting another email?",
        default=True,
        input_func=input_func,
    ):
        return None
    store.retry_job(snapshot.job_id)
    print(f"Saved job {snapshot.job_id} is queued for asset resolution.")
    return snapshot.job_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Save and validate configuration without starting the supervisor.",
    )
    parser.add_argument(
        "--skip-gmail-check",
        action="store_true",
        help="Skip the SMTP/IMAP and Drive access checks (intended only for offline diagnostics).",
    )
    args = parser.parse_args()

    print("=" * 68)
    print("10MinVideoMaker one-click setup and start")
    print("=" * 68)
    try:
        environment = setup_environment()
        if not args.skip_gmail_check:
            while True:
                try:
                    _validate_gmail(environment)
                    break
                except MailTransportError:
                    if not _yes_no(
                        "Google services validation failed. Reconfigure OAuth/Gmail authentication and retry?",
                        default=True,
                    ):
                        raise
                    configure_gmail(environment, force_authentication=True)
                    save_project_environment(PROJECT_ROOT, environment)
        if args.setup_only:
            print("\nSetup is complete. The supervisor was not started.")
            return 0
        _ensure_comfyui(environment)
        offer_saved_job_retry()
    except (
        ConfigurationError,
        MailConfigurationError,
        MailTransportError,
        OAuthError,
        RuntimeError,
    ) as error:
        print(f"\nSetup could not continue: {error}")
        return 1

    print("\nStarting the 10MinVideoMaker supervisor.")
    print("The first idle tick sends the pipeline request email. Press Ctrl+C to stop.")
    print("LTX generation is VRAM-intensive once a job arrives.")
    child_environment = os.environ.copy()
    child_environment.update(environment)
    supervisor_script = PROJECT_ROOT / "scripts" / "run_supervisor.py"
    os.execve(
        sys.executable,
        [sys.executable, str(supervisor_script)],
        child_environment,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
