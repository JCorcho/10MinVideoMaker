"""Discord delivery configuration shared by setup, supervisor, and workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlparse

DISCORD_WEBHOOK_ENV = "TENMIN_DISCORD_WEBHOOK_URL"
TEMPLATE_WEBHOOK_PLACEHOLDER = (
    "https://discord.com"
    + "/api/webhooks/000000000000000000/REPLACE_WITH_ENCRYPTED_RUNTIME_WEBHOOK"
)

_DISCORD_WEBHOOK_HOSTS = frozenset(
    {
        "discord.com",
        "discordapp.com",
        "canary.discord.com",
        "ptb.discord.com",
    }
)


def valid_discord_webhook_url(value: str) -> bool:
    """Accept only Discord's HTTPS webhook endpoint shape."""
    parsed = urlparse(value.strip())
    parts = [part for part in parsed.path.split("/") if part]
    return (
        parsed.scheme == "https"
        and parsed.hostname in _DISCORD_WEBHOOK_HOSTS
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
        and len(parts) == 4
        and parts[:2] == ["api", "webhooks"]
        and parts[2].isdigit()
        and bool(parts[3])
    )


@dataclass(frozen=True)
class DiscordDeliverySettings:
    webhook_url: str

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str],
        *,
        required: bool = True,
    ) -> "DiscordDeliverySettings | None":
        value = environment.get(DISCORD_WEBHOOK_ENV, "").strip()
        if not value:
            if required:
                raise ValueError(
                    "Discord delivery is not configured. Run Start 10MinVideoMaker.bat "
                    "and enter the Discord webhook."
                )
            return None
        if not valid_discord_webhook_url(value):
            raise ValueError("The configured Discord webhook URL is invalid.")
        return cls(value)
