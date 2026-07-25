from __future__ import annotations

import unittest

from tenminvideomaker.delivery import (
    DiscordDeliverySettings,
    valid_discord_webhook_url,
)

FAKE_WEBHOOK = "https://discord.com" + "/api/webhooks/123456789/token-value"


class DeliveryTests(unittest.TestCase):
    def test_only_discord_https_webhook_shape_is_accepted(self) -> None:
        self.assertTrue(valid_discord_webhook_url(FAKE_WEBHOOK))
        self.assertFalse(valid_discord_webhook_url("http://discord.com/api/webhooks/1/token"))
        self.assertFalse(valid_discord_webhook_url("https://example.com/api/webhooks/1/token"))
        self.assertFalse(valid_discord_webhook_url("https://discord.com/channels/1/2"))

    def test_environment_loader_requires_valid_configured_webhook(self) -> None:
        with self.assertRaisesRegex(ValueError, "not configured"):
            DiscordDeliverySettings.from_environment({})
        with self.assertRaisesRegex(ValueError, "invalid"):
            DiscordDeliverySettings.from_environment(
                {"TENMIN_DISCORD_WEBHOOK_URL": "https://example.com/not-discord"}
            )
        settings = DiscordDeliverySettings.from_environment(
            {
                "TENMIN_DISCORD_WEBHOOK_URL": FAKE_WEBHOOK
            }
        )
        self.assertEqual(settings.webhook_url, FAKE_WEBHOOK)


if __name__ == "__main__":
    unittest.main()
