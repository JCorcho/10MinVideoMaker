from __future__ import annotations

import unittest

from scripts.setup_and_start import (
    _local_comfy_url,
    configure_gmail,
    edit_optional_settings,
    required_gmail_ready,
)


def _answers(*values: str):
    iterator = iter(values)
    return lambda _prompt: next(iterator)


class SetupAndStartTests(unittest.TestCase):
    def test_required_detection_accepts_app_password_or_refresh_token(self) -> None:
        self.assertTrue(
            required_gmail_ready(
                {
                    "TENMIN_GMAIL_USERNAME": "owner@example.com",
                    "TENMIN_GMAIL_AUTH_MODE": "app_password",
                    "TENMIN_GMAIL_APP_PASSWORD": "sixteencharacters",
                }
            )
        )
        self.assertTrue(
            required_gmail_ready(
                {
                    "TENMIN_GMAIL_USERNAME": "owner@example.com",
                    "TENMIN_GMAIL_AUTH_MODE": "oauth2",
                    "TENMIN_GMAIL_OAUTH_CLIENT_ID": "client-id",
                    "TENMIN_GMAIL_OAUTH_CLIENT_SECRET": "client-secret",
                    "TENMIN_GMAIL_OAUTH_REFRESH_TOKEN": "refresh-token",
                }
            )
        )
        self.assertFalse(required_gmail_ready({"TENMIN_GMAIL_USERNAME": "not-an-email"}))

    def test_app_password_setup_collects_required_values(self) -> None:
        environment = {}
        opened = []
        configure_gmail(
            environment,
            input_func=_answers("owner@example.com", "1", "n"),
            secret_input=lambda _prompt: "abcd efgh ijkl mnop",
            open_url=opened.append,
        )
        self.assertEqual(environment["TENMIN_GMAIL_AUTH_MODE"], "app_password")
        self.assertEqual(environment["TENMIN_GMAIL_APP_PASSWORD"], "abcdefghijklmnop")
        self.assertEqual(environment["TENMIN_GMAIL_RECIPIENT"], "owner@example.com")
        self.assertEqual(opened, [])

    def test_oauth_setup_runs_browser_authorization_and_saves_refresh_details(self) -> None:
        environment = {}
        authorization = {}

        def authorize(**kwargs):
            authorization.update(kwargs)
            return "refresh-token"

        configure_gmail(
            environment,
            input_func=_answers("owner@example.com", "2", "n", "client-id"),
            secret_input=lambda _prompt: "client-secret",
            open_url=lambda _url: None,
            oauth_authorize=authorize,
        )
        self.assertEqual(environment["TENMIN_GMAIL_AUTH_MODE"], "oauth2")
        self.assertEqual(environment["TENMIN_GMAIL_OAUTH_CLIENT_ID"], "client-id")
        self.assertEqual(environment["TENMIN_GMAIL_OAUTH_REFRESH_TOKEN"], "refresh-token")
        self.assertEqual(authorization["login_hint"], "owner@example.com")

    def test_optional_editor_lists_and_changes_selected_value(self) -> None:
        environment = {"TENMIN_GMAIL_USERNAME": "owner@example.com"}
        edit_optional_settings(
            environment,
            input_func=_answers("4", "60", "0"),
        )
        self.assertEqual(environment["TENMIN_POLL_SECONDS"], "60")

    def test_comfy_url_is_restricted_to_the_authorized_local_server(self) -> None:
        self.assertTrue(_local_comfy_url("http://127.0.0.1:8188"))
        self.assertTrue(_local_comfy_url("http://localhost:8188"))
        self.assertFalse(_local_comfy_url("https://example.com:8188"))


if __name__ == "__main__":
    unittest.main()
