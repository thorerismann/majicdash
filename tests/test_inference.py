"""Regression coverage for the CSCS client and the journalistic angle workflow."""

import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

from modules import inference
from modules.language import translator


FRENCH = "La bibliothèque municipale ouvrira le dimanche dès novembre."
ENGLISH = "The public library will open every Sunday starting in November."
APP_PATH = Path(__file__).resolve().parents[1] / "apps" / "writer_dashboard.py"


def api_response(payload):
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(payload).encode("utf-8")
    return response


class InferenceClientTests(unittest.TestCase):
    def test_request_uses_text_prompt_and_configured_key(self):
        response = api_response({"choices": [{"message": {"content": "  Un accès élargi à la lecture.  "}}]})
        with patch.object(inference, "get_api_key", return_value="test-key"), patch.object(
            inference, "urlopen", return_value=response
        ) as send:
            result = inference.request_angle_analysis(FRENCH, "  Proposez un angle.  ", "fr")

        self.assertEqual(result, "Un accès élargi à la lecture.")
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, inference.API_URL)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], inference.MODEL)
        self.assertEqual(payload["messages"], [
            {"role": "system", "content": "Proposez un angle."},
            {"role": "user", "content": FRENCH},
        ])
        self.assertEqual(send.call_args.kwargs["timeout"], 120)

    def test_empty_input_and_missing_key_do_not_send_requests(self):
        for language in ("fr", "en"):
            for text, prompt, key, error in (
                (" ", "Prompt", "test-key", "empty_request"),
                (FRENCH, " ", "test-key", "empty_prompt"),
                (FRENCH, "Prompt", "", "api_key_missing"),
            ):
                with self.subTest(language=language, error=error), patch.object(
                    inference, "get_api_key", return_value=key
                ), patch.object(inference, "urlopen") as send:
                    with self.assertRaises(RuntimeError) as raised:
                        inference.request_angle_analysis(text, prompt, language)
                    self.assertEqual(str(raised.exception), translator(language)(error))
                    send.assert_not_called()

    def test_malformed_and_empty_responses_are_reported(self):
        payloads = [None, {}, {"choices": []}]
        payloads.extend({"choices": [{"message": {"content": value}}]} for value in (None, "", "  ", [], 42))
        for payload in payloads:
            with self.subTest(payload=payload), patch.object(inference, "get_api_key", return_value="test-key"), patch.object(
                inference, "urlopen", return_value=api_response(payload)
            ):
                with self.assertRaises(RuntimeError) as raised:
                    inference.request_angle_analysis(FRENCH, "Prompt", "en")
                self.assertEqual(str(raised.exception), translator("en")("api_no_analysis"))

    def test_invalid_json_and_transport_errors_are_localised(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"not JSON"
        with patch.object(inference, "get_api_key", return_value="test-key"), patch.object(
            inference, "urlopen", return_value=response
        ):
            with self.assertRaises(RuntimeError) as raised:
                inference.request_angle_analysis(FRENCH, "Prompt", "fr")
            self.assertEqual(str(raised.exception), translator("fr")("api_invalid"))

        for error, expected in (
            (HTTPError(inference.API_URL, 401, "Unauthorized", {}, io.BytesIO()), "HTTP 401"),
            (URLError("connection refused"), "connection refused"),
            (TimeoutError(), translator("fr")("api_timeout")),
        ):
            with self.subTest(error=error), patch.object(inference, "get_api_key", return_value="test-key"), patch.object(
                inference, "urlopen", side_effect=error
            ):
                with self.assertRaises(RuntimeError) as raised:
                    inference.request_angle_analysis(FRENCH, "Prompt", "fr")
                self.assertIn(expected, str(raised.exception))


class InferenceTabTests(unittest.TestCase):
    def test_generate_edit_retry_and_language_switch_in_dashboard(self):
        from streamlit.testing.v1 import AppTest

        with patch.object(inference, "request_angle_analysis", return_value="Premier résultat.") as request:
            dashboard = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
            self.assertFalse(dashboard.exception)
            dashboard.text_area(key="writing_text").set_value(FRENCH)
            dashboard.button(key="analyze_writing").click().run()
            self.assertFalse(dashboard.exception)
            self.assertEqual(len(dashboard.tabs), 6)
            self.assertEqual(dashboard.tabs[5].label, translator("fr")("journalistic_angle"))
            request.assert_not_called()

            dashboard.text_area(key="angle_prompt_fr").set_value("Mon angle personnalisé.")
            dashboard.button(key="generate_angle").click().run()
            self.assertFalse(dashboard.exception)
            self.assertFalse(dashboard.error)
            request.assert_called_once_with(FRENCH, "Mon angle personnalisé.", "fr")
            self.assertEqual(dashboard.text_area(key="angle_result_fr").value, "Premier résultat.")

            dashboard.text_area(key="angle_result_fr").set_value("Résultat corrigé.").run()
            self.assertEqual(request.call_count, 1)
            request.side_effect = RuntimeError("Service temporairement indisponible.")
            dashboard.button(key="generate_angle").click().run()
            self.assertFalse(dashboard.exception)
            self.assertEqual(dashboard.error[0].value, "Service temporairement indisponible.")
            self.assertEqual(dashboard.text_area(key="angle_result_fr").value, "Résultat corrigé.")

            request.side_effect = None
            request.return_value = "Nouveau résultat."
            dashboard.button(key="generate_angle").click().run()
            self.assertFalse(dashboard.exception)
            self.assertEqual(dashboard.text_area(key="angle_result_fr").value, "Nouveau résultat.")

            dashboard.text_area(key="writing_text").set_value(ENGLISH)
            dashboard.button(key="analyze_writing").click().run()
            self.assertFalse(dashboard.exception)
            self.assertEqual(dashboard.text_area(key="angle_result_en").value, "")
            self.assertEqual(request.call_count, 3)
            dashboard.text_area(key="writing_text").set_value(FRENCH)
            dashboard.button(key="analyze_writing").click().run()
            self.assertEqual(dashboard.text_area(key="angle_prompt_fr").value, "Mon angle personnalisé.")
            self.assertEqual(dashboard.text_area(key="angle_result_fr").value, "Nouveau résultat.")

            dashboard.text_area(key="angle_prompt_fr").set_value(" ")
            dashboard.button(key="generate_angle").click().run()
            self.assertFalse(dashboard.exception)
            self.assertEqual(dashboard.warning[0].value, translator("fr")("empty_prompt"))
            self.assertEqual(request.call_count, 3)


if __name__ == "__main__":
    unittest.main()
