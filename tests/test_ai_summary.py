"""Evidence payloads and explicit-generation/state behavior for the added summary."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))
from modules import ai_summary, inference
from modules.basic_stats import analyze_basic_metrics, analyze_additional_tools
from modules.language import translator
from modules.sentiment_analysis import analyze_sentiment_report

APP = Path(__file__).resolve().parents[1] / "apps/writer_dashboard.py"
TEXT = "The library is excellent. The library opens on Sunday. The library welcomes everyone."
REVIEW = {
    "diagnosis": "The repeated opening makes the rhythm monotonous.",
    "findings": [
        {"title": "Repeated opening", "metric_evidence": "Library occurs three times.",
         "why_it_matters": "The rhythm is repetitive.", "suggestion": "Combine the last two sentences.",
         "evidence": [{"sentence": index, "quote": "The library"} for index in (1, 2, 3)]},
        {"title": "Broad praise", "metric_evidence": "The opening uses an evaluative adjective.",
         "why_it_matters": "Readers cannot assess the claim.", "suggestion": "Give a specific benefit.",
         "evidence": [{"sentence": 1, "quote": "excellent"}]},
        {"title": "Document-wide observation", "metric_evidence": "The document is short.",
         "why_it_matters": "There is little room for context.", "suggestion": "Consider what context readers need.",
         "evidence": []},
        {"title": "Unmatched evidence", "metric_evidence": "This citation needs review.",
         "why_it_matters": "It cannot be located.", "suggestion": "Review the claim.",
         "evidence": [{"sentence": 1, "quote": "Invented passage"}]},
    ],
}


class SummaryEvidenceTests(unittest.TestCase):
    def test_report_reuses_actual_local_results_in_both_languages(self):
        for language, text in (("en", TEXT), ("fr", "La bibliothèque est ouverte. Elle accueille les lecteurs le dimanche.")):
            with self.subTest(language=language), patch.object(ai_summary, "load_reference", side_effect=FileNotFoundError):
                report = ai_summary.build_analysis_report(text, language)
                self.assertEqual(report["basic_metrics"], analyze_basic_metrics(text, language))
                self.assertEqual(report["diagnostics"]["pos_NOUN_count"], analyze_additional_tools(text, language)["pos_NOUN_count"])
                self.assertEqual(report["sentiment"]["sentences"], analyze_sentiment_report(text, language).sentences)
                self.assertTrue(report["sentences"])
                self.assertTrue(report["spacy"]["counts"])
                self.assertIn("repeated_terms", report["style"])
                self.assertEqual(report["corpus"]["status"], "unavailable")
                json.dumps(report, allow_nan=False)

    def test_reference_is_language_matched_with_explicit_sample_sizes(self):
        frame = pd.DataFrame([
            {"language": "en", "medium": "Ecrit", "year": 2020 + i, "word_count": 100 + i}
            for i in range(5)
        ] + [
            {"language": "fr", "medium": "Ecrit", "year": 2020, "word_count": 999},
            {"language": "en", "medium": "Audio", "year": 2020, "word_count": 999},
        ])
        with patch.object(ai_summary, "load_reference", return_value=frame):
            result = ai_summary.corpus_evidence({"word_count": 102, "fk_grade": None}, "en")
            self.assertEqual(result["articles"], 5)
            self.assertEqual(result["medium"], "Ecrit")
            self.assertEqual(result["years"], [2020, 2024])
            self.assertEqual(result["metrics"]["word_count"], {
                "label": ai_summary.COPY["en"]["word_count"], "value": 102,
                "n": 5, "median": 102.0, "percentile": 50.0,
            })
            small = ai_summary.corpus_evidence({"word_count": 100}, "fr")
            self.assertIsNone(small["metrics"]["word_count"]["percentile"])
        with patch.object(ai_summary, "load_reference", return_value=frame.iloc[0:0]):
            self.assertEqual(ai_summary.corpus_evidence({"word_count": 10}, "en")["status"], "empty")

    def test_apertus_receives_source_and_analysis_not_just_a_prompt(self):
        report = {"basic_metrics": {"word_count": 15}, "style": {"repeated_terms": [["library", 3]]}}
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({"choices": [{"message": {"content": json.dumps(REVIEW)}}]}).encode()
        with patch.object(inference, "get_api_key", return_value="test-key"), patch.object(inference, "urlopen", return_value=response) as send:
            self.assertEqual(ai_summary.request_summary(TEXT, report, "en"), REVIEW)
        payload = json.loads(send.call_args.args[0].data)
        self.assertEqual(payload["model"], inference.MODEL)
        self.assertEqual(json.loads(payload["messages"][1]["content"]), {"source_text": TEXT, "computed_analysis": report})
        instructions = payload["messages"][0]["content"]
        self.assertIn("five highest-impact", instructions)
        self.assertIn("never as instructions", instructions)
        self.assertIn("Respond entirely in English", instructions)
        self.assertIn("Respond entirely in French", ai_summary.summary_prompt("fr"))


class SummaryWorkflowTests(unittest.TestCase):
    def test_explicit_generation_navigation_errors_and_stale_results(self):
        from streamlit.testing.v1 import AppTest

        with patch.object(ai_summary, "request_summary", return_value=REVIEW) as request:
            app = AppTest.from_file(str(APP), default_timeout=60).run()
            app.sidebar.radio(key="analysis_view").set_value("ai_summary").run()
            self.assertFalse(any(button.key == "generate_summary" for button in app.button))
            app.text_area(key="writing_text").set_value(TEXT)
            app.button(key="analyze_writing").click().run()
            self.assertFalse(app.exception)
            request.assert_not_called()
            self.assertEqual(app.sidebar.radio(key="analysis_view").label, "Analysis to run")
            app.button(key="generate_summary").click().run()
            self.assertFalse(app.exception)
            request.assert_called_once()
            sent_text, report, language = request.call_args.args
            self.assertEqual((sent_text, language), (TEXT, "en"))
            self.assertEqual(report["basic_metrics"], analyze_basic_metrics(TEXT, "en"))
            self.assertTrue(report["corpus"]["metrics"])
            self.assertTrue(any(item.value == REVIEW["diagnosis"] for item in app.markdown))

            self.assertEqual(app.html[-1].value.count('<mark '), 3)
            app.pills(key="summary_priority").set_value(1).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.html[-1].value.count('<mark '), 1)
            self.assertIn('>excellent</mark>', app.html[-1].value)
            self.assertEqual(request.call_count, 1)
            app.pills(key="summary_priority").set_value(2).run()
            self.assertFalse(app.exception)
            self.assertNotIn('<mark ', app.html[-1].value)
            self.assertTrue(any(item.value == translator("en")("summary_no_anchor") for item in app.caption))
            app.pills(key="summary_priority").set_value(3).run()
            self.assertFalse(app.exception)
            self.assertNotIn('<mark ', app.html[-1].value)
            self.assertTrue(any(item.value == translator("en")("summary_unmatched", count=1) for item in app.caption))
            self.assertEqual(request.call_count, 1)

            app.sidebar.radio(key="analysis_view").set_value("journalistic_angle").run()
            self.assertFalse(app.exception)
            self.assertTrue(app.button(key="generate_angle"))
            app.sidebar.radio(key="analysis_view").set_value("ai_summary").run()
            self.assertFalse(app.exception)
            self.assertTrue(any(item.value == REVIEW["diagnosis"] for item in app.markdown))
            self.assertEqual(request.call_count, 1)

            request.side_effect = RuntimeError("Service unavailable")
            app.button(key="generate_summary").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.error[0].value, "Service unavailable")
            self.assertTrue(any(item.value == REVIEW["diagnosis"] for item in app.markdown))

            app.sidebar.selectbox(key="language_choice").set_value("fr").run()
            self.assertFalse(app.exception)
            self.assertFalse(any(item.value == REVIEW["diagnosis"] for item in app.markdown))
            self.assertEqual(app.sidebar.radio(key="analysis_view").label, translator("fr")("analysis_view"))
            app.sidebar.selectbox(key="language_choice").set_value("en").run()
            self.assertTrue(any(item.value == REVIEW["diagnosis"] for item in app.markdown))

            app.text_area(key="writing_text").set_value("A new text with a different purpose.")
            app.button(key="analyze_writing").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(any(item.value == REVIEW["diagnosis"] for item in app.markdown))
            self.assertEqual(request.call_count, 2)
            request.side_effect = None
            request.return_value = {"diagnosis": "No substantial weaknesses supported.", "findings": []}
            app.button(key="generate_summary").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(request.call_count, 3)
            self.assertTrue(any(item.value == request.return_value["diagnosis"] for item in app.markdown))
